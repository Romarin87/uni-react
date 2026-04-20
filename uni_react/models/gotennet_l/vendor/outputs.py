"""Official GotenNet QM9 output heads adapted for local imports."""

from typing import Optional, Union

import ase
import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch.autograd import grad
from torch_geometric.utils import scatter

from .gotennet_layers import (
    Dense,
    GetItem,
    ScaleShift,
    SchnetMLP,
    shifted_softplus,
    str2act,
)
from . import utils

log = utils.get_logger(__name__)


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation=F.silu,
        sactivation=None,
    ):
        super().__init__()
        self.n_vout = n_vout
        self.mix_vectors = Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        vmix = self.mix_vectors(vectors)
        vectors_v, vectors_w = torch.split(vmix, self.n_vout, dim=-1)
        vectors_vn = torch.norm(vectors_v, dim=-2)

        ctx = torch.cat([scalars, vectors_vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [x.shape[-1] - self.n_vout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_w

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


class Atomwise(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: Optional[str] = "sum",
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation=shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        derivative: Optional[str] = None,
        negative_dr: bool = True,
        create_graph: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        atomref: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
        return_vector: Optional[str] = None,
        standardize: bool = True,
    ):
        super().__init__()

        self.return_vector = return_vector
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        if type(activation) is str:
            activation = str2act(activation)

        self.atomref = (
            nn.Embedding.from_pretrained(atomref.type(torch.float32))
            if atomref is not None
            else None
        )
        self.equivariant = False
        self.out_net = (
            nn.Sequential(GetItem("representation"), SchnetMLP(n_in, n_out, n_hidden, n_layers, activation))
            if outnet is None
            else outnet
        )

        if self.standardize and (mean is not None and stddev is not None):
            log.info(f"Using standardization with mean {mean} and stddev {stddev}")
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()

        self.aggregation_mode = aggregation_mode

    def forward(self, inputs):
        atomic_numbers = inputs.z
        result = {}

        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation
            for eqlayer in self.out_net:
                l0, l1 = eqlayer(l0, l1)
            if self.return_vector:
                result[self.return_vector] = l1
            yi = l0
        else:
            yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            yi = yi + self.atomref(atomic_numbers)

        y = (
            torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)
            if self.aggregation_mode is not None
            else yi
        )
        result[self.property] = y

        if self.contributions:
            result[self.contributions] = yi

        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=self.create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy
        return result


class Dipole(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[int] = None,
        activation=F.silu,
        property: str = "dipole",
        predict_magnitude: bool = False,
        output_v: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.stddev = stddev
        self.mean = mean
        self.output_v = output_v
        n_hidden = n_in if n_hidden is None else n_hidden

        self.property = property
        self.derivative = None
        self.predict_magnitude = predict_magnitude
        self.equivariant_layers = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_in,
                    n_sout=n_hidden,
                    n_vout=n_hidden,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation,
                ),
                GatedEquivariantBlock(
                    n_sin=n_hidden,
                    n_vin=n_hidden,
                    n_sout=1,
                    n_vout=1,
                    n_hidden=n_hidden,
                    activation=activation,
                ),
            ]
        )
        self.aggregation_mode = "sum"

    def forward(self, inputs):
        positions = inputs.pos
        l0 = inputs.representation
        l1 = inputs.vector_representation[:, :3, :]

        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        if self.stddev is not None:
            l0 = self.stddev * l0 + self.mean

        atomic_dipoles = torch.squeeze(l1, -1)
        charges = l0
        dipole_offsets = positions * charges
        y = atomic_dipoles + dipole_offsets
        y = torch_scatter.scatter(y, inputs.batch, dim=0, reduce=self.aggregation_mode)

        result = {self.property: torch.norm(y, dim=1, keepdim=True) if self.predict_magnitude else y}
        if self.output_v:
            y_vector = torch_scatter.scatter(l1, inputs.batch, dim=0, reduce=self.aggregation_mode)
            result[self.property + "_vector"] = y_vector
        return result


class ElectronicSpatialExtentV2(Atomwise):
    def __init__(
        self,
        n_in: int,
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation=shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
    ):
        super().__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_hidden,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            contributions=contributions,
        )
        self.register_buffer("atomic_mass", torch.from_numpy(ase.data.atomic_masses).float())

    def forward(self, inputs):
        positions = inputs.pos
        x = self.out_net(inputs)
        mass = self.atomic_mass[inputs.z].view(-1, 1)
        c = scatter(mass * positions, inputs.batch, dim=0) / scatter(mass, inputs.batch, dim=0)

        yi = torch.norm(positions - c[inputs.batch], dim=1, keepdim=True)
        yi = yi ** 2 * x
        y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)

        result = {self.property: y}
        if self.contributions:
            result[self.contributions] = x
        return result
