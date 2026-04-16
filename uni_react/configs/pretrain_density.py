"""Dataclass schema for electron-density pretraining runs."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DensityPretrainConfig:
    """All hyper-parameters for an electron-density pretraining run."""

    # ------------------------------------------------------------------
    # Training mode / encoder
    # ------------------------------------------------------------------
    train_mode: str = "electron_density"
    encoder_type: str = "single_mol"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_h5: List[str] = field(default_factory=list)
    val_h5: Optional[List[str]] = None
    batch_size: int = 16
    num_workers: int = 16
    num_query_points: int = 2048
    center_coords: bool = True
    mask_ratio: float = 0.0
    noise_std: float = 0.0
    no_center_coords: bool = False
    no_recenter_noisy: bool = False

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    emb_dim: int = 256
    inv_layer: int = 2
    se3_layer: int = 4
    heads: int = 8
    atom_vocab_size: int = 128
    cutoff: float = 5.0
    num_kernel: int = 128
    path_dropout: float = 0.1
    activation_dropout: float = 0.1
    attn_dropout: float = 0.1
    point_hidden_dim: int = 128
    cond_hidden_dim: int = 64
    head_hidden_dim: int = 512
    radial_sigma: float = 1.5

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    epochs: int = 20
    descriptor_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    lr_scheduler: str = "cosine"

    # ------------------------------------------------------------------
    # Compatibility fields from YAMLs / old scripts
    # ------------------------------------------------------------------
    lr: Optional[float] = None
    warmup_steps: int = 0
    density_grid_size: int = 64
    density_weight: float = 1.0
    save_optimizer: bool = True

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "runs/single_mol_density"
    save_every: int = 5
    log_interval: int = 100
    log_file: str = "train.log"
    init_ckpt: Optional[str] = None
    init_strict: bool = False
    restart: str = ""
    restart_ignore_config: bool = False

    def __post_init__(self) -> None:
        if self.encoder_type not in {"single_mol", "reacformer_se3", "reacformer_so2", "reacformer_hybrid"}:
            raise ValueError(
                "encoder_type must be one of "
                "single_mol/reacformer_se3/reacformer_so2/reacformer_hybrid, "
                f"got {self.encoder_type!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_query_points <= 0:
            raise ValueError(f"num_query_points must be > 0, got {self.num_query_points}")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {self.mask_ratio}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {self.emb_dim}")
        if self.inv_layer < 1:
            raise ValueError(f"inv_layer must be >= 1, got {self.inv_layer}")
        if self.se3_layer < 0:
            raise ValueError(f"se3_layer must be >= 0, got {self.se3_layer}")
        if self.heads <= 0:
            raise ValueError(f"heads must be > 0, got {self.heads}")
        if self.emb_dim % self.heads != 0:
            raise ValueError(f"emb_dim ({self.emb_dim}) must be divisible by heads ({self.heads})")
        if self.atom_vocab_size <= 0:
            raise ValueError(f"atom_vocab_size must be > 0, got {self.atom_vocab_size}")
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}")
        if self.num_kernel <= 0:
            raise ValueError(f"num_kernel must be > 0, got {self.num_kernel}")
        if not 0.0 <= self.path_dropout <= 1.0:
            raise ValueError(f"path_dropout must be in [0, 1], got {self.path_dropout}")
        if not 0.0 <= self.activation_dropout <= 1.0:
            raise ValueError(f"activation_dropout must be in [0, 1], got {self.activation_dropout}")
        if not 0.0 <= self.attn_dropout <= 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1], got {self.attn_dropout}")
        if self.point_hidden_dim <= 0 or self.cond_hidden_dim <= 0 or self.head_hidden_dim <= 0:
            raise ValueError("point_hidden_dim/cond_hidden_dim/head_hidden_dim must all be > 0")
        if self.radial_sigma <= 0:
            raise ValueError(f"radial_sigma must be > 0, got {self.radial_sigma}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.descriptor_lr <= 0:
            raise ValueError(f"descriptor_lr must be > 0, got {self.descriptor_lr}")
        if self.head_lr <= 0:
            raise ValueError(f"head_lr must be > 0, got {self.head_lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.lr_scheduler not in {"none", "cosine"}:
            raise ValueError(f"lr_scheduler must be one of none/cosine, got {self.lr_scheduler!r}")
        if self.save_every <= 0:
            raise ValueError(f"save_every must be > 0, got {self.save_every}")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
