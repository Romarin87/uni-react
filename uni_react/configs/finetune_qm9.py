"""Dataclass schema for QM9 fine-tuning runs."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FinetuneQM9Config:
    """All hyper-parameters for a QM9 fine-tuning run."""

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root: str = "qm9_pyg"
    split: str = "egnn"
    """Dataset split strategy. See ``QM9_SPLIT_MODES`` in the dataset module."""
    force_reload: bool = False
    target: str = "gap"
    """Single-target shorthand; overridden when ``targets`` is non-empty."""
    targets: Optional[List[str]] = None
    """Multi-target list, e.g. ``[gap, homo, lumo]``; use ``[all]`` for all 12 standard targets."""
    batch_size: int = 256
    num_workers: int = 16
    no_center_coords: bool = False

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    encoder_type: str = "single_mol"
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
    head_hidden_dim: int = 256
    head_dropout: float = 0.1

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    epochs: int = 100
    backbone_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    freeze_backbone_epochs: int = 0

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    out_dir: str = ""
    save_every: int = 10
    log_interval: int = 100
    log_file: str = "train.log"
    save_optimizer: bool = True
    pretrained_ckpt: Optional[str] = None
    pretrained_strict: bool = False
    restart: Optional[str] = None
    restart_ignore_config: bool = False

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Architecture validation
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {self.emb_dim}")
        
        if self.inv_layer < 1:
            raise ValueError(f"inv_layer must be >= 1, got {self.inv_layer}")
        
        if self.se3_layer < 0:
            raise ValueError(f"se3_layer must be >= 0, got {self.se3_layer}")
        
        if self.heads <= 0:
            raise ValueError(f"heads must be > 0, got {self.heads}")
        
        if self.emb_dim % self.heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by heads ({self.heads})"
            )
        
        if self.atom_vocab_size <= 0:
            raise ValueError(f"atom_vocab_size must be > 0, got {self.atom_vocab_size}")
        
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}")
        
        if self.num_kernel <= 0:
            raise ValueError(f"num_kernel must be > 0, got {self.num_kernel}")
        
        if self.head_hidden_dim <= 0:
            raise ValueError(f"head_hidden_dim must be > 0, got {self.head_hidden_dim}")

        valid_encoders = {"single_mol", "reacformer_se3", "reacformer_so2"}
        if self.encoder_type not in valid_encoders:
            raise ValueError(
                f"encoder_type must be one of {valid_encoders}, got {self.encoder_type!r}"
            )
        
        # Dropout validation
        if not 0.0 <= self.path_dropout <= 1.0:
            raise ValueError(f"path_dropout must be in [0, 1], got {self.path_dropout}")
        
        if not 0.0 <= self.activation_dropout <= 1.0:
            raise ValueError(f"activation_dropout must be in [0, 1], got {self.activation_dropout}")
        
        if not 0.0 <= self.attn_dropout <= 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1], got {self.attn_dropout}")
        
        if not 0.0 <= self.head_dropout <= 1.0:
            raise ValueError(f"head_dropout must be in [0, 1], got {self.head_dropout}")
        
        # Optimization validation
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        
        if self.backbone_lr <= 0:
            raise ValueError(f"backbone_lr must be > 0, got {self.backbone_lr}")
        
        if self.head_lr <= 0:
            raise ValueError(f"head_lr must be > 0, got {self.head_lr}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0, got {self.grad_clip}")
        
        if self.freeze_backbone_epochs < 0:
            raise ValueError(f"freeze_backbone_epochs must be >= 0, got {self.freeze_backbone_epochs}")
        
        # Runtime validation
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        
        if self.save_every <= 0:
            raise ValueError(f"save_every must be > 0, got {self.save_every}")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}")
        
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        
        # Split validation
        valid_splits = {"egnn", "dimenet"}
        if self.split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got {self.split!r}"
            )
