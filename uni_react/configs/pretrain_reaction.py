"""Dataclass schema for stage-3 reaction triplet pretraining."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReactionPretrainConfig:
    """All hyper-parameters for a stage-3 reaction consistency pretraining run."""

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_h5: str = ""
    """Path to the training reaction triplet HDF5 file."""
    val_h5: str = ""
    """Optional validation HDF5 file.  If empty, val_ratio is used to split train."""
    val_ratio: float = 0.1
    """Fraction of train data used as validation when val_h5 is empty."""
    neg_ratio: float = 0.5
    """Probability of building a negative consistency sample per item."""
    hard_negative: bool = True
    """Use same-size / same-composition molecules as hard negatives."""
    batch_size: int = 16
    num_workers: int = 16

    # ------------------------------------------------------------------
    # Model architecture (must match init_ckpt backbone)
    # ------------------------------------------------------------------
    model_name: str = "single_mol"
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
    head_hidden_dim: int = 512

    # ------------------------------------------------------------------
    # EMA teacher
    # ------------------------------------------------------------------
    teacher_momentum: float = 0.999

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    epochs: int = 20
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    consistency_weight: float = 1.0
    completion_weight: float = 128.0

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device: str = "cuda"
    seed: int = 42
    out_dir: str = "runs/single_mol_reaction"
    save_every: int = 5
    log_interval: int = 100
    log_file: str = "train.log"
    save_optimizer: bool = True
    init_ckpt: Optional[str] = None
    """Path to a stage-1/2 checkpoint used to warm-start the backbone."""
    init_strict: bool = False
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

        valid_models = {
            "single_mol",
            "gotennet_s",
            "gotennet_b",
            "gotennet_l",
            "gotennet_s_hat",
            "gotennet_b_hat",
            "gotennet_l_hat",
        }
        if self.model_name not in valid_models:
            raise ValueError(
                f"model_name must be one of {valid_models}, got {self.model_name!r}"
            )
        
        # Dropout validation
        if not 0.0 <= self.path_dropout <= 1.0:
            raise ValueError(f"path_dropout must be in [0, 1], got {self.path_dropout}")
        
        if not 0.0 <= self.activation_dropout <= 1.0:
            raise ValueError(f"activation_dropout must be in [0, 1], got {self.activation_dropout}")
        
        if not 0.0 <= self.attn_dropout <= 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1], got {self.attn_dropout}")
        
        # Data validation
        if not 0.0 <= self.val_ratio <= 1.0:
            raise ValueError(f"val_ratio must be in [0, 1], got {self.val_ratio}")
        
        if not 0.0 <= self.neg_ratio <= 1.0:
            raise ValueError(f"neg_ratio must be in [0, 1], got {self.neg_ratio}")
        
        # EMA validation
        if not 0.0 <= self.teacher_momentum <= 1.0:
            raise ValueError(f"teacher_momentum must be in [0, 1], got {self.teacher_momentum}")
        
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
        
        # Loss weight validation
        if self.consistency_weight < 0:
            raise ValueError(f"consistency_weight must be >= 0, got {self.consistency_weight}")
        
        if self.completion_weight < 0:
            raise ValueError(f"completion_weight must be >= 0, got {self.completion_weight}")
        
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
