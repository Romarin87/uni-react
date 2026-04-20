import torch


def create_access_mask(cross_dist: torch.Tensor, cutoff: float, padding_mask: torch.Tensor = None) -> torch.Tensor:
    access_mask = cross_dist < cutoff
    tlist = torch.arange(cross_dist.shape[-1], device=cross_dist.device)
    access_mask[..., tlist, tlist] = False
    if padding_mask is not None:
        padding_part = padding_mask[..., None] | padding_mask[..., None, :]
        access_mask = access_mask & (~padding_part)
    return access_mask


def create_attn_mask(cross_dist: torch.Tensor, cutoff: float, padding_mask: torch.Tensor = None) -> torch.Tensor:
    attn_mask = cross_dist >= cutoff
    if padding_mask is not None:
        attn_mask = attn_mask | padding_mask[..., None, :]
        attn_mask = attn_mask & (~padding_mask[..., None])
    return attn_mask
