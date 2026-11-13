"""SCC-PRNet: recurrent multi-stage spectral-spatial HSI denoiser.

This implementation retains the tested Lightning and DCMall interfaces.
It uses a light-weight Mamba scan in each branch and requires ``mamba_ssm`` on
the training server.  Missing Mamba dependencies raise an explicit error so a
different recurrent operator cannot be trained accidentally.
"""

import logging

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import BaseModel

try:
    from mamba_ssm import Mamba
    MAMBA_IMPORT_ERROR = None
except ImportError as exc:
    Mamba = None
    MAMBA_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


class UnidirectionalSelectiveScan(nn.Module):
    """One-way Mamba scan used for the fast preliminary-training setting.

    The recurrent SRB state and the three SCC-PRNet stages still propagate
    information repeatedly; only the costly reverse Mamba pass is omitted.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if Mamba is None:
            raise ImportError(
                "SCC-PRNet requires mamba_ssm on the training server; refusing to "
                "replace its selective scan with a memory-heavy GRU fallback. "
                f"Original import error: {MAMBA_IMPORT_ERROR}"
            ) from MAMBA_IMPORT_ERROR
        self.forward_scan = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, sequence):
        sequence = self.norm(sequence)
        return self.forward_scan(sequence)


# ---------------------------------------------------------------------------
# Original bidirectional scan kept here for the final, accuracy-oriented run.
# To restore it, uncomment this class and change ``self.scan`` in
# StateRefinementBlock to ``BidirectionalSelectiveScan(...)``.
#
# class BidirectionalSelectiveScan(nn.Module):
#     """Mamba scan with explicit forward and reverse passes."""
#
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         if Mamba is None:
#             raise ImportError("SCC-PRNet requires mamba_ssm.") from MAMBA_IMPORT_ERROR
#         self.forward_scan = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
#         self.backward_scan = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
#
#     def forward(self, sequence):
#         sequence = self.norm(sequence)
#         forward = self.forward_scan(sequence)
#         backward = self.backward_scan(sequence.flip(1))
#         return 0.5 * (forward + backward.flip(1))
# ---------------------------------------------------------------------------


class StateRefinementBlock(nn.Module):
    """Selective state update + gated recurrent state writing + residual MLP."""

    def __init__(self, dim, d_state, d_conv, expand, mlp_ratio=2):
        super().__init__()
        self.scan = UnidirectionalSelectiveScan(dim, d_state, d_conv, expand)
        self.write_gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, tokens, recurrent_state=None):
        candidate = self.scan(tokens)
        if recurrent_state is None:
            recurrent_state = torch.zeros_like(candidate)
        gate_input = torch.cat([candidate, recurrent_state], dim=-1)
        write = self.write_gate(gate_input)
        recurrent_state = write * candidate + (1.0 - write) * recurrent_state
        read = self.read_gate(torch.cat([tokens, recurrent_state], dim=-1))
        tokens = tokens + read * recurrent_state
        return tokens + self.mlp(tokens), recurrent_state


class RecurrentStateRefinementModule(nn.Module):
    """A cascade of SRBs operating on one ordered sequence.

    Each SRB is activation-checkpointed during training.  This preserves the
    configured SRB recurrence exactly while recomputing block activations in the
    backward pass, which makes full-resolution DCMall training feasible on a
    22 GB GPU.
    """

    def __init__(self, dim, depth, d_state, d_conv, expand):
        super().__init__()
        self.blocks = nn.ModuleList(
            [StateRefinementBlock(dim, d_state, d_conv, expand) for _ in range(depth)]
        )

    def forward(self, tokens):
        state = torch.zeros_like(tokens)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                tokens, state = checkpoint(
                    block, tokens, state, use_reentrant=False
                )
            else:
                tokens, state = block(tokens, state)
        return tokens


class CrossBranchRecurrentInteraction(nn.Module):
    """Align spatial features to spectral layout and update a stage-persistent fusion state."""

    def __init__(self, dim):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv3d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(nn.Conv3d(2 * dim, dim, kernel_size=1), nn.Sigmoid())

    def forward(self, spectral, spatial, previous_fusion):
        # Both branches are aligned as [B, D, bands, H, W] before fusion.
        candidate = self.mix(torch.cat([spectral, spatial], dim=1))
        if previous_fusion is None:
            previous_fusion = torch.zeros_like(candidate)
        gate = self.gate(torch.cat([candidate, previous_fusion], dim=1))
        fusion = gate * candidate + (1.0 - gate) * previous_fusion
        return fusion


class RecurrentDenoisingStage(nn.Module):
    """One stage: residual-conditioned encoding, dual RSRM, CRIM, and residual recovery."""

    def __init__(self, dim, srb_depth, d_state, d_conv, expand, sequence_batch_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        self.spectral_rsrm = RecurrentStateRefinementModule(dim, srb_depth, d_state, d_conv, expand)
        self.spatial_rsrm = RecurrentStateRefinementModule(dim, srb_depth, d_state, d_conv, expand)
        self.sequence_batch_size = sequence_batch_size
        self.crim = CrossBranchRecurrentInteraction(dim)
        self.recovery = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, 1, kernel_size=3, padding=1),
        )
        self.fusion_to_image = nn.Conv3d(dim, 1, kernel_size=1)

    def _refine_in_chunks(self, tokens, module):
        """Limit peak memory without changing the independent sequence ordering."""
        if self.sequence_batch_size is None or tokens.shape[0] <= self.sequence_batch_size:
            return module(tokens)
        return torch.cat(
            [module(chunk) for chunk in tokens.split(self.sequence_batch_size, dim=0)], dim=0
        )

    def forward(self, observation, estimate, previous_fusion):
        # x_t receives observation, previous estimate residual, and prior fusion injection.
        residual = observation - estimate
        if previous_fusion is None:
            fusion_image = torch.zeros_like(observation)
        else:
            fusion_image = self.fusion_to_image(previous_fusion).squeeze(1)
        stage_input = torch.stack([observation, residual, fusion_image], dim=1)
        features = self.encoder(stage_input)  # [B, D, C, H, W]
        b, d, c, h, w = features.shape

        spectral_tokens = features.permute(0, 3, 4, 2, 1).reshape(b * h * w, c, d)
        spectral = self._refine_in_chunks(spectral_tokens, self.spectral_rsrm)
        spectral = spectral.reshape(b, h, w, c, d).permute(0, 4, 3, 1, 2).contiguous()

        spatial_tokens = features.permute(0, 2, 3, 4, 1).reshape(b * c, h * w, d)
        spatial = self._refine_in_chunks(spatial_tokens, self.spatial_rsrm)
        spatial = spatial.reshape(b, c, h, w, d).permute(0, 4, 1, 2, 3).contiguous()

        fusion = self.crim(spectral, spatial, previous_fusion)
        correction = self.recovery(fusion).squeeze(1)
        return estimate + correction, fusion


class SCCPRNet(BaseModel):
    """Multi-stage recurrent HSI denoiser compatible with the original trainer."""

    def __init__(
        self,
        base,
        channels=191,
        stages=3,
        dim=32,
        srb_depth=4,
        d_state=16,
        d_conv=4,
        expand=2,
        sequence_batch_size=256,
        ssl=0,
        n_ssl=0,
        ckpt=None,
    ):
        super().__init__(**base)
        self.channels = channels
        self.ssl = ssl
        self.n_ssl = n_ssl
        self.stages = nn.ModuleList(
            [
                RecurrentDenoisingStage(
                    dim, srb_depth, d_state, d_conv, expand, sequence_batch_size
                )
                for _ in range(stages)
            ]
        )
        if ckpt is not None:
            checkpoint = torch.load(ckpt, map_location="cpu")
            self.load_state_dict(checkpoint.get("state_dict", checkpoint))

    def forward(self, x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs):
        del mode, img_id, sigmas, ssl_idx, kwargs
        estimate = x.float()
        fusion = None
        for stage in self.stages:
            estimate, fusion = stage(x.float(), estimate, fusion)
        return estimate
