import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from .registry import register_model
from ..utils.warp_utils import WarpMedian
from ..utils.sampling_utils import make_3d_uniform_coord_triangle
from .block.config import dinov3_model_configs
from .block.prompt_models import GeneralPromptModel, SelfAttnPromptModel
from .block.implicit_decoder import ImplicitHead
from .block.convolution import BasicEncoder
from .disparity_refiner import DisparitySparseRefiner, bound_disparity_residual

acc_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)


def _resolve_local_dinov3_repo() -> str:
    """Always use the in-repo local DINOv3 torchhub path."""
    dinov3_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "block", "torchhub", "dinov3"))
    if not os.path.isdir(dinov3_repo):
        raise FileNotFoundError(
            "DINOv3 local torchhub repo not found at fixed path: "
            f"{dinov3_repo}"
        )
    return dinov3_repo


def _make_dense_query_coord(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create dense 2D query coordinates in [-1, 1], order (y, x)."""
    ys = ((torch.arange(h, device=device, dtype=torch.float32) + 0.5) / max(float(h), 1.0)) * 2.0 - 1.0
    xs = ((torch.arange(w, device=device, dtype=torch.float32) + 0.5) / max(float(w), 1.0)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    query = torch.stack([grid_y, grid_x], dim=-1).reshape(1, -1, 2)
    return query.expand(batch, -1, -1).contiguous()
                      

@dataclass
class _InferenceState:
    gt_depth: Optional[torch.Tensor] = None
    gt_depth_mask: Optional[torch.Tensor] = None
    prompt_depth: Optional[torch.Tensor] = None
    prompt_mask: Optional[torch.Tensor] = None
    reference_meta: Optional[torch.Tensor] = None
    query_coord: Optional[torch.Tensor] = None


@dataclass
class InfiniDepthEncoding:
    """Image features reused by arbitrary disparity queries and fixed-grid refinement."""

    dino_features: torch.Tensor
    basic_features: torch.Tensor
    patch_size: Tuple[int, int]
    dino_tokens: torch.Tensor


@dataclass
class DisparityRefinementOutput:
    disparity: torch.Tensor
    disparity_sequence: List[torch.Tensor]
    raw_residuals: List[torch.Tensor]
    bounded_residuals: List[torch.Tensor]
    voxel_statistics: List[Dict[str, object]]
    reference_scale: Optional[torch.Tensor] = None


class _BaseInfiniDepthModel(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder: str = "vitl16",
    ):
        super().__init__()
        self.model_config = dinov3_model_configs[encoder]
        local_dinov3_repo = _resolve_local_dinov3_repo()
        self.pretrained = torch.hub.load(
            local_dinov3_repo,
            f"dinov3_{encoder}",
            source="local",
            pretrained=False,
        )
        self.patch_size = 16
        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.basic_encoder = BasicEncoder(
            input_dim=3,
            output_dim=128,
            stride=4,
        )
        self.depth_implicit_head = ImplicitHead(
            hidden_dim=dim,
            basic_dim=128,
            fusion_type="concat",
            out_dim=1,
            hidden_list=[1024, 256, 32],
        )
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._init_variant_modules()
        self.disparity_refiner: Optional[DisparitySparseRefiner] = None

        if model_path is not None:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                self.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
            else:
                raise FileNotFoundError(f"Model file {model_path} not found")

        # only for inference
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to initialize InfiniDepth models.")
        self.cuda()
        self.eval()

    def _init_variant_modules(self):
        """Subclasses can attach extra modules or state."""

    def _transform_features(
        self,
        features,
        patch_h: int,
        patch_w: int,
        state: _InferenceState,
    ):
        """Subclasses can inject prompt/depth conditioning."""
        return features

    def _prepare_inference(self, state: _InferenceState) -> _InferenceState:
        """Subclasses can pre-process inference inputs before forward passes."""
        return state

    def _postprocess_inference(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _prepare_backbone_features(
        self,
        x: torch.Tensor,
        state: _InferenceState,
    ):
        h, w = x.shape[-2:]
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        dino_tokens = features[-1][0].clone()
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        features = self._transform_features(
            features,
            patch_h,
            patch_w,
            state,
        )

        x_basic = 2.0 * x - 1.0
        basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]
        return features, basic_feat, patch_h, patch_w, dino_tokens

    def _encode_image_with_state(
        self,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> InfiniDepthEncoding:
        features, basic_feat, patch_h, patch_w, dino_tokens = self._prepare_backbone_features(
            image,
            state=state,
        )
        dino_features = self.depth_implicit_head._encode_feat(features, patch_h, patch_w)
        return InfiniDepthEncoding(
            dino_features=dino_features,
            basic_features=basic_feat,
            patch_size=(patch_h, patch_w),
            dino_tokens=dino_tokens,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> InfiniDepthEncoding:
        state = _InferenceState(prompt_depth=prompt_depth, prompt_mask=prompt_mask)
        return self._encode_image_with_state(image, state)

    def decode_disparity(
        self,
        encoding: InfiniDepthEncoding,
        query_coord: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if query_coord.ndim != 3 or query_coord.shape[-1] != 2:
            raise ValueError(f"Expected query coordinates [B,N,2], got {tuple(query_coord.shape)}")
        if query_coord.shape[0] != encoding.dino_features.shape[0]:
            raise ValueError("Encoding and query batch sizes do not match")
        count = query_coord.shape[1]
        if count <= 0:
            raise ValueError("At least one query coordinate is required")
        if chunk_size is None:
            chunk_size = max(1, count)
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        predictions = []
        for start in range(0, count, int(chunk_size)):
            predictions.append(
                self.depth_implicit_head._decode_dpt(
                    encoding.dino_features,
                    encoding.basic_features,
                    query_coord[:, start : start + int(chunk_size)],
                )
            )
        return torch.cat(predictions, dim=1)

    def attach_disparity_refiner(
        self,
        *,
        backend: str = "spconv",
        voxel_resolution: float = 200.0,
        max_disparity_span: Optional[int] = None,
    ) -> DisparitySparseRefiner:
        visual_dim = int(self.pretrained.blocks[0].attn.qkv.in_features)
        self.disparity_refiner = DisparitySparseRefiner(
            visual_dim=visual_dim,
            voxel_resolution=voxel_resolution,
            backend=backend,
            max_disparity_span=max_disparity_span,
        ).to(next(self.parameters()).device)
        return self.disparity_refiner

    def forward_dense_refined(
        self,
        image: torch.Tensor,
        *,
        prompt_disparity: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        query_hw: Tuple[int, int] = (384, 512),
        num_refinement_steps: int = 3,
        residual_scale: float = 1.0,
        detach_base_from_refiner: bool = False,
        chunk_size: int = 10000,
    ) -> DisparityRefinementOutput:
        if self.disparity_refiner is None:
            raise RuntimeError("Call attach_disparity_refiner() before refined inference")
        height, width = (int(value) for value in query_hw)
        if height <= 0 or width <= 0:
            raise ValueError("query_hw must contain positive values")
        if not 0 <= int(num_refinement_steps) <= 7:
            raise ValueError("num_refinement_steps must be in [0, 7]")
        residual_scale = float(residual_scale)
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1]")
        state = _InferenceState(
            prompt_depth=prompt_disparity,
            prompt_mask=prompt_mask,
        )
        if prompt_disparity is not None or prompt_mask is not None:
            state = self._prepare_inference(state)
            encoding = self._encode_image_with_state(image, state)
        else:
            encoding = self.encode_image(image)
        query = _make_dense_query_coord(image.shape[0], height, width, image.device)
        decoded = self.decode_disparity(encoding, query, chunk_size=chunk_size)
        base_disparity = decoded[..., 0].reshape(image.shape[0], height, width).float()
        disparity_sequence = [base_disparity]
        raw_residuals: List[torch.Tensor] = []
        bounded_residuals: List[torch.Tensor] = []
        voxel_statistics: List[Dict[str, object]] = []
        refined = base_disparity.detach() if detach_base_from_refiner else base_disparity
        visual = encoding.dino_features.float()
        if detach_base_from_refiner:
            visual = visual.detach()
        with torch.autocast(device_type=image.device.type, enabled=False):
            for _ in range(int(num_refinement_steps)):
                raw, statistics = self.disparity_refiner(refined.float(), visual)
                bounded = bound_disparity_residual(raw, max_abs=0.1) * residual_scale
                refined = refined + bounded
                raw_residuals.append(raw)
                bounded_residuals.append(bounded)
                voxel_statistics.append(statistics)
                disparity_sequence.append(refined)
        return DisparityRefinementOutput(
            disparity=disparity_sequence[-1],
            disparity_sequence=disparity_sequence,
            raw_residuals=raw_residuals,
            bounded_residuals=bounded_residuals,
            voxel_statistics=voxel_statistics,
            reference_scale=state.reference_meta,
        )

    def _to_depth_disparity(self, pred: torch.Tensor):
        pred_disparity = pred
        pred_depth = 1.0 / torch.clamp(pred, min=5e-3)
        return pred_depth, pred_disparity

    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        query_coord: torch.Tensor,
        use_batch_infer=True,
        return_dino_tokens: bool = False,
        gt_depth: Optional[torch.Tensor] = None,
        gt_depth_mask: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ):
        state = _InferenceState(
            gt_depth=gt_depth,
            gt_depth_mask=gt_depth_mask,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
            query_coord=query_coord,
        )
        state = self._prepare_inference(state)

        if use_batch_infer:
            pred = self.batch_forward(
                image,
                query_coord,
                bsize=10000,
                return_dino_tokens=return_dino_tokens,
                prompt_depth=state.prompt_depth,
                prompt_mask=state.prompt_mask,
            )
        else:
            pred = self.forward(
                image,
                query_coord,
                return_dino_tokens=return_dino_tokens,
                prompt_depth=state.prompt_depth,
                prompt_mask=state.prompt_mask,
            )

        if return_dino_tokens:
            pred, dino_tokens = pred

        pred_depth, pred_disparity = self._postprocess_inference(
            pred,
            image,
            state,
        )

        if return_dino_tokens:
            return pred_depth, pred_disparity, dino_tokens
        return pred_depth, pred_disparity

    def batch_forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        bsize: int = 3000,
        return_dino_tokens: bool = False,
    ):
        """Forward pass with batching to avoid OOM."""
        state = _InferenceState(prompt_depth=prompt_depth, prompt_mask=prompt_mask)
        encoding = self._encode_image_with_state(x, state)
        pred = self.decode_disparity(encoding, coord, chunk_size=bsize)
        if return_dino_tokens:
            return pred, encoding.dino_tokens
        return pred

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        prompt_depth: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        return_dino_tokens: bool = False,
    ):
        state = _InferenceState(prompt_depth=prompt_depth, prompt_mask=prompt_mask)
        encoding = self._encode_image_with_state(x, state)
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            depth = self.decode_disparity(encoding, coords)
        if return_dino_tokens:
            return depth, encoding.dino_tokens
        return depth

    def _prepare_dense_depthmap_for_gs(
        self,
        pred_depth: torch.Tensor,
        batch: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        if pred_depth.ndim == 4:
            return pred_depth
        if pred_depth.ndim == 3 and pred_depth.shape[1] == h * w and pred_depth.shape[2] == 1:
            return pred_depth.permute(0, 2, 1).reshape(batch, 1, h, w)
        raise ValueError(
            f"Unsupported pred_depth shape for dense GS depthmap conversion: {tuple(pred_depth.shape)}"
        )

    @torch.no_grad()
    def inference_for_gs(
        self,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        gt_depth: torch.Tensor,
        gt_depth_mask: torch.Tensor,
        prompt_depth: torch.Tensor,
        prompt_mask: torch.Tensor,
        sky_mask: Optional[torch.Tensor] = None,
        sample_point_num: int = 200000,
        coord_deterministic_sampling: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """GS-specific inference with two steps:
        step1: 2D-uniform dense query; step2: 3D-uniform triangle query."""
        b, _, h, w = image.shape

        query = _make_dense_query_coord(b, h, w, image.device)
        pred_depth, _, dino_tokens = self.inference(
            image=image,
            query_coord=query,
            gt_depth=gt_depth,
            gt_depth_mask=gt_depth_mask,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
            use_batch_infer=True,
            return_dino_tokens=True,
        )
        depthmap = self._prepare_dense_depthmap_for_gs(pred_depth, b, h, w)

        sampled_coords = []
        for bi in range(b):
            coord_b = make_3d_uniform_coord_triangle(
                depth_hw=depthmap[bi, 0],
                fx=float(intrinsics[bi, 0, 0].item()),
                fy=float(intrinsics[bi, 1, 1].item()),
                cx=float(intrinsics[bi, 0, 2].item()),
                cy=float(intrinsics[bi, 1, 2].item()),
                N=sample_point_num,
                coord_norm="minus_one_to_one",
                sample_filter_mode="max_depth",
                sky_mask_hw=None if sky_mask is None else sky_mask[bi],
                deterministic=coord_deterministic_sampling,
            )
            sampled_coords.append(coord_b)
        query_3d_uniform_coord = torch.stack(sampled_coords, dim=0)  # [B, N, 2]

        pred_depth_3d, _ = self.inference(
            image=image,
            query_coord=query_3d_uniform_coord,
            gt_depth=gt_depth,
            gt_depth_mask=gt_depth_mask,
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,
            use_batch_infer=True,
            return_dino_tokens=False,
        )
        return depthmap, dino_tokens, query_3d_uniform_coord, pred_depth_3d


@register_model("InfiniDepth_DepthSensor")
class InfiniDepth_DepthSensor(_BaseInfiniDepthModel):
    def _init_variant_modules(self):
        self.prompt_model = GeneralPromptModel(
            prompt_stage=[3],
            block=SelfAttnPromptModel(num_blocks=4, pe="qk"),
        )
        self.warp_func = WarpMedian()

    def _transform_features(
        self,
        features,
        patch_h: int,
        patch_w: int,
        state: _InferenceState,
    ):  
        return self.prompt_model(
            features,
            state.prompt_depth,
            state.prompt_mask,
            patch_h,
            patch_w,
        )

    def _prepare_inference(self, state: _InferenceState) -> _InferenceState:
        if state.prompt_depth is None or state.prompt_mask is None:
            raise ValueError(
                "InfiniDepth_DepthSensor inference requires prompt_depth and prompt_mask."
            )
        warp_kwargs = {
            "prompt_depth": state.prompt_depth,
            "prompt_mask": state.prompt_mask,
        }
        if state.gt_depth is not None and state.gt_depth_mask is not None:
            warp_kwargs.update(
                ground_truth=state.gt_depth,
                ground_truth_mask=state.gt_depth_mask,
            )
        prompt_depth, prompt_mask, reference_meta = self.warp_func.warp(
            state.prompt_depth,
            **warp_kwargs,
        )
        state.prompt_depth = prompt_depth
        state.prompt_mask = prompt_mask
        state.reference_meta = reference_meta
        return state

    def _postprocess_inference(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.reference_meta is None:
            raise ValueError("reference_meta is required for InfiniDepth_DepthSensor postprocessing.")
        pred = self.warp_func.unwarp(
            pred,
            reference_meta=state.reference_meta[..., 0],
        )
        return self._to_depth_disparity(pred)


@register_model("InfiniDepth")
class InfiniDepth(_BaseInfiniDepthModel):
    def _init_variant_modules(self):
        poly_features = PolynomialFeatures(degree=1, include_bias=False)
        ransac = RANSACRegressor(max_trials=1000)
        self.ransac_model = make_pipeline(poly_features, ransac)
        self._cached_denorm_scale: Optional[torch.Tensor] = None
        self._cached_denorm_shift: Optional[torch.Tensor] = None

    def _get_cached_denorm_params(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self._cached_denorm_scale is None or self._cached_denorm_shift is None:
            return None

        scale = self._cached_denorm_scale
        shift = self._cached_denorm_shift
        if scale.numel() == 1 and batch > 1:
            scale = scale.expand(batch)
            shift = shift.expand(batch)
        elif batch == 1 and scale.numel() > 1:
            scale = scale[:1]
            shift = shift[:1]
        elif scale.numel() != batch:
            return None

        return (
            scale.to(device=device, dtype=dtype).reshape(batch, 1, 1),
            shift.to(device=device, dtype=dtype).reshape(batch, 1, 1),
        )

    def _ransac_align_depth(self, pred, gt, mask0=None):
        if type(pred).__module__ == torch.__name__:
            pred = pred.cpu().numpy()
        if type(gt).__module__ == torch.__name__:
            gt = gt.cpu().numpy()
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        gt = gt.squeeze()
        pred = pred.squeeze()
        mask = (gt > 1e-8)  # & (pred > 1e-8)
        if mask0 is not None and mask0.sum() > 0:
            if type(mask0).__module__ == torch.__name__:
                mask0 = mask0.cpu().numpy()
            mask0 = mask0.squeeze()
            mask0 = mask0 > 0
            mask = mask & mask0
        gt_mask = gt[mask].astype(np.float32)
        pred_mask = pred[mask].astype(np.float32)

        gt_mask = np.clip(gt_mask, 1e-8, None)

        try:
            self.ransac_model.fit(pred_mask[:, None], gt_mask[:, None])
            a, b = (
                self.ransac_model.named_steps["ransacregressor"].estimator_.coef_,
                self.ransac_model.named_steps["ransacregressor"].estimator_.intercept_,
            )
            a = a.item()
            b = b.item()
        except Exception:
            a, b = 1, 0

        if not np.isfinite(a):
            a = 1.0
        if not np.isfinite(b):
            b = 0.0

        if a > 0:
            pred_metric = a * pred + b
        else:
            if pred_mask.size > 0 and gt_mask.size > 0:
                pred_mean = max(float(np.mean(pred_mask)), 1e-8)
                gt_mean = float(np.mean(gt_mask))
                a = gt_mean / pred_mean
            else:
                a = 1.0
            b = 0.0
            pred_metric = a * pred + b

        return torch.from_numpy(pred_metric).unsqueeze(0).unsqueeze(0), float(a), float(b)

    @staticmethod
    def _infer_dense_query_hw(query_coord: Optional[torch.Tensor], n_query: int) -> Optional[tuple[int, int]]:
        if query_coord is None or query_coord.ndim != 3 or query_coord.shape[1] != n_query:
            return None

        query0 = query_coord[0]
        if query0.ndim != 2 or query0.shape[1] != 2:
            return None

        h_query = int(torch.unique(query0[:, 0]).numel())
        w_query = int(torch.unique(query0[:, 1]).numel())
        if h_query * w_query != n_query:
            return None
        return h_query, w_query

    def _postprocess_inference(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        state: _InferenceState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, _, _ = image.shape
        n_query = pred.shape[1]
        dense_query_hw = self._infer_dense_query_hw(state.query_coord, n_query)
        if dense_query_hw is not None and state.gt_depth is not None:
            h_query, w_query = dense_query_hw
            pred_map = pred.permute(0, 2, 1).reshape(b, 1, h_query, w_query)
            gt_align = state.gt_depth
            gt_mask_align = state.gt_depth_mask
            if gt_align.shape[-2:] != (h_query, w_query):
                gt_align = F.interpolate(
                    gt_align,
                    size=(h_query, w_query),
                    mode="bilinear",
                    align_corners=False,
                )
                if gt_mask_align is not None:
                    gt_mask_align = F.interpolate(
                        gt_mask_align.float(),
                        size=(h_query, w_query),
                        mode="nearest",
                    )
            aligned = []
            scales = []
            shifts = []
            for i in range(b):
                aligned_i, scale_i, shift_i = self._ransac_align_depth(
                    pred_map[i: i + 1],
                    gt_align[i: i + 1],
                    None if gt_mask_align is None else gt_mask_align[i: i + 1],
                )
                aligned_i = aligned_i.to(device=image.device, dtype=pred.dtype)
                aligned.append(aligned_i)
                scales.append(scale_i)
                shifts.append(shift_i)
            pred_map = torch.cat(aligned, dim=0)
            pred = pred_map.reshape(b, 1, -1).permute(0, 2, 1)
            self._cached_denorm_scale = torch.tensor(scales, dtype=torch.float32)
            self._cached_denorm_shift = torch.tensor(shifts, dtype=torch.float32)
        else:
            cached_params = self._get_cached_denorm_params(
                batch=b,
                device=pred.device,
                dtype=pred.dtype,
            )
            if cached_params is not None:
                scale, shift = cached_params
                pred = pred * scale + shift
        return self._to_depth_disparity(pred)
