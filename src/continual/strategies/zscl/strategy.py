from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.continual.task_manager import ContinualTask, ContinualTaskManager
from src.continual.strategies.base import BaseContinualStrategy
from src.continual.strategies.registry import register_strategy
from src.engine.model_engine import VoxTellEngine
from src.utils.checkpoint import atomic_torch_save


@register_strategy
class ZSCLStrategy(BaseContinualStrategy):
    """ZSCL-style continual learning using teacher distillation and weight averaging.

    This implementation follows the paper's core ideas in a model-agnostic way:
    1. keep a frozen teacher snapshot from the initial task state;
    2. add a distillation regularizer on current logits to preserve old feature-space behavior;
    3. maintain an exponential weight-ensemble snapshot that can be saved for reuse.
    """

    strategy_name = "zscl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model: torch.nn.Module | None = None
        self.ensemble_model: torch.nn.Module | None = None
        self._ensemble_steps = 0
        self._ensemble_dirty = False   # ensemble changed since the last recovery save

        continual_cfg = dict(self.cfg.get("continual", {}))
        zscl_cfg = dict(continual_cfg.get("zscl", {}))
        self.distillation_weight = float(zscl_cfg.get("distillation_weight", 0.1))
        self.weight_ensemble_interval = int(zscl_cfg.get("weight_ensemble_interval", 0))

    def build_engine(self):
        engine = VoxTellEngine(self.base_model_cfg, return_features=True)

        self.teacher_model = copy.deepcopy(engine.model)
        self.teacher_model.cpu().eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.ensemble_model = copy.deepcopy(engine.model)
        self.ensemble_model.to(engine.device).eval()
        for parameter in self.ensemble_model.parameters():
            parameter.requires_grad = False

        self.logger.info(
            "ZSCL strategy initialized | distillation_weight=%.4f | weight_ensemble_interval=%d",
            self.distillation_weight,
            self.weight_ensemble_interval,
        )
        return engine

    def compute_loss(
        self,
        *,
        task: ContinualTask,
        batch: dict[str, Any],
        outputs: Any,
        base_loss: Any,
    ) -> Any:
        if self.teacher_model is None or self.distillation_weight <= 0.0:
            return base_loss

        s_logits, s_masks, s_imgs = outputs
        t_logits, t_masks, t_imgs = self._teacher_logits(batch)
        
        B = s_masks[4].shape[0]

        s_mask = s_masks[self.teacher_model.selected_decoder_layer].view(B, self.teacher_model.num_heads, -1)
        t_mask = t_masks[self.teacher_model.selected_decoder_layer].view(B, self.teacher_model.num_heads, -1)
        s_img = s_imgs[self.teacher_model.selected_decoder_layer]
        t_img = t_imgs[self.teacher_model.selected_decoder_layer]

        loss = base_loss

        # =========================================================
        # IMAGE–PROMPT ZSCL DISTILLATION (core part)
        # =========================================================
        if s_mask is not None and s_img is not None:

            # normalize
            s_mask = F.normalize(s_mask.float(), dim=-1)
            t_mask = F.normalize(t_mask.float().detach(), dim=-1)

            s_img = s_img.float()
            t_img = t_img.float().detach()

            # -------------------------------------------------
            # STEP 1: compute prompt → voxel similarity
            # -------------------------------------------------

            # student: (B, N, C) x (B, C, H, W, D)
            sim_student = torch.einsum(
                'b c h w d, b n c -> b n h w d',
                s_img, s_mask
            )

            # teacher: same
            sim_teacher = torch.einsum(
                'b c h w d, b n c -> b n h w d',
                t_img, t_mask
            )

            # -------------------------------------------------
            # STEP 2: convert to distributions (ZSCL idea)
            # -------------------------------------------------
            # normalize over spatial voxels
            p_student = F.log_softmax(sim_student.flatten(2), dim=-1)
            p_teacher = F.softmax(sim_teacher.flatten(2), dim=-1)

            # -------------------------------------------------
            # STEP 3: KL divergence (ZSCL distillation)
            # -------------------------------------------------
            distill_loss = F.kl_div(
                p_student,
                p_teacher,
                reduction="batchmean",
                log_target=False
            )

            loss = loss + self.distillation_weight * distill_loss

        return loss

    def _teacher_logits(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if self.teacher_model is None:
            return None

        engine = self.require_engine()
        imgs = batch["image"].to(engine.device)
        text_embeddings = engine.text_encoder.embed(batch["prompts"]).clone()

        self.teacher_model.to(engine.device)
        with torch.no_grad():
            outputs  = self.teacher_model(imgs, text_embeddings, return_features=True)
        self.teacher_model.cpu()

        if isinstance(outputs, tuple):
            logits, mask_embeds, img_feats = outputs
        else:
            logits, mask_embeds, img_feats = outputs, None, None
        return logits, mask_embeds, img_feats

    def on_train_batch_end(
        self,
        task: ContinualTask,
        batch_idx: int,
        epoch: int,
        batch_loss: float,
    ) -> None:
        if self.weight_ensemble_interval <= 0 or self.ensemble_model is None:
            return

        if (batch_idx + 1) % self.weight_ensemble_interval != 0:
            return

        engine = self.require_engine()
        self._ensemble_steps += 1

        current_state = engine.model.state_dict()
        ensemble_state = self.ensemble_model.state_dict()

        alpha = 1.0 / float(self._ensemble_steps + 1)
        blended = {
            name: (alpha * current_state[name]) + ((1.0 - alpha) * ensemble_state[name])
            for name in current_state
        }
        self.ensemble_model.load_state_dict(blended, strict=True)
        self._ensemble_dirty = True

    # ------------------------------------------------------------------
    # Persistence of the weight ensemble (evaluation and job resumes)
    # ------------------------------------------------------------------
    def _save_ensemble(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            {"network_weights": self.ensemble_model.state_dict(), "ensemble_steps": int(self._ensemble_steps)},
            path,
        )

    def _load_ensemble(self, path: Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        self.ensemble_model.load_state_dict(checkpoint["network_weights"], strict=True)
        self._ensemble_steps = int(checkpoint.get("ensemble_steps", 0))
        self._ensemble_dirty = False
        self.logger.info("Restored weight ensemble from %s (steps=%d)", path, self._ensemble_steps)

    def on_recovery_checkpoint(self, *, task, checkpoint_path, epoch, batch_idx, stage) -> None:
        super().on_recovery_checkpoint(
            task=task, checkpoint_path=checkpoint_path, epoch=epoch, batch_idx=batch_idx, stage=stage
        )
        if self.ensemble_model is not None and self._ensemble_dirty:
            self._save_ensemble(Path(checkpoint_path).parent / "ensemble_recovery.pt")
            self._ensemble_dirty = False

    def on_task_start(self, task: ContinualTask, task_dir: Path, resuming_within_task: bool) -> None:
        if not resuming_within_task or self.ensemble_model is None:
            return
        recovery = task_dir / "checkpoints" / "ensemble_recovery.pt"
        if recovery.exists():
            self._load_ensemble(recovery)
        elif task.index > 0:
            previous = self.tasks[task.index - 1]
            previous_dir = self.dirs["root"] / "tasks" / self.task_manager.task_dir_name(previous)
            self._load_ensemble(previous_dir / "checkpoints" / "ensemble_model.pt")

    def restore_after_task(self, task: ContinualTask, task_dir: Path) -> None:
        super().restore_after_task(task, task_dir)
        if self.ensemble_model is not None:
            self._load_ensemble(task_dir / "checkpoints" / "ensemble_model.pt")

    def after_task(
        self,
        task: ContinualTask,
        task_dir: Path,
        task_training_cfg: dict[str, Any],
        task_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.ensemble_model is None:
            return

        # Saved next to the student checkpoints so that the evaluator can load it with
        # checkpoint_name="ensemble_model.pt" (the earlier location task_dir/best_model.pt was not evaluated).
        ensemble_path = task_dir / "checkpoints" / "ensemble_model.pt"
        self._save_ensemble(ensemble_path)
        self.logger.info("Saved ZSCL weight ensemble snapshot to %s (steps=%d)", ensemble_path, self._ensemble_steps)


def run_zscl_strategy(
    *,
    cfg: dict[str, Any],
    task_manager: ContinualTaskManager,
    tasks: list[ContinualTask],
    dirs: dict[str, Path],
    logger,
) -> None:
    ZSCLStrategy(
        cfg=cfg,
        task_manager=task_manager,
        tasks=tasks,
        dirs=dirs,
        logger=logger,
    ).run()
