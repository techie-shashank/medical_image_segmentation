# Reruns of October 2026

Configs for the reruns decided on 2026-10-22 (see `thesis_notes.md` in the thesis repository).
Every config is a copy of the `config.yaml` of the original run, so data, seed, epochs and
optimiser settings are identical; only the settings listed below differ.

| Config | Original run | Purpose |
|---|---|---|
| `ft_original.yaml` | exp_20260816_155351_cfa776e7 | sequential fine-tuning, last-epoch evaluation |
| `ft_reverse.yaml` | exp_20260924_164721_1bf4a4dc | same, reverse task order |
| `ft_5_shot.yaml`, `ft_10_shot.yaml`, `ft_20_shot.yaml` | exp_20260926_112746_e697f061, exp_20260926_104124_ccaf6702, exp_20260925_000502_4e1f0da2 | same, few-shot |
| `distillation_original.yaml` | exp_20260924_204124_7ce6758c | last epoch + best validation + **weight ensemble** evaluated |
| `prompt_tuning_scaled_original.yaml` | exp_20260816_161201_b39ea60e | per-task gradient scaling **active**, on tokens and meta-net |
| `joint_ft.yaml`, `joint_ft_{5,10,20}_shot.yaml` | (new) | joint fine-tuning on all six tasks: upper reference |
| `shared_lora_replay_no_control.yaml` | exp_20261014_120720_abe47a15 | replay **without** control-task volumes (control task held out) |
| `ft_5_shot_equal_updates.yaml`, `task_specific_lora_5_shot_equal_updates.yaml`, `shared_lora_5_shot_equal_updates.yaml` | exp_20260926_112746_e697f061, exp_20260926_120018_3ab58c90, exp_20260930_164451_55aada20 | k = 5 with 216 epochs = 1,080 updates per task, as with full data (separates less data from fewer updates); validation every 24 epochs |

The LoRA runs need no `evaluation` section: they are always evaluated with the adapter after the last epoch.

Every run is evaluated twice (and the distillation run three times):
- main evaluation (experiment root, `eval_matrix.csv`): parameters after the last epoch (`last_model.pt`);
- `evaluations/best_validation/`: checkpoint with the lowest validation loss (`best_model.pt`), comparable with the original runs;
- `evaluations/weight_ensemble/` (distillation only): ZSCL weight ensemble (`ensemble_model.pt`).

## Code changes (all backward compatible; old configs behave as before)
1. `training/trainer.py`: also saves `checkpoints/last_model.pt` after the last epoch of every task; `best_model.pt` and `last_model.pt` hold the weights only (no optimiser state, which only the recovery checkpoint needs).
2. `continual/strategies/base.py`: writes `run_info.json` (git commit, dirty flag, config hash) at the start of every job; sets the task index before `configure_engine` (needed by CPE-CLIP); when a job resumes at a task boundary, restores the state after the previous task (`last_model.pt`) instead of starting from the pretrained model; new hook `on_task_start`; writes a task's final files (`after_task`) before marking it completed; removes the task's `*_recovery.pt` files once it is completed.
3. `continual/strategies/lora/task_specific/strategy.py`: no state is restored at a task boundary (every task starts from the pretrained model). `lora/shared/strategy.py`: at a task boundary only the adapter is restored (`lora_adapter.pt`). Neither LoRA strategy is rerun; these only keep resumes correct.
4. `continual/strategies/zscl/strategy.py`: the weight ensemble is saved as `checkpoints/ensemble_model.pt` (with its step counter), saved with every recovery checkpoint (`ensemble_recovery.pt`) and restored on resume (inside a task from `ensemble_recovery.pt`, or from the previous task's `ensemble_model.pt`).
5. `continual/strategies/cpe_clip/{model,transformer}.py`: option `cpe_clip.scale_meta_net` applies the per-task gradient scale to the meta-net as well; a log line states which parameters are scaled.
6. `continual/evaluator.py`, `cli/continual_eval.py`, `cli/continual.py`: releases the text encoder and model of every evaluation cell before the next one (`gc.collect`, `torch.cuda.empty_cache`); checkpoint taken from `continual.evaluation.checkpoint`; `--tag` / `continual.evaluation.extra` write further evaluations into `evaluations/<tag>/`; `continual.evaluation.tasks` lets the evaluated tasks differ from the trained ones (joint training).
7. `data/datamodule.py`: `dataset.joint_names` builds one dataset from the splits of several task datasets.
8. `continual/strategies/base.py` (replay): option `replay_memory.include_retention_tasks` (default true) leaves the control task out of the memory; the memory is stored in `continual_state.json` (`replay_memory`, `replay_tasks_added`) and restored on resume, and each task's samples are added only once. Before, a resumed job rebuilt the memory with the control-task volumes only.
9. `training/trainer.py`, `utils/training_plot.py`: option `training.validation_interval` (default 1) validates every n-th epoch and always after the last one; best-model selection uses the validated epochs only; the recovery checkpoint stores the validated epochs (`val_epochs`).

## 0. Before anything: same code on the HPC, enough storage
- The local `code/` was at commit 50a654b when these changes were made. On the HPC, `git status` and `git log -1` must show the same commit and no uncommitted changes; otherwise compare first.
- Storage: about 6 x 5.3 GB per full sequential fine-tuning run (best + last model per task), about 40 GB for the distillation run, less for the others; roughly 200 GB in total. Check the vault quota (e.g. `shownicerquota.pl`) before launching.

- Memory of the distillation run (the largest): the original job peaked at 38.8 GiB of the 40 GiB GPU and completed, including one resume. The rerun adds no GPU allocation during training (ensemble saves and restores copy into existing tensors). Host memory: text encoder (kept on the CPU between calls), CPU copy of the teacher (1.8 GB), transient copies of 1.8-3.5 GB when a checkpoint is loaded on resume. To see the real headroom, run `sacct -j 1821183,1822517 --format=JobID,MaxRSS,ReqMem,State` for the original distillation jobs and compare MaxRSS with ReqMem.

## 1. Before launching: commit and smoke test
```bash
git add -A && git commit -m "Reruns 2026-10: last-epoch checkpoints, ensemble evaluation, CPE-CLIP scaling, joint training"
for c in ft_original distillation_original prompt_tuning_scaled_original joint_ft shared_lora_replay_no_control ft_5_shot_equal_updates; do
  sbatch --time=01:00:00 scripts/continual.sh configs/continual/rerun_2026_10/smoke/$c.yaml
done
```
Check after the four smoke jobs (folders under `.../experiments/continual/smoke_test/`):
- [ ] every run: `run_info.json` with the commit and `"git_dirty": false`; `tasks/*/checkpoints/last_model.pt` and `best_model.pt` (weights only, about half the size of an old `best_model.pt`); no `latest_recovery.pt` left in completed tasks; `eval_matrix.csv` and `evaluations/best_validation/eval_matrix.csv`; per-cell folders `test/` and `test_best_validation/`.
- [ ] distillation: `sacct -j <smoke job> --format=JobID,MaxRSS,ReqMem,State` shows MaxRSS well below ReqMem and State COMPLETED;
  `tasks/*/checkpoints/ensemble_model.pt` and `ensemble_recovery.pt`; `evaluations/weight_ensemble/eval_matrix.csv`; log line "Saved ZSCL weight ensemble snapshot ... (steps=1)".
- [ ] prompt tuning: log lines "gradient scaling hooks registered | tokens=True | meta_net=True" and, for the second task, "CPE-CLIP session scaling | task_index=2 | alpha_t=0.5000".
- [ ] joint: log "Joint splits over 2 datasets"; `eval_matrix.csv` has the rows of 2 tasks + aeropath and the columns "Pretrained" and "After skm_tea_joint".

- [ ] replay without control: log "Replay memory starts empty (include_retention_tasks=false)", then "Replay memory size after task skm_tea_lateral_tibial: 1 samples"; `continual_state.json` contains `replay_memory` (1 then 2 entries) and `replay_tasks_added`; no `lung` prompt among the `[DATA]` lines of `run.log`; the control task (aeropath) still appears in `eval_matrix.csv`.

- [ ] validation interval: per task, `run.log` shows "Epoch 1/3 completed | Training Loss=... | no validation (validation_interval=2)", then "Epoch 2/3 completed ... Validation Loss=..." and "Epoch 3/3 completed ... Validation Loss=..."; `plots/loss_metrics.json` has `"best_epoch"` 2 or 3.

Resume at a task boundary (once on the distillation smoke run, once on the replay smoke run): in its `continual_state.json` set
`"status": "in_progress"`, `"completed_tasks": [0]`, `"current_task_index": 1`, `"current_checkpoint": null`, then
```bash
sbatch --time=01:00:00 scripts/resume_continual.sh <smoke distillation experiment folder>
```
- [ ] distillation: log lines "Restored the state after task skm_tea_lateral_tibial" and "Restored weight ensemble ... (steps=...)".
- [ ] replay: log lines "Restored replay memory with 1 samples from the run state" and "Restored the shared LoRA adapter after task skm_tea_lateral_tibial"; the second task trains with `train_batches=2` (1 task volume + 1 memory entry).

## 2. Launch
```bash
for c in ft_original ft_reverse ft_5_shot ft_10_shot ft_20_shot distillation_original prompt_tuning_scaled_original joint_ft joint_ft_5_shot joint_ft_10_shot joint_ft_20_shot shared_lora_replay_no_control ft_5_shot_equal_updates task_specific_lora_5_shot_equal_updates shared_lora_5_shot_equal_updates; do
  sbatch scripts/continual.sh configs/continual/rerun_2026_10/$c.yaml
done
```
Runs that hit the 24 h limit (full-data fine-tuning, distillation, joint) are continued with
`sbatch scripts/resume_continual.sh <experiment folder>`. Evaluation runs automatically at the end.

Expected duration on one A100 (from the original runs): sequential fine-tuning full data about 24 h training + 2 evaluations (about 5 h), so two jobs; distillation about 36 h + 3 evaluations, two jobs; prompt tuning about 19 h + 2 evaluations, possibly two jobs; joint full data about 24 h (648 samples per epoch, 138 validation volumes), two jobs; few-shot runs a few hours each; the three equal-updates runs about 22 h training each (1,080 updates per task as with full data, 9 validations per task), sequential fine-tuning possibly two jobs. With 6 parallel jobs: start the long runs first (distillation, full-data fine-tuning, joint full data, equal-updates runs). A second job is started with `resume_continual.sh`; it continues training or, if training is finished, the remaining evaluations.

## 3. After the runs
Copy the new experiment folders and SLURM logs into `results/hpc/` of the thesis repository (not the `smoke_test/` folders); the
analysis scripts there are extended to read `evaluations/<tag>/` and the joint runs.
