# XPolicyLab Policy Import Workflow

Use this checklist after `SKILL.md` triggers and the local XPolicyLab reference files have been inspected.

## 1. Workspace

- Derive `<PolicyName>` from the repo unless the user provides one.
- Use the upstream repository basename as `<upstream_project_dir>` by default, for example `diffusion_policy.git` becomes `diffusion_policy`.
- Scaffold with `skills/xpolicylab-import-policy/scripts/scaffold_policy_import.py`.
- Place upstream source at `XPolicyLab/policy/<PolicyName>/<upstream_project_dir>/`, matching `XPolicyLab/policy/DP/diffusion_policy`.
- Delete `XPolicyLab/policy/<PolicyName>/<upstream_project_dir>/.git` immediately.
- Delete every nested `.gitignore` inside the upstream tree, recursively:
  `find XPolicyLab/policy/<PolicyName>/<upstream_project_dir> -name .gitignore -delete`.
  Upstream `.gitignore` files only govern upstream's contributor workflow; in our snapshot they become active filters that any `.gitignore`-respecting sync tool (rsync `--filter=':- .gitignore'`, VSCode/PyCharm remote sync, `tar --exclude-from`) will apply when copying XPolicyLab to another host, silently dropping any upstream-tracked-but-ignored files on the destination. Recursive because upstream often vendors sub-projects that carry their own `.gitignore`. Audit first to know what was at risk:
  ```bash
  git clone <upstream> /tmp/check && cd /tmp/check
  git ls-files | git check-ignore --stdin --no-index
  ```
  If any listed file hard-codes paths to the upstream maintainer's machine (common for accelerate / DeepSpeed yamls), ship a parallel copy under `XPolicyLab/policy/<PolicyName>/xpolicylab_adapter/` and point the adapter scripts at the adapter copy.
- Add `XPolicyLab/policy/<PolicyName>/.gitignore` before the commit gate. Mirror `XPolicyLab/policy/DP/.gitignore` as the minimal baseline (`data/*`, `checkpoints/*`), then append policy-specific large/generated outputs (e.g. `runs/*` for training output roots, `weights/*` for separately-downloaded encoders, training log/cache directories that the policy actually produces). Skip patterns already covered by the parent `XPolicyLab/.gitignore` (`__pycache__/`, `*.egg-info`).
- Commit the freshly scaffolded policy directory in the XPolicyLab git repository before adaptation edits. If this is blocked, report why the snapshot was not committed.
- Do not overwrite existing policy folders without explicit user approval.

## 2. Upstream Audit

Read upstream README, install, train, eval, deploy, config, and checkpoint docs. Locate:

- model construction and config entrypoints;
- checkpoint loading and required weight paths;
- inference APIs and reset/stateful rollout behavior;
- expected observation keys, cameras, image shape, channel order, normalization, and tensor layout;
- robot state preprocessing and action output format;
- dataset format and conversion scripts;
- text/language encoder and embedding generation requirements for language-conditioned policies.

Search upstream source for image and channel-order boundaries:

```bash
rg "cv2\.imread|cv2\.imdecode|VideoCapture|PIL\.Image|rgb|bgr|transpose|moveaxis|normalize|permute|channel" XPolicyLab/policy/<PolicyName>/<upstream_project_dir>
```

Watch for config import side effects such as eager Hugging Face downloads inside Hydra/LazyConfig modules. Replace eager remote resolution with local or lazy/env-resolved paths when downloads are not explicitly approved.

## 3. Dependency Capture

- Prefer a per-policy Conda or uv environment.
- Keep upstream installation scripts intact when present; add thin wrappers or short notes instead of rewriting.
- Translate Docker-only upstream docs into Conda or uv commands; do not add Docker files or Docker commands.
- Keep `INSTALLATION.md` short and command-oriented, like `policy/Pi_05/INSTALLATION.md`.
- Do not put source provenance, import metadata, validation notes, or manual patch logs in `INSTALLATION.md`.
- Do not generate policy-root provenance files such as `IMPORT_NOTES.md`; keep source URL, commit hash, clone date, license notes, and manual patch notes in the git commit message, PR description, or external project notes unless the user explicitly asks for an in-tree record.
- Include commands for text encoder or checkpoint acquisition only as documentation unless the user approves actual downloads.
- Never put absolute paths in scripts or docs.

## 4. Data Processing

- Default sample data is `data/RoboDojo/test_data/arx_x5`.
- General dataset layout is `data/<dataset_name>/<task_name>/<env_cfg_type>/`.
- Keep `process_data.sh` compatible with DP style:

```bash
bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type>
```

- Run conversion in the policy environment, not the XPolicyLab base environment and not `eval_env_conda_env`.
- Prefer:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
)
```

- Use `action_type` (`ee` or `joint`) and `get_robot_action_dim_info(env_cfg_type)` to choose state/action fields and dimensions.
- Keep pose format `[x, y, z, qw, qx, qy, qz]`.
- Store converted data under `XPolicyLab/policy/<PolicyName>/data/...` unless upstream strictly requires another layout.
- Use `tqdm` for slow episode conversion, text embedding generation, and any other multi-item phase. One bar per phase, attach per-item context via `set_postfix(...)` (e.g. `frames`, `total`, `tasks`), and use `tqdm.write(...)` for the final summary. Do not emit per-episode or per-frame `print()` lines — they fight the bar and turn long runs into walls of log. Sample pattern:

  ```python
  from tqdm import tqdm

  bar = tqdm(episodes, desc=f"convert {dataset_id}", unit="ep", dynamic_ncols=True)
  for episode_index, episode_path in enumerate(bar):
      ...
      bar.set_postfix(frames=n_frames, total=total_frames, tasks=len(tasks_index))
  bar.close()
  tqdm.write(f"[convert] wrote {len(episodes)} episodes / {total_frames} frames to {output_root}")
  ```

Image rule for every frame:

```python
img = decode_image_bit(img_bit)
assert img.ndim == 3 and img.shape[-1] == 3
img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
assert img.shape == (240, 320, 3)
```

If using OpenCV readers/decoders that produce BGR, convert explicitly:

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

For language-conditioned policies, preserve the original task description/instruction in conversion output. Placeholder embedding files must be clearly named as placeholders, and real training must fail unless real embeddings from the upstream text encoder exist.

## 5. Model And Deployment Adapter

Implement `model.py` with `class Model(ModelTemplate)`.

- Convert `observation["vision"][camera]["color"]` images to `(240, 320, 3)` RGB before upstream preprocessing.
- Use the same image shape and channel order in training conversion and deployment.
- Use `pack_robot_state`, `unpack_robot_state`, and `get_robot_action_dim_info` for state/action mapping.
- Return actions as a list of per-step action dictionaries from `get_action()`.
- Implement `update_obs_batch()` and `get_action_batch(env_idx_list)` when upstream supports batch inference; otherwise fail loudly with a helpful message.
- Keep `deploy.py` close to `demo_policy/deploy.py` unless the policy requires temporal/chunk-specific rollout logic.

## 6. Scripts And Configs

- `deploy.yml` must keep `policy_name` equal to the folder/import name.
- Use `null` for values overridden by `setup_eval_policy_server.sh`.
- `deploy.yml` must declare `eval_env` (`debug` / `sim` / `real`) and `eval_batch` (`true` / `false`). Switching client mode is done by editing these two fields, not by editing any `.sh` file.
- `process_data.sh` may stay simple and assume the caller has already activated the policy environment. During deployment, the agent must run data-processing tests in the policy environment, not the XPolicyLab base environment or `eval_env_conda_env`.
- `train.sh` should fail loudly when required language embeddings are missing or placeholders. A smoke-test override must be explicitly named, for example `ALLOW_PLACEHOLDER_T5=true`.
- Eval is split into three coordinated scripts (mirror `policy/demo_policy`):
  - `eval.sh` — orchestrator. Accepts `dataset_name task_name ckpt_name env_cfg_type expert_data_num action_type seed policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env`, allocates a `policy_server_port`, backgrounds the server script, runs the client script, and traps EXIT for cleanup. Build `additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"`.
  - `setup_eval_policy_server.sh` — activates the policy environment and `exec`s `XPolicyLab/setup_policy_server.py` with `policy_server_port`, `policy_server_host`, and policy-specific overrides (`checkpoint_path`, `model_path`, etc.). Use `CUDA_VISIBLE_DEVICES="${policy_gpu_id}"` only inside this script.
  - `setup_eval_env_client.sh` — activates `eval_env_conda_env` and delegates to `XPolicyLab/utils/setup_env_client.sh`, which dispatches to `run_debug_env_client.sh`, `run_sim_env_client.sh`, or `run_real_policy_client.sh` based on `deploy.yml`'s `eval_env`. Accepts an optional trailing `policy_server_ip` for cross-machine deployment.
- Keep `task_name` and `ckpt_name` strictly separate: `task_name` is the simulator task forwarded to the env client; `ckpt_name` resolves the checkpoint directory/file in the server script. They may differ (`ckpt_name=cotrain` evaluated on multiple `task_name`s).
- For uv-based policies, follow `policy/Pi_05`: pass `policy_uv_env_path`, activate `${policy_uv_env_path}/.venv/bin/activate` inside `setup_eval_policy_server.sh`, and keep the debug client in `eval_env_conda_env`.
- For Conda-based policies, pass `policy_conda_env` and `eval_env_conda_env` separately.

Current debug-client wrapper signature (called transitively by `setup_eval_env_client.sh` → `setup_env_client.sh` when `eval_env: debug`):

```bash
bash XPolicyLab/utils/run_debug_env_client.sh \
  <eval_batch> <eval_env_conda_env> <policy_server_port> \
  <dataset_name> <task_name> <env_cfg_type> <policy_name> \
  <additional_info> <ROOT_DIR> <seed> <env_gpu_id> \
  [<policy_server_ip>]
```

Use `eval_batch: false` first for single-environment `eval_one_episode`. Switch to `true` only after validating batch methods. `policy_server_ip` defaults to `localhost`; pass it only for cross-machine deployment.

## 7. Reproducibility Notes

Record repo URL, commit hash, clone date, upstream license, checkpoint locations, required model downloads, and manual patches only when useful, and prefer the git commit message, PR description, or external project notes over files in the policy directory. Do not turn `INSTALLATION.md` into a provenance or troubleshooting document, and do not create `IMPORT_NOTES.md` unless the user explicitly asks for it.

Prefer small compatibility patches in adapter files over modifying upstream source. If upstream source must be patched, keep the patch minimal and document why near the patch or in a concise policy-local note.

## 8. Tiered Validation

The full code path must be implemented even when local full training or real-model inference cannot be run. Do not skip `process_data.sh`, `train.sh`, `model.py`, `deploy.py`, or `eval.sh` because a later validation tier is unavailable.

Run every validation tier that is locally feasible:

1. Static checks that do not require heavy policy dependencies:
   - shell syntax checks for generated scripts;
   - Python syntax compile for adapter files;
   - `deploy.yml` `policy_name` equals the folder/import name;
   - `setup_policy_server.py` import path resolves to `XPolicyLab.policy.<PolicyName>.model.Model`;
   - data and deployment paths enforce `(240, 320, 3)` RGB before model preprocessing.
2. Data-conversion smoke tests when the policy environment and required lightweight dependencies are available:
   - load the first HDF5 episode in the policy environment;
   - run the conversion on sample data such as `data/RoboDojo/test_data/arx_x5`;
   - verify converted image/state/action shapes.
3. Model import, server startup, and single-environment debug-client tests when the policy environment, checkpoints, and runtime dependencies are available:
   - set `deploy.yml` to `eval_env: debug`, `eval_batch: false`;
   - run `eval.sh` end to end — it backgrounds `setup_eval_policy_server.sh` and runs `setup_eval_env_client.sh`, which transitively reaches `run_debug_env_client.sh`;
   - confirm model server/client communication, deploy loop, observation encoding, action dictionary shape, and reset flow.
4. Batch debug tests when batch support is implemented and single-environment debug has passed.
5. Real training only when the user explicitly asks and the required compute, data, dependencies, and artifacts are available.

Passing the debug client means the import path, model server/client communication, deploy loop, observation encoding, action dictionary shape, and reset flow are wired. It does not prove checkpoint quality or simulator performance.

Do not deliver a half-finished scaffold as a completed integration. If dependencies, checkpoints, text encoders, embeddings, data, GPU, or environment issues block a validation tier, report the exact blocked command, missing requirement, last tier that passed, and next user action needed to unblock it.

## 9. Common Deployment Failures

| Symptom | Likely cause |
|---------|--------------|
| Connection refused on port | Server not up yet, wrong `policy_server_port`, `policy_server_ip` mismatch across machines, or crash during model init |
| Import error for `Model` | `policy_name` mismatch or missing `model.py` |
| Client unknown `eval_env` | Typo in `deploy.yml`; must be `debug`, `sim`, or `real` |
| Looks for checkpoint under `task_name` | `setup_eval_policy_server.sh` overrides resolve the checkpoint via `task_name` instead of `ckpt_name` — fix to use `ckpt_name` |
| CUDA OOM on server | Wrong `CUDA_VISIBLE_DEVICES` or policy/eval GPUs swapped — set per-script inside `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, never globally export in `eval.sh` |
| Action key errors | `deploy.py` / `Model.get_action` output does not match env expectations; see policy contract reference |
| Arm flies / over-extends past joint limits | Adapter inverts action normalization with the wrong stats family (e.g. `min/max` on a `q99`-mode checkpoint, or vice versa). `dataset_statistics.json` stores all of `min/max/mean/std/q01/q99` regardless of training mode, so picking the wrong pair silently rescales actions by `(used_high-used_low)/(true_high-true_low)`. Mirror the dataloader's `normalization_modes` for the active robot type. See "Action Normalization Mode" in `references/xpolicylab-policy-contract.md`. |
| Gripper appears stuck / never grasps | Adapter honors the saved `mask` field in `dataset_statistics.json` for gripper dims. That mask is often generated from key names (`"gripper" -> False`), not from the trainer's per-element pass-through check (`q01 != q99`). Continuous grippers with `q01=0, q99=1` are normalized normally at training but pass through unchanged at deployment, leaving them in `[-1, 1]` instead of `[0, 1]`. Use `inv_mask = high != low` instead of the saved mask. |
| Padded model output joints get garbage commands | Model emits actions at the per-key-padded width (gr00t-style: `arm->7, gripper_close->1, ...` summed = e.g. 16 for arx_x5) but `dataset_statistics.json` stats are at the raw width (e.g. 14). Adapter must strip per-key pad columns *before* unnormalization and `unpack_robot_state(...)`, otherwise stats and actions misalign silently. See "Action Normalization Mode" → padded vs raw dims. |
| Arm twitches in place but does not execute the task | Deployment image preprocessing diverges from the dataloader's spatial transform — usually skipping a letterbox/`expand2square` step so a `(240, 320)` frame is stretched to `(224, 224)` while training mean-pads then resizes. The diffusion / flow-matching head sees OOD vision conditioning and collapses to ≈mean output, which after unnormalization is approximately the current joint state. Reproduce the upstream loader's full transform exactly inside `_standardize_rgb_image` (same letterbox color, same final size, same interpolation). See "Image Spatial Alignment" in `references/xpolicylab-policy-contract.md`. |
| Arm twitches in place on hard tasks but moves on simple ones; or `state_dict` load fails on `obs_merger.weight` shape mismatch | Deployment `encode_obs` sends a single current frame to a model whose `obs_horizon > 1`. Released checkpoint's obs-merging linear was trained with multi-frame input (e.g. LDA-1B pretrain: `obs_merger.weight.shape = (1536, 1152)` ⇒ `obs_horizon=2` since `1152/384 - 1 = 2`), so the deploy frame count must equal `obs_horizon * num_views`. Common variant: training yaml says `obs_horizon=1` and `pretrained_checkpoint: null`, masking the misconfiguration by silently falling back to from-scratch — the run trains for 100k+ steps but never converges on harder tasks. Sync all three layers: training config (`obs_horizon` matches the pretrain ckpt), dataloader (`observation_indices` length matches `obs_horizon`, e.g. `[-5, 0]` for 2 frames), and adapter (`encode_obs` keeps a per-env per-view rolling buffer, pulls frames at `observation_indices` offsets clamped to `0` at episode start, clears on `reset()`, and flattens view-major / time-minor). See "Observation Window / History Frames" in `references/xpolicylab-policy-contract.md`. |
