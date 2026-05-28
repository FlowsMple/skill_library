# XPolicyLab Policy Contract

## Import Path

Current `XPolicyLab/setup_policy_server.py` loads:

```python
importlib.import_module(f"XPolicyLab.policy.{policy_name}.model")
Model = getattr(module, "Model")
model = Model(deploy_cfg)
```

Therefore `deploy.yml` must contain a `policy_name` matching the policy folder's importable module name, and `model.py` must define an instantiable `Model` class.

## README Integration Steps

The project README defines the custom policy path:

1. Create a policy template from `XPolicyLab/` with `bash create_policy.sh <policy_name>`. The template now ships three coordinated eval scripts (see Eval Script Layout below).
2. Clone/copy the external project under `XPolicyLab/policy/<policy_name>/<upstream_project_dir>`, following the `DP/diffusion_policy` layout.
3. Delete the nested upstream `.git` directory so it is a plain source snapshot, not a submodule or nested repository.
4. Keep `XPolicyLab/policy/` tidy like `XPolicyLab/policy/DP`: put new policy-specific helper scripts inside the policy's original upstream project directory under `XPolicyLab/policy/<policy_name>/<upstream_project_dir>/` whenever practical.
5. Commit `policy/<policy_name>` in the `XPolicyLab` git repository before adaptation edits to preserve the original imported project record.
6. Fill `process_data.sh` to convert trajectories from `data/<dataset_name>/<task_name>/<env_cfg_type>/` into the upstream training format. Use `data/RoboDojo/test_data/arx_x5` by default.
7. Fill `train.sh`.
8. Fill `model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `setup_eval_policy_server.sh`, and `setup_eval_env_client.sh` for inference/deployment.
9. Use `XPolicyLab/debug_env_client.py` through the new three-script eval path (see below) with `deploy.yml` `eval_env: debug` first; switch to `eval_env: sim` (or `real`) when ready for simulator/real deployment — no edits to `eval.sh` required.

A completed deployment integration includes `install.sh`, `INSTALLATION.md`, `process_data.sh`, `train.sh`, `model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `setup_eval_policy_server.sh`, and `setup_eval_env_client.sh`. Do not treat these as optional deliverables for deployment-chain work. The code path must be implemented even when local full training or real-model inference cannot be run. If dependencies or artifacts prevent a script from running end to end, keep the script explicit and report the blocked command, missing requirement, last validation tier that passed, and next user action needed.

## Eval Script Layout

The current `policy/demo_policy` reference splits evaluation into three coordinated scripts. New policies must keep the same split so cross-machine deployment, `eval_env` switching, and `task_name` vs `ckpt_name` separation all work.

- `eval.sh` (orchestrator)
  - Args (in order): `dataset_name task_name ckpt_name env_cfg_type expert_data_num action_type seed policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env`.
  - Allocates a free `policy_server_port` via `XPolicyLab/utils/get_free_port.sh`.
  - Sets `policy_server_ip=localhost` by default.
  - Backgrounds `setup_eval_policy_server.sh`, sleeps briefly, then runs `setup_eval_env_client.sh` in the foreground; traps EXIT to kill the server.
  - Builds `additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"` and forwards it to the client.
- `setup_eval_policy_server.sh` (model side)
  - Activates `policy_conda_env`, computes `action_dim` via `XPolicyLab/utils/get_action_dim.sh`, and `exec`s `XPolicyLab/setup_policy_server.py`.
  - Passes `policy_server_port` and `policy_server_host` as `--overrides`. Add all policy-specific overrides here (`checkpoint_path`, `model_path`, etc.).
  - Uses `CUDA_VISIBLE_DEVICES="${policy_gpu_id}"` only inside this script — do not export it globally in `eval.sh`.
- `setup_eval_env_client.sh` (env side)
  - Activates `eval_env_conda_env` and delegates to `XPolicyLab/utils/setup_env_client.sh`, which reads `eval_env` (and `eval_batch`) from `deploy.yml` and forwards to one of:
    - `run_debug_env_client.sh` for `eval_env: debug`,
    - `run_sim_env_client.sh` for `eval_env: sim`,
    - `run_real_policy_client.sh` for `eval_env: real`.
  - Accepts an optional final `policy_server_ip` argument so the env can target a server on another host.

Cross-machine deployment: run `setup_eval_policy_server.sh` on the GPU host with an explicit `policy_server_port`/`policy_server_host`, then run `setup_eval_env_client.sh` on the simulator host with the same port plus the server's IP as the trailing argument. Same `deploy.yml` controls debug/sim/real on the client side.

### task_name vs ckpt_name

`task_name` is the simulator task name (forwarded to the env client and used in trajectory paths). `ckpt_name` selects the checkpoint and may differ (for example `cotrain` weights evaluated across multiple `task_name` values). The two MUST be wired separately:

- Forward `task_name` only to the env client (already handled by the templates).
- Resolve `checkpoint_path` from `ckpt_name` (and `dataset_name` / `env_cfg_type` / `seed`) in `setup_eval_policy_server.sh` `--overrides`.
- Never use `task_name` as a checkpoint directory key.

## Environment Contract

Policy environments can use Conda or uv. Docker is not allowed for setup, training, deployment, or validation in this project.

Rules:

- Keep one independent environment per policy for policy-specific dependencies.
- Run policy-specific data conversion in that same policy environment. `process_data.sh` belongs to the policy because each policy has different preprocessing, codecs, tensor layouts, dataset formats, and training dependencies.
- Codex writes and updates environment installation scripts and documentation, but the user controls actual environment installation. Do not run `install.sh`, `uv sync`, `pip install`, `conda env create`, `conda create`, or equivalent dependency installation commands unless the user explicitly asks for installation in the current turn.
- If the upstream project already contains `install.sh` or related installation scripts, preserve them and make the smallest necessary edits for XPolicyLab compatibility.
- Do not download weights, checkpoints, text encoders, or other large pretrained artifacts without explicit user approval in the current turn. Document what is required and wait for approval before downloading.
- Keep the policy environment separate from the XPolicyLab project/base environment.
- Keep the evaluation/debug client environment separate from the policy environment.
- The XPolicyLab environment is responsible for the upper-layer simulator, real-robot client, and debug/eval client flow. It should not be used as an implicit fallback for policy data conversion.
- Conda-based policies usually pass `policy_conda_env` and `eval_env_conda_env` through `eval.sh`.
- uv-based policies can follow `policy/Pi_05`: pass `policy_uv_env_path`, activate `${policy_uv_env_path}/.venv/bin/activate`, then start the model server.
- Run model server commands after activating the policy environment.
- Run debug/eval client commands after activating `eval_env_conda_env`.
- If upstream documentation is Docker-only, translate dependency installation to Conda or uv.
- Keep `INSTALLATION.md` concise and command-oriented. Follow `policy/DP/install.sh` and `policy/Pi_05/INSTALLATION.md`: show installation commands only, with minimal headings if helpful. Do not put long policy design notes, adapter explanations, source metadata, validation results, or troubleshooting narratives in `INSTALLATION.md`.
- Keep `INSTALLATION.md`, `install.sh`, and other generated docs/scripts portable. Do not include absolute paths; write commands relative to the repository root, policy folder, or upstream source folder.
- Do not add Dockerfiles, compose files, or scripts that require Docker for the integration.

## Debug Client Gate

`XPolicyLab/debug_env_client.py` is the required fake-environment validation client. It verifies the full local deployment loop before real simulator evaluation:

- starts a `TestEnv` with mock observations;
- connects to the model server through `ModelClient`;
- imports `XPolicyLab.policy.<policy_name>.deploy`;
- calls `eval_one_episode` or `eval_one_episode_batch`;
- sends mock camera observations with `cam_head`, `cam_left_wrist`, and `cam_right_wrist` color images shaped `(480, 640, 3)`;
- validates returned action dictionary keys and dimensions with `validate_robot_state_dict`;
- runs 10 short mock episodes with `episode_step_limit = 5`.

Current wrapper signature (called by `setup_env_client.sh` after reading `eval_env: debug` from `deploy.yml`):

```bash
bash XPolicyLab/utils/run_debug_env_client.sh \
  <eval_batch> <eval_env_conda_env> <policy_server_port> \
  <dataset_name> <task_name> <env_cfg_type> <policy_name> \
  <additional_info> <ROOT_DIR> <seed> <env_gpu_id> \
  [<policy_server_ip>]
```

`policy_server_ip` defaults to `localhost`; pass it explicitly only for cross-machine setups. In normal usage you do not invoke this script directly — the policy's `setup_eval_env_client.sh` calls `setup_env_client.sh`, which dispatches based on `deploy.yml`'s `eval_env`. Set `eval_batch: false` in `deploy.yml` first; switch to `true` only after implementing and validating `update_obs_batch`/`get_action_batch`.

Passing this gate means the import path, server/client communication, deploy loop, observation encoding, action return shape, and reset flow are wired correctly. It does not prove checkpoint quality or simulator performance.

## Model Interface

Implement `class Model(ModelTemplate)` with:

- `__init__(self, model_cfg)`: load config, build upstream model, load checkpoint, initialize action/observation metadata.
- `update_obs(self, obs)`: encode one environment observation and store it in the upstream policy.
- `update_obs_batch(self, obs_list)`: encode batch observations when supported.
- `get_action(self)`: return a list of action dictionaries, one dictionary per rollout step.
- `get_action_batch(self, env_idx_list)`: return a list per environment, each containing per-step action dictionaries.
- `reset(self)`: clear temporal state between episodes.

Use:

```python
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
```

## Observation Shape

Common environment observation shape:

```python
obs["vision"][camera_name]["color"]  # HWC RGB/BGR-like uint8 image, camera name varies
```

Model adapters must standardize camera images before upstream preprocessing:

- Use RGB as the standard channel order.
- Use width 320, height 240 for every frame.
- Assert HWC shape `(240, 320, 3)`.
- Then move image axis to CHW, normalize to `0..1`, or apply upstream-specific preprocessing.

Reference pattern from `XPolicyLab/policy/DP/diffusion_policy/process_data.py`:

```python
img = decode_image_bit(img_bit)
assert img.ndim == 3 and img.shape[-1] == 3
img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
assert img.shape == (240, 320, 3)
```

### Channel Order Convention (critical, project-specific)

XPolicyLab images arrive RGB-ordered at every boundary an adapter touches:

- `obs["vision"][cam]["color"]` from the live env/debug client/simulator is RGB.
- `XPolicyLab.utils.process_data.decode_image_bit(...)` returns **RGB**, *not* the
  BGR that OpenCV's default `cv2.imdecode(..., IMREAD_COLOR)` documentation
  implies. The HDF5 producer encodes RGB arrays through `cv2.imencode` (which
  OpenCV labels as BGR-in), so the bytes on disk place R/B in the "BGR" slots
  that `cv2.imdecode` reads back — net effect is an RGB-ordered return value.
  This was confirmed by visual round-trip against `data/<dataset>/<task>/<env>/preview_video/*.mp4`
  (the preview videos are the ground-truth color reference).

Therefore in a new policy adapter:

- **Do not** insert `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` right after
  `decode_image_bit(...)` or `cv2.imdecode(...)` of XPolicyLab HDF5 bytes, and
  do not insert it right after reading `obs["vision"][cam]["color"]`. Doing so
  silently swaps R and B in every training frame and every inference frame.
  Train/deploy may stay self-consistent (both swapped), but the upstream
  backbone (e.g. a Qwen-VL / SigLIP / DINO encoder trained on natural RGB)
  will see swapped colors and degrade.
- **Only** insert `BGR2RGB` when the source is provably BGR — e.g. a real-robot
  camera SDK that documents BGR output, or `cv2.VideoCapture(...).read()` on
  an arbitrary mp4 that you did not write yourself.
- Verify in one shot before trusting either direction:

  ```python
  from PIL import Image
  from XPolicyLab.utils.process_data import decode_image_bit

  arr = decode_image_bit(hdf5_colors[0])      # already RGB-ordered for this project
  Image.fromarray(arr, mode="RGB").save("/tmp/check.png")
  # Open /tmp/check.png and compare to data/<dataset>/<task>/<env>/preview_video/...mp4
  # If colors match, the channel order is correct; do NOT add cvtColor.
  ```

Training conversion and deployment/test-time observation encoding must use the
same channel order. If they differ (one swapped, one not), the model will fail
silently — symptoms look like "checkpoint loaded fine but rollout is garbage".

Historic precedent: `XPolicyLab/policy/LDA_1B/xpolicylab_adapter/fix_video_channels.py`
exists specifically to repair already-produced LeRobot videos that were
generated under the wrong "BGR2RGB after decode_image_bit" assumption.
`XPolicyLab/policy/RISE/{model.py,process_data.py}` carried the same bug and
were fixed alongside this doc note. Audit any other policy that does
`cv2.cvtColor(... COLOR_BGR2RGB)` immediately after `decode_image_bit` or
after `obs["vision"][cam]["color"]` — known suspects at the time of writing:
`policy/RDT_1B/rdt/data/aloha/hdf5totfrecords.py`,
`policy/HoloBrain/RoboOrchardLab/projects/holobrain/process_data.py`,
`policy/Cosmos_Policy/cosmos-policy/process_data.py`.

### Image Spatial Alignment (critical, often missed)

XPolicyLab's standard frame is `(240, 320, 3)` RGB at process-data time, but
many upstream training pipelines apply *additional* spatial transforms inside
the dataloader before the visual backbone sees the image — most commonly:

- **Letterbox / square-padding**: pad the rectangular frame to a square with a
  fixed background color (often the ImageNet mean `(0.485, 0.456, 0.406) * 255`),
  then resize to the model's actual input size (typically `224x224`). This
  preserves aspect ratio.
- **Center crop + resize**, **stride-aware resize**, **letterbox to a different
  input size** (e.g. SigLIP's `384x384`), etc.

If the deployment adapter only does `cv2.resize(..., (320, 240))` and trusts
the upstream HF image processor or `resize_images(...)` to handle the rest,
those processors will silently **stretch the aspect ratio** to the target
square. Training and deployment then see two different visual distributions:
one with proper aspect plus mean-color borders, one with a 4:3 rectangle
squashed to a square. The diffusion / flow-matching head reacts to OOD vision
conditioning by collapsing to a near-mean prediction, which after action
unnormalization is approximately the *current* joint state — visible in the
simulator as "arm sits there and twitches in `O(1e-3)` rad oscillations,
never executes the task".

Therefore in a new policy adapter:

- **Read the dataloader's full visual transform pipeline**, not just the
  channel-order step. For LDA-1B that means
  `lda/dataloader/gr00t_lerobot/datasets.py` — the
  `expand2square(image, mean_color)` + `image.resize((224, 224))` block
  appears in the `get_step_data_with_transform(...)` and the multi-task
  `__getitem__(...)` paths and runs on every frame at training time. Same
  pattern in many gr00t-derived loaders.
- **Reproduce the exact spatial transform inside the adapter's
  `_standardize_rgb_image(...)` / `encode_obs(...)`** — same letterbox color,
  same final size, same interpolation family (PIL's default `Image.resize`
  is bilinear, equivalent to `cv2.INTER_LINEAR`). Do not let the HF
  processor / `resize_images` do the resizing for you, because they do not
  apply the letterbox.
- **Make the downstream resize a no-op** by feeding the model an image that
  already matches its `image_size` (e.g. `224x224`). DINOv3/SigLIP/Qwen-VL
  processors will then short-circuit the resize and the only normalization
  applied is the ImageNet mean/std, which matches training.
- Verify by saving a deployment frame and a training frame side-by-side
  before trusting the rollout:

  ```python
  from PIL import Image
  Image.fromarray(deploy_frame).save("/tmp/deploy.png")  # what the adapter feeds the model
  Image.fromarray(train_frame).save("/tmp/train.png")    # what the dataloader emits
  # Compare visually: both must be the same size, same aspect handling,
  # and roughly the same brightness/border treatment.
  ```

  A cheaper numeric sanity check is to feed both through the same vision
  encoder and verify the final-token L2 norms / activation distributions
  agree to within a few percent.

Historic precedent: `XPolicyLab/policy/LDA_1B/model.py:_standardize_rgb_image`
shipped initially with `cv2.resize → (240, 320)` only. Combined with a wrong
action unnormalization (see next section) the symptom was "arm flies out of
joint limits"; once the unnormalization was fixed, the residual bug surfaced
as "arm twitches in place". Adding `_expand2square_uint8(...)` + a final
`cv2.resize → (224, 224)` to mirror the dataloader resolved both. Audit any
other policy whose adapter resizes to the dataset/parquet resolution
(`(240, 320)`) instead of the model input resolution (`(224, 224)` or
similar) — the symptom on a freshly trained checkpoint is a robot that
appears to "freeze" after the first chunk.

## Trajectory Data For Processing

Use `data/RoboDojo/test_data/arx_x5` as the sample dataset when no policy-specific data is supplied. `RoboDojo` is the default `dataset_name`, and dataset directories should follow `data/<dataset_name>/<task_name>/<env_cfg_type>/`.

Expected relevant paths:

- `data/RoboDojo/test_data/arx_x5/data/episode_0000000.hdf5`
- `data/RoboDojo/test_data/arx_x5/data/episode_0000001.hdf5`
- `data/RoboDojo/test_data/arx_x5/data/episode_0000002.hdf5`
- `data/RoboDojo/test_data/arx_x5/traj_data/*.pkl`
- `data/RoboDojo/test_data/arx_x5/preview_video/*`

Read HDF5 through `XPolicyLab.utils.load_file.load_hdf5` in an environment with `h5py`. Decode encoded image bytes with `decode_image_bit`.

Every converted camera/video frame should be resized to `(240, 320, 3)` RGB before saving to the upstream training format. If the upstream stores NCHW tensors, first standardize as HWC RGB `(240, 320, 3)`, then transform to NCHW.

Run data conversion smoke tests inside the policy environment, not in the XPolicyLab evaluation/client environment.

Converted policy training data should default to the policy folder's `data/` directory, following `XPolicyLab/policy/DP/data/...`. Use an external converted-data location only when required by the upstream policy layout.

The debug client sends `480x640x3` mock camera images, so deployment adapters must perform the same resize/RGB standardization at inference time, not only during training data conversion.

## Action Shape

Adapters should return a list of action dictionaries. Use `unpack_robot_state(vector, action_type, robot_action_dim_info, source_type="obs")` when upstream outputs a flat vector.

Single-arm examples generally use `arm_joint_state` or `ee_pose` plus `ee_joint_state`. Dual-arm examples generally use `left_ee_pose`, `right_ee_pose`, `left_ee_joint_state`, `right_ee_joint_state`, or joint-state equivalents. Confirm exact keys from the active environment.

### Action Normalization Mode (critical, often missed)

Most VLA policies (LDA-1B, Pi-0.5, gr00t-derived stacks, OpenVLA-OFT, etc.)
normalize their action targets at training time before feeding them to the
diffusion / regression head, then expect the deployment adapter to invert
that normalization on the model output. The bug surface here is wide:

1. **Wrong stats family.** `dataset_statistics.json` (or the equivalent
   per-checkpoint stats file) typically stores **all** of `min`, `max`,
   `mean`, `std`, `q01`, `q99` regardless of which one was actually used at
   training time. The mode is encoded in the dataloader's transform —
   `normalization_modes={key: "q99"}` for `q99`, `"min_max"` for min-max,
   `"mean_std"` for z-score, `"binary"` for thresholded grippers. If the
   adapter inverts with the wrong family, every action is rescaled by a
   factor of `(used_high - used_low) / (true_high - true_low)`, which for
   joints with skewed distributions can be ~2x and pushes outputs past joint
   limits ("robot flies"). For arx_x5 in LDA-1B, `min/max` range was 2.03x
   the `q01/q99` range — `norm=+1` mapped to `+1.754 rad` instead of the
   correct `+0.543 rad`, a 70° error.

2. **Misleading saved `mask` field.** Some loaders write a `mask` key into
   the saved stats that *looks* like the per-element on/off mask the
   `Normalizer` uses internally (typically `q01 != q99` to skip degenerate
   dims), but is actually generated from **key names** — e.g. gr00t's
   `generate_action_mask_for_used_keys` sets `False` for any key with
   `"gripper"` in it. For an arx_x5 gripper with `q01=0`, `q99=1` the
   trainer normalized it normally (`{0,1} -> {-1,1}`) but the saved mask
   says "passthrough" — using that mask at deployment leaves the gripper in
   `[-1, 1]` instead of `[0, 1]` and the simulator either rejects it or
   never actuates the gripper. Reconstruct the real mask from
   `q01 != q99` (or whatever the trainer's `Normalizer` actually checks),
   not from the saved field.

3. **Padded vs raw dims.** Several gr00t-style loaders pad each action key
   to a per-key max width (`pad_action_state_with_key`: arm→7,
   gripper_close→1, ...) and concatenate, so the *model output* dimension
   (e.g. arx_x5 = `7+1+7+1 = 16`) is wider than the *robot's raw action*
   (`6+1+6+1 = 14`). The saved stats are usually keyed at the **raw**
   width, while the model emits at the **padded** width. Strip the per-key
   padding columns *before* unnormalization and *before*
   `unpack_robot_state(...)`, otherwise the stats vector and the action
   vector will silently misalign and the only signal is "some joints get
   correct commands, the padding-slot joints get garbage".

4. **Saved `[:, idx] = where(... < 0.5, 0, 1)` gripper threshold.** Some
   upstream "official" `unnormalize_actions` helpers (e.g. for binary
   grippers) hard-code an index like `[:, 6]`. That index is **only valid
   for the specific embodiment those helpers were written for**. Copy-paste
   onto a different embodiment (different dim layout, or a continuous
   gripper) and the threshold either lands on the wrong column or
   discretizes a continuous gripper command. If the adapter inherits one of
   these helpers, double-check the layout assumption against the active
   `env_cfg_type`.

Concrete recipe for a new policy:

- Locate the dataloader's `normalization_modes={...}` for the active robot
  type (often in `data_config.py`, keyed on `env_cfg_type` or
  `embodiment_tag`). Note the mode **per action key**.
- Locate the `Normalizer.forward(...)` (or equivalent) implementation and
  understand the per-element pass-through condition (`q01 != q99`,
  `std != 0`, `min != max`).
- Implement the inverse in the adapter as a clean mirror — for `q99` mode:

  ```python
  low  = np.asarray(stats["q01"], dtype=np.float64)
  high = np.asarray(stats["q99"], dtype=np.float64)
  clipped = np.clip(model_normalized_output, -1.0, 1.0)
  inv_mask = high != low                              # match Normalizer's internal mask
  raw = np.where(inv_mask,
                 0.5 * (clipped + 1.0) * (high - low) + low,
                 clipped)
  ```

  Do **not** read the saved `mask` field unless you have verified it matches
  the trainer's actual per-element check.
- After unnormalization, sanity-check the output range against the dataset
  raw stats (q01/q99 or min/max), and against `env_cfg/robot/_robot_info.json`
  joint limits for the active robot. A printed histogram of one chunk's
  output is enough to catch most mistakes.

Historic precedent: `XPolicyLab/policy/LDA_1B/model.py:_normalize_actions`
originally preferred `min/max` (copy-paste from
`LDA-1B/examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py`
which is for a `min_max`-mode config) on a checkpoint trained with `q99`
mode — fixing the inversion stats family was the first half of getting
arx_x5 simulation rollouts working. The same file also originally honored
the saved `mask` for grippers, leaving them in `[-1, 1]`; the second-half
fix was to ignore the saved mask and use `inv_mask = high != low`.

### Observation Window / History Frames (critical, often missed)

Many VLA / diffusion policies condition the action head on **multiple
observation frames**, not just the current frame. The temporal contract is
spread across three components and they must agree exactly:

1. **Model architecture (training-time):** A configurable `obs_horizon`
   (a.k.a. `n_obs_steps`, `history_frames`, `past_obs_window`) sets the
   width of the obs-merging layer or the time dimension of a 3D vision
   patchifier. For LDA-1B / `MMDiT_ActionHeader`, `obs_merger` is a
   `Linear(num_chans * (obs_horizon + 1), input_embedding_dim)` —
   `obs_horizon=2` produces a `1152→1536` linear (`num_chans=384` for
   DINOv3-ViT-S, the `+1` is the noisy next-obs slot for video co-train),
   `obs_horizon=1` produces a `768→1536` linear, and the two are not
   load-compatible.
2. **Dataloader (training-time):** `observation_indices` (or
   `obs_indices`, `history_offsets`, ...) selects which past frames are
   stacked into each training sample. The list **length must equal**
   `obs_horizon`, and the offsets are relative to the current frame
   (negative = past, last entry conventionally `0`). gr00t's standard
   convention is to clamp negative indices to `0` at episode boundaries
   (replicate the first frame). For LDA-1B's `ArxX5DataConfig` the value
   is `[-5, 0]`; other embodiment configs in the same file use the same
   `[-5, 0]` window. Single-frame `[0]` is also legal and corresponds to
   `obs_horizon=1`.
3. **Deployment adapter (inference-time):** `encode_obs` must build a
   flat list of `V * obs_horizon` frames in the order the upstream
   `predict_action` reshapes them — for LDA-1B's `Qwen_MMDiT.predict_action`
   the contract is `rearrange(curr_imgs, "b (v t) c h w -> b v t c h w", v=num_views)`,
   i.e. **view-major / time-minor**:
   `[v0_t0, v0_t1, ..., v0_t(T-1), v1_t0, ..., v(V-1)_t(T-1)]`. The
   adapter therefore needs a per-env per-view rolling image buffer (kept
   across `update_obs(...)` calls within an episode and **cleared on
   `reset()`**) and must pull frames at the same offsets the trainer
   used.

Failure modes when these three drift:

- **Single-frame deploy adapter against a multi-frame model.** Most
  common: copy-pasting `images = [extract_camera(obs, name) for name in cams]`
  from a single-frame example and forgetting that the released checkpoint
  was trained with `obs_horizon=2`. The flat list is half the expected
  length, so `predict_action`'s `(B, V*T, C, H, W)` reshape silently
  collapses to `T=1`, the obs-merger linear receives the wrong number of
  channels, and downstream it either errors out at `obs_merger.weight`
  matmul or — if the deploy model was *itself* re-instantiated with
  `obs_horizon=1` to "fix" the shape — runs through cleanly but only
  loads ~half the trained weights, sending the diffusion head OOD and
  collapsing rollouts to ≈mean output (the tell-tale "trembling in
  place").

- **Training yaml ↔ released checkpoint disagree on `obs_horizon`.** The
  `<run>/config.yaml` is the *attempted* config; the released checkpoint
  may have been trained on a different value (e.g. LDA-1B's pretrain
  ckpt has `obs_merger.weight.shape = (1536, 1152)` ⇒ `obs_horizon=2`,
  but a from-scratch fine-tune yaml might say `obs_horizon=1`). When
  `pretrained_checkpoint` is non-null, `load_pretrained_backbones`
  refuses the partial load with a shape mismatch on `obs_merger.weight`
  and the training run silently falls back to from-scratch — masking
  the misconfiguration. **Verify by inspecting the checkpoint:**

  ```python
  sd = torch.load("LDA-pretrain.pt", map_location="cpu")
  w = sd.get("model", sd)["action_model.obs_merger.weight"]
  num_chans = vision_encoder_hidden_size  # 384 for DINOv3-ViT-S
  obs_horizon = w.shape[1] // num_chans - 1
  ```

- **Buffer not cleared on episode reset.** A rolling buffer keyed only
  on view (or held in a single global list) leaks frames from the
  previous episode through the `[-5, 0]` history slot, so the first
  ~5 steps of every new episode see the *previous* task's last frames
  as "history". Symptom is a few seconds of nonsensical actions at the
  start of every episode, then it self-corrects. Always key the buffer
  on `(env_idx, camera_name)` and clear in `reset()`.

- **Wrong frame ordering at multi-view × multi-time.** Mixing up
  view-major vs time-major when `V > 1` and `T > 1` will not crash (the
  flat length is the same) but feeds the wrong frame into each view
  slot. Always cross-check against the upstream `predict_action`'s
  `rearrange` pattern.

Concrete recipe when importing a multi-frame VLA:

- Open the dataloader config for the active embodiment, read
  `observation_indices` (length = `obs_horizon`, last entry should be
  `0`). Same place will tell you the boundary policy (clamp to first
  frame is the gr00t default).
- Open the upstream `predict_action` (or the head's `forward`) and
  copy the exact `rearrange` pattern that consumes `curr_imgs` — this
  is the deploy frame-ordering contract.
- In the adapter, allocate `self._image_history: dict[(env_idx, cam_name),
  list[ndarray]]`, append on each `update_obs(...)`, trim to
  `abs(min(obs_indices)) + 1` frames, pull frames at each offset with
  negative indices clamped to `0`, and clear the dict in `reset()`.
- Sanity-check the produced flat-image list length equals
  `num_views * obs_horizon` before passing to `predict_action`.

Historic precedent: `XPolicyLab/policy/LDA_1B/model.py:encode_obs`
originally returned `[extract_camera(obs, name) for name in cams]` — a
single current frame per view — which collided with the LDA-1B pretrain
checkpoint's `obs_horizon=2`. Combined with `<run>/config.yaml` claiming
`obs_horizon: 1` and `pretrained_checkpoint: null` (the bundled
`LDA-pretrain.pt` had been downloaded but the train script was not
loading it because the env-var override was unset), the training run
silently fell back to from-scratch and the deploy adapter silently
fed half the expected frames. The result was a perfectly clean stack
trace of zero errors and a robot that twitched in place on harder
tasks. The fix touched all three layers: train script (auto-detect and
load `LDA-pretrain.pt`, default `LDA_OBS_HORIZON=2`),
`ArxX5DataConfig.observation_indices` (`[0]` → `[-5, 0]`), and
`Model.encode_obs` (single frame → per-env rolling buffer with
view-major / time-minor flattening).
