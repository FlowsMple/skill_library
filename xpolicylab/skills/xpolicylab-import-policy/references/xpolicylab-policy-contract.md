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

1. Create a policy template from `XPolicyLab/` with `bash create_policy.sh <policy_name>`.
2. Clone/copy the external project under `XPolicyLab/policy/<policy_name>/<upstream_project_dir>`, following the `DP/diffusion_policy` layout.
3. Delete the nested upstream `.git` directory so it is a plain source snapshot, not a submodule or nested repository.
4. Keep `XPolicyLab/policy/` tidy like `XPolicyLab/policy/DP`: put new policy-specific helper scripts inside the policy's original upstream project directory under `XPolicyLab/policy/<policy_name>/<upstream_project_dir>/` whenever practical.
5. Commit `policy/<policy_name>` in the `XPolicyLab` git repository before adaptation edits to preserve the original imported project record.
6. Fill `process_data.sh` to convert trajectories from `data/<dataset_name>/<task_name>/<env_cfg_type>/` into the upstream training format. Use `data/RoboDojo/test_data/arx_x5` by default.
7. Fill `train.sh`.
8. Fill `model.py`, `deploy.py`, `deploy.yml`, and `eval.sh` for inference/deployment.
9. Use `XPolicyLab/debug_env_client.py` through `XPolicyLab/utils/run_debug_env_client.sh` first, then switch from `run_debug_env_client` to `run_policy_client` for simulator deployment.

A completed deployment integration includes `install.sh`, `INSTALLATION.md`, `process_data.sh`, `train.sh`, `model.py`, `deploy.py`, `deploy.yml`, and `eval.sh`. Do not treat these as optional deliverables for deployment-chain work. The code path must be implemented even when local full training or real-model inference cannot be run. If dependencies or artifacts prevent a script from running end to end, keep the script explicit and report the blocked command, missing requirement, last validation tier that passed, and next user action needed.

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

Current wrapper signature:

```bash
bash XPolicyLab/utils/run_debug_env_client.sh \
  <eval_batch> <eval_env_conda_env> <FREE_PORT> \
  <dataset_name> <task_name> <env_cfg_type> <policy_name> \
  <additional_info> <ROOT_DIR> <seed> <env_gpu_id>
```

Use `false` for single-environment flow validation first. Use `true` only after implementing batch methods and wanting to validate `update_obs_batch`/`get_action_batch`.

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

Audit BGR/RGB boundaries carefully. OpenCV image readers and decoders commonly produce BGR, while this project should standardize adapters to RGB. Convert explicitly with:

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

Only skip conversion after proving the current source already provides RGB. Training conversion and deployment/test-time observation encoding must use the same channel order.

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
