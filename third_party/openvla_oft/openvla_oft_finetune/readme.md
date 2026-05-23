# openvla_oft — finetune reference

For installation and the end-to-end training pipeline see the canonical guides:

- Install: `docs/environment/openvla_oft.md`
- Training (and dataset processing via the RLDS env): `docs/best_practice/train/openvla_oft.md`

The notes below cover reference material that does **not** live in the main
docs — namely, how to register a new dataset in the openvla-oft codebase so
the `finetune_*.sh` launchers can consume it.

## Adding a new dataset

Files to edit inside `third_party/openvla_oft/openvla-oft/`:

- `prismatic/vla/constants.py`
- `prismatic/vla/datasets/rlds/oxe/configs.py`
- `prismatic/vla/datasets/rlds/oxe/transforms.py`

### action and state conventions
- `action` is 10-dim: xyz + 6-DoF rotation + 1-dim gripper.
- `state` is 8-dim: xyz + 3-DoF rotation + 2-dim gripper.

Action uses 6-DoF rotation because it is easier to learn. State stays at 8-dim
because the state layout is harder to change.

The launch command must contain the substring `6drotate` so
`detect_robot_platform()` picks up the right constants.

### 3D-rotation dataset example

Example name: `vase_sponge_test1_oft_dataset`

- action 8d: 3+3+2 (xyz, roll-pitch-yaw, pad, gripper)
- state 8d:  3+3+2 (xyz, roll-pitch-yaw, pad, gripper)

Edit
`third_party/openvla_oft/openvla-oft/prismatic/vla/datasets/rlds/oxe/configs.py`
and add the dataset entry:

```python
"vase_sponge_test1_oft_dataset": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
},
```

Add a matching transform in
`third_party/openvla_oft/openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py`:

```python
def vase_sponge_test1_oft_dataset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close); clip to 0...1 and flip so +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["state"] = trajectory["observation"]["state"]
    return trajectory
```

### 6D-rotation dataset example

Example name: `vase_sponge_test1_oft_6drotate_dataset`

- action 10d: (xyz, 6d-rotation, gripper)
- state 8d:  3+3+2 (xyz, roll-pitch-yaw, pad, gripper). The openvla-oft configs
  file does not provide a 6D-rotation state representation, so the state side
  stays at 8d.

You also need to register the action-dim in
`third_party/openvla_oft/openvla-oft/prismatic/vla/constants.py`:

```python
Rotate6d_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 10,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

and extend `detect_robot_platform()` to return `"Rotate6d"` whenever the
command line contains `6drotate`. The full layout of `constants.py` is left as
upstream — only those two additions are required.
