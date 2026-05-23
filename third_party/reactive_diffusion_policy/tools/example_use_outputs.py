#!/usr/bin/env python3
"""
Example: how to consume the separated fields returned by Policy.predict_action().

Demonstrates how to obtain and use:
1. Robot-control action (10 dims)
2. Tactile-embedding prediction (15 dims)
"""

import torch
import numpy as np

def example_policy_output_usage():
    """
    Mock the output of policy.predict_action() and demonstrate how to use each field.
    """

    # Fake policy output
    batch_size = 4
    horizon = 16
    n_action_steps = 8

    result = {
        'action': torch.randn(batch_size, n_action_steps, 10),  # robot control
        'action_pred': torch.randn(batch_size, horizon, 10),    # full robot trajectory
        'action_pred_full': torch.randn(batch_size, horizon, 25),  # full 25-dim trajectory
        'action_pred_tactile': torch.randn(batch_size, horizon, 15),  # tactile trajectory
        'action_tactile': torch.randn(batch_size, n_action_steps, 15)  # tactile (n_action_steps)
    }

    print("=" * 60)
    print("Policy output example")
    print("=" * 60)

    # 1. Robot-control action (sent to the robot)
    robot_action = result['action']
    print(f"\n1. Robot-control action:")
    print(f"   shape: {robot_action.shape}")
    print(f"   purpose: send to the robot for execution")
    print(f"   dims: (x, y, z, 6d_rotation, gripper_width)")

    # 2. Tactile prediction (analysis)
    tactile_pred = result['action_pred_tactile']
    print(f"\n2. Tactile-embedding prediction:")
    print(f"   shape: {tactile_pred.shape}")
    print(f"   purpose: evaluate how well the model predicts tactile signals")
    print(f"   dims: PCA-reduced 15-dim tactile marker offsets")

    # 3. Tactile for the executed steps
    tactile_action = result['action_tactile']
    print(f"\n3. Tactile embedding (executed steps):")
    print(f"   shape: {tactile_action.shape}")
    print(f"   purpose: tactile info aligned with the steps the robot executes")

    # 4. Full prediction (for debugging)
    full_pred = result['action_pred_full']
    print(f"\n4. Full prediction:")
    print(f"   shape: {full_pred.shape}")
    print(f"   purpose: full prediction containing both robot-control and tactile dims")

    # Verify dimension layout
    assert full_pred[..., :10].shape == result['action_pred'].shape
    assert full_pred[..., 10:].shape == result['action_pred_tactile'].shape
    print(f"\n[OK] Dimension check passed:")
    print(f"  full_pred[:, :, :10] == action_pred")
    print(f"  full_pred[:, :, 10:] == action_pred_tactile")

    return result


def example_evaluation_usage(gt_action_25dim, pred_result):
    """
    Example: how to use the separated outputs at evaluation time.

    Args:
        gt_action_25dim: ground-truth action (25 dims)
        pred_result: output of Policy.predict_action()
    """
    print("\n" + "=" * 60)
    print("Evaluation example")
    print("=" * 60)

    # 1. Evaluate the robot-control action (first 10 dims)
    gt_robot = gt_action_25dim[..., :10]
    pred_robot = pred_result['action_pred']

    robot_mse = torch.nn.functional.mse_loss(pred_robot, gt_robot)
    print(f"\n1. Robot action MSE: {robot_mse.item():.6f}")
    print(f"   GT shape: {gt_robot.shape}")
    print(f"   Pred shape: {pred_robot.shape}")

    # 2. Evaluate the tactile prediction (last 15 dims)
    gt_tactile = gt_action_25dim[..., 10:]
    pred_tactile = pred_result['action_pred_tactile']

    tactile_mse = torch.nn.functional.mse_loss(pred_tactile, gt_tactile)
    print(f"\n2. Tactile-embedding MSE: {tactile_mse.item():.6f}")
    print(f"   GT shape: {gt_tactile.shape}")
    print(f"   Pred shape: {pred_tactile.shape}")

    # 3. Full-action evaluation
    full_mse = torch.nn.functional.mse_loss(pred_result['action_pred_full'], gt_action_25dim)
    print(f"\n3. Full-action MSE: {full_mse.item():.6f}")

    return {
        'robot_mse': robot_mse.item(),
        'tactile_mse': tactile_mse.item(),
        'full_mse': full_mse.item()
    }


def example_robot_control_usage(pred_result):
    """
    Example: how to consume the output during real robot control.
    """
    print("\n" + "=" * 60)
    print("Robot-control example")
    print("=" * 60)

    # Grab the robot-control action
    robot_action = pred_result['action']  # [B, n_action_steps, 10]

    # Assume batch_size == 1 (single prediction)
    robot_action = robot_action[0]  # [n_action_steps, 10]

    print(f"\nRobot executes {robot_action.shape[0]} steps:")
    for step_idx in range(min(3, robot_action.shape[0])):  # print only the first 3 steps
        action_step = robot_action[step_idx]
        print(f"\n  Step {step_idx + 1}:")
        print(f"    position (x, y, z): {action_step[:3].numpy()}")
        print(f"    rotation (6d):     {action_step[3:9].numpy()}")
        print(f"    gripper width:     {action_step[9].item():.4f}")

    # Optional: visualize/monitor with the tactile info
    tactile_info = pred_result['action_tactile'][0]  # [n_action_steps, 15]
    print(f"\nMatching tactile info:")
    print(f"  shape: {tactile_info.shape}")
    print(f"  uses:  visualization, anomaly detection, tactile feedback, ...")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# Policy-output field usage example")
    print("#" * 60)

    # 1. Basic output example
    result = example_policy_output_usage()

    # 2. Evaluation example
    batch_size, horizon = 4, 16
    gt_action_25dim = torch.randn(batch_size, horizon, 25)
    metrics = example_evaluation_usage(gt_action_25dim, result)

    # 3. Robot-control example
    example_robot_control_usage(result)

    print("\n" + "#" * 60)
    print("# Example finished")
    print("#" * 60)
    print("\nSummary:")
    print("  [OK] action:                robot control (10 dims)")
    print("  [OK] action_pred:           evaluation (10 dims)")
    print("  [OK] action_pred_tactile:   tactile analysis (15 dims)")
    print("  [OK] action_tactile:        tactile visualization (15 dims)")
    print("  [OK] action_pred_full:      debug (25 dims)")
    print()
