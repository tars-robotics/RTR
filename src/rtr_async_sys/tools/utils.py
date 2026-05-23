from typing import List, Dict, Any
import numpy as np

def mean_abs_delta_xyz(plot_actions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute mean absolute delta_x/y/z, where delta = predict - fact.
    plot_actions: formatted as [{'fact': [xyz_mm, ...], 'predict': [xyz_mm, ...]}, ...]
                 xyz_mm is a length-3 array/list in millimeters, already multiplied by 1000
    Return: {'mean_abs_delta_x': ..., 'mean_abs_delta_y': ..., 'mean_abs_delta_z': ...}
    """
    if len(plot_actions) == 0:
        return {"mean_abs_delta_x": float("nan"), "mean_abs_delta_y": float("nan"), "mean_abs_delta_z": float("nan")}

    deltas = []
    for item in plot_actions:
        fact = np.asarray(item["fact"], dtype=np.float32)      # [T,3]
        pred = np.asarray(item["predict"], dtype=np.float32)   # [T,3]
        if fact.shape != pred.shape:
            raise ValueError(f"fact/predict shape mismatch: {fact.shape} vs {pred.shape}")
        if fact.ndim != 2 or fact.shape[1] < 3:
            raise ValueError(f"expect [T,3+] but got {fact.shape}")
        deltas.append(pred[:, :3] - fact[:, :3])               # [T,3]

    deltas = np.concatenate(deltas, axis=0)                    # [N,3]
    mad = np.mean(np.abs(deltas), axis=0)                      # [3]
    return {
        "mean_abs_delta_x": float(mad[0]),
        "mean_abs_delta_y": float(mad[1]),
        "mean_abs_delta_z": float(mad[2]),
    }