"""
If the first few visualized points collapse together, the issue may be the data rather than the policy: the episode may pause too long at the beginning.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("--mode", type=int, default=0, help="0 for draw; 1 for only get delta")
parser.add_argument("--pkl_path", type=str, default="outputs/vis_outputs/dp_plot_actions.pkl")
parser.add_argument("--output_dir", type=str, default="outputs/vis_outputs/dp_plot_xyz")
parser.add_argument("--vis_len", type=int, default=200, help="number of vis imgs")

args = parser.parse_args()
mode = args.mode

# ========= Config: choose view =========
# optional: "xy", "xz", "yz", "xyz"
view = "xyz"

# Output directory
# pkl_path = 'data/outputs/vis_outputs/plot_actions.pkl'
# output_dir = f'data/outputs/vis_outputs/plot_{view}'

# pkl_path = 'outputs/vis_outputs/dp_plot_actions.pkl'
# output_dir = f'outputs/vis_outputs/dp_plot_{view}'
pkl_path = args.pkl_path
output_dir = args.output_dir
vis_len = args.vis_len

# pkl_path = 'data/outputs/vis_outputs/dp_plot_actions_reverse.pkl'
# output_dir = f'data/outputs/vis_outputs/dp_plot_{view}_reverse'

os.makedirs(output_dir, exist_ok=True)

# ========= Define 2D view mapping =========
VIEW_MAP_2D = {
    "xy": (0, 1, "X axis", "Y axis"),
    "xz": (0, 2, "X axis", "Z axis"),
    "yz": (1, 2, "Y axis", "Z axis"),
}

# ========= Load pkl =========
with open(pkl_path, 'rb') as f:
    plot_actions = pickle.load(f)

dx_list = []
dy_list = []
dz_list = []
l1_list = []

# ========= Iterate over each timestep =========
for i, step in enumerate(tqdm(plot_actions, desc="Processing steps")):
    if i>=vis_len:
        break
    fact = np.array(step['fact'])      # [N, 3]
    predict = np.array(step['predict'])# [N, 3]

    # step_l1 = np.mean(np.abs(fact - predict))
    step_l1 = torch.mean(torch.abs(torch.Tensor(fact[None,:,:6]) - torch.Tensor(predict[None,:,:6])))
    l1_list.append(step_l1)

    fact[:,0:3] *= 1000 
    predict[:,0:3] *= 1000 

    # ======= Compute prediction error delta = fact_end - predict_end =======
    fx1, fy1, fz1 = fact[-1][0:3]
    px1, py1, pz1 = predict[-1][0:3]

    for j in range(fact.shape[0]):

        dx = fact[j,0] - predict[j,0]
        dy = fact[j,1] - predict[j,1]
        dz = fact[j,2] - predict[j,2]

#     err_norm = np.sqrt(dx*dx + dy*dy + dz*dz)
        # print(f"[Step {j}] Δ(pred error): Δx={dx:.5f}, Δy={dy:.5f}, Δz={dz:.5f}")
        dx_list.append(abs(dx))
        dy_list.append(abs(dy))
        dz_list.append(abs(dz))

    if mode == 1:
        continue

    # ========= 3D view =========
    if view == "xyz":
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        N_fact = len(fact)
        N_pred = len(predict)

        # fact
        ax.scatter(fact[:,0], fact[:,1], fact[:,2],
                   c=cm.Blues(np.linspace(0.4, 1.0, N_fact)),
                   s=18, label="fact")
        ax.plot(fact[:,0], fact[:,1], fact[:,2],
                color="blue", alpha=0.6)

        # predict
        ax.scatter(predict[:,0], predict[:,1], predict[:,2],
                   c=cm.Oranges(np.linspace(0.4, 1.0, N_pred)),
                   s=18, label="predict")
        ax.plot(predict[:,0], predict[:,1], predict[:,2],
                color="orange", alpha=0.6)

        # ===== start/end annotations =====
        fx0, fy0, fz0 = fact[0][0:3]
        px0, py0, pz0 = predict[0][0:3]

        ax.scatter(fx0, fy0, fz0, color="cyan", s=60, edgecolors="black")
        ax.text(fx0, fy0, fz0, "fact_start", color="cyan")

        ax.scatter(fx1, fy1, fz1, color="navy", s=60, edgecolors="black")
        ax.text(fx1, fy1, fz1, "fact_end", color="navy")

        ax.scatter(px0, py0, pz0, color="yellow", s=60, edgecolors="black")
        ax.text(px0, py0, pz0, "pred_start", color="gold")

        ax.scatter(px1, py1, pz1, color="red", s=60, edgecolors="black")
        ax.text(px1, py1, pz1, "pred_end", color="red")

        # coordinate axes
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title(f"XYZ Trajectory - step {i}")
        ax.legend()

        # coordinate range
        all_pts = np.vstack([fact, predict])
        mins, maxs = all_pts.min(0), all_pts.max(0)
        ax.set_xlim(mins[0]-0.05, maxs[0]+0.05)
        ax.set_ylim(mins[1]-0.05, maxs[1]+0.05)
        ax.set_zlim(mins[2]-0.05, maxs[2]+0.05)

        ax.view_init(elev=25, azim=135)

        save_path = os.path.join(output_dir, f"plot_xyz_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        # break
        continue

    # ========= 2D views xy / xz / yz =========
    dim1, dim2, label1, label2 = VIEW_MAP_2D[view]

    fact_2d = fact[:, [dim1, dim2]]
    pred_2d = predict[:, [dim1, dim2]]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # fact
    N_fact = len(fact_2d)
    ax.scatter(fact_2d[:,0], fact_2d[:,1],
               c=cm.Blues(np.linspace(0.4,1.0,N_fact)),
               s=18, label="fact")
    ax.plot(fact_2d[:,0], fact_2d[:,1],
            color="blue", alpha=0.5)

    # predict
    N_pred = len(pred_2d)
    ax.scatter(pred_2d[:,0], pred_2d[:,1],
               c=cm.Oranges(np.linspace(0.4,1.0,N_pred)),
               s=18, label="predict")
    ax.plot(pred_2d[:,0], pred_2d[:,1],
            color="orange", alpha=0.5)

    # start & end
    ax.scatter(fact_2d[0,0], fact_2d[0,1],
               color="cyan", s=60, edgecolors="black")
    ax.text(fact_2d[0,0], fact_2d[0,1], "fact_start", color="cyan")

    ax.scatter(fact_2d[-1,0], fact_2d[-1,1],
               color="navy", s=60, edgecolors="black")
    ax.text(fact_2d[-1,0], fact_2d[-1,1], "fact_end", color="navy")

    ax.scatter(pred_2d[0,0], pred_2d[0,1],
               color="yellow", s=60, edgecolors="black")
    ax.text(pred_2d[0,0], pred_2d[0,1], "pred_start", color="gold")

    ax.scatter(pred_2d[-1,0], pred_2d[-1,1],
               color="red", s=60, edgecolors="black")
    ax.text(pred_2d[-1,0], pred_2d[-1,1], "pred_end", color="red")

    # coordinate axes
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(f"{view.upper()} Trajectory - step {i}")
    ax.legend()

    all_pts = np.vstack([fact_2d, pred_2d])
    mins, maxs = all_pts.min(0), all_pts.max(0)
    ax.set_xlim(mins[0]-0.05, maxs[0]+0.05)
    ax.set_ylim(mins[1]-0.05, maxs[1]+0.05)
    ax.set_aspect("equal", "box")

    save_path = os.path.join(output_dir, f"plot_{view}_{i:03d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[✓] Saved {save_path}")

dx_mean = sum(dx_list) / len(dx_list)
dy_mean = sum(dy_list) / len(dy_list)
dz_mean = sum(dz_list) / len(dz_list)
l1_mean = sum(l1_list) / len(l1_list)  # average L1 over all steps

print(
    f"pkl_path is {pkl_path}, for {len(plot_actions)} steps, "
    f"dx_mean is {dx_mean}, dy_mean is {dy_mean}, dz_mean is {dz_mean}, "
    f"l1_mean (torch.mean(|fact-predict|) style) is {l1_mean}"
)