import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
tactile_path = "outputs/log_dir/tactile_realenv.pkl"
with open(tactile_path,'rb') as f:
    realenv_tactile_dict = pickle.load(f)

dataset_tactile_path = "outputs/log_dir/tactile_dataset.pkl"
with open(dataset_tactile_path,'rb') as f:
    dataset_tactile_dict = pickle.load(f)

# Function to compute mean absolute values
def compute_mean_abs(data_dict):
    keys = sorted(data_dict.keys())
    values = [np.mean(np.abs(data_dict[k])) for k in keys]
    return keys, values

realenv_keys, realenv_values = compute_mean_abs(realenv_tactile_dict)
dataset_keys, dataset_values = compute_mean_abs(dataset_tactile_dict)

# Plot realenv tactile magnitudes
plt.figure()
plt.plot(realenv_keys, realenv_values)
plt.xlabel('Key (Step)')
plt.ylabel('Mean Absolute Value')
plt.title('RealEnv Tactile Magnitude')
plt.grid(True)
plt.savefig("outputs/realenv_tactile.png")

# Plot dataset tactile magnitudes
plt.figure()
plt.plot(dataset_keys, dataset_values)
plt.xlabel('Key (Step)')
plt.ylabel('Mean Absolute Value')
plt.title('Dataset Tactile Magnitude')
plt.grid(True)
plt.savefig("outputs/dataset_tactile.png")