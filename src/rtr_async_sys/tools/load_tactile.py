import pickle
import numpy as np


pca_mean1 = np.load("data/ckpts/vase_sponge_test1/rdp_pca/pca_mean1.npy")
print(pca_mean1)

pca_matrix1 = np.load("data/ckpts/vase_sponge_test1/rdp_pca/pca_matrix1.npy")
print(pca_matrix1)




tactile_path = "outputs/log_dir/tactile_realenv.pkl"
with open(tactile_path,'rb') as f:
    realenv_tactile_dict = pickle.load(f)

print(realenv_tactile_dict.keys())

print(realenv_tactile_dict[0])
print(realenv_tactile_dict[10])
print(realenv_tactile_dict[20])
print(realenv_tactile_dict[30])
print(realenv_tactile_dict[40])
print(realenv_tactile_dict[50])

# for i in range(list(realenv_tactile_dict.keys())[-1]):
#     if i in realenv_tactile_dict.keys():
#         print(realenv_tactile_dict[i])


dataset_tactile_path = "outputs/log_dir/tactile_dataset.pkl"
with open(dataset_tactile_path,'rb') as f:
    dataset_tactile_dict = pickle.load(f)

print(len(list(dataset_tactile_dict.keys())))

print(dataset_tactile_dict[0])
print(dataset_tactile_dict[10])
print(dataset_tactile_dict[20])
print(dataset_tactile_dict[30])
print(dataset_tactile_dict[40])
print(dataset_tactile_dict[50])

# for i in range(list(dataset_tactile_dict.keys())[-1]):
#     if i in dataset_tactile_dict.keys():
#         print(dataset_tactile_dict[i])