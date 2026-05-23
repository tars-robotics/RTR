"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

Rotate6d_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 12,  # NOTE: action chunk is hard-coded here.
    "ACTION_DIM": 10,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

Rotate6d_vae_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 48,  # NOTE: action chunk is hard-coded here.
    "ACTION_DIM": 10,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

Rotate6d_rdp_vae_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 12, #12 for latent-oft, multi by 4 get vae horizon(48)
    "ACTION_DIM": 10,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.NORMAL,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "rdp_vae" in cmd_args:
        return "RdpVae"
    if "6drotate" in cmd_args and "train_vae_two_stage" in cmd_args:# use vla to get the latent, so chunk_size match vla
        return "Rotate6d"
    if "6drotate" in cmd_args and "train_vae" in cmd_args:
        return "Rotate6d_vae"
    if "6drotate" in cmd_args:
        return "Rotate6d"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"

def detect_downsample_ratio():
    cmd_args = " ".join(sys.argv).lower()

    if "down16" in cmd_args or "downsample16" in cmd_args:
        return 16
    elif "down2" in cmd_args or "downsample2" in cmd_args:
        return 2
    elif "down4" in cmd_args or "downsample4" in cmd_args:
        return 4
    elif "down8" in cmd_args or "downsample8" in cmd_args:
        return 8
    elif "down1" in cmd_args or "downsample1" in cmd_args:
        return 1
    else:
        return -1
    

# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()
downsample_ratio = detect_downsample_ratio()
print(f"downsample_ratio is {downsample_ratio}")
# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "Rotate6d":
    constants = Rotate6d_CONSTANTS
elif ROBOT_PLATFORM == "Rotate6d_vae":
    constants = Rotate6d_vae_CONSTANTS
elif ROBOT_PLATFORM == "RdpVae":
    print("use rdp vae constants")
    constants = Rotate6d_rdp_vae_CONSTANTS

if downsample_ratio != -1:
    constants["NUM_ACTIONS_CHUNK"] = 48 // downsample_ratio

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  downsample_ratio = {downsample_ratio}")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
