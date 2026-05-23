#!/bin/bash

# ================== Configuration ==================

# Resolve the script directory rather than relying on the caller's pwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_MAIN_DIR="${SCRIPT_DIR}/inference_node"
XARM_CMD="python ./gello_trajectory_pub_node.py"
CAMERA_CMD="python ./camera_pub_node.py"
TACTILE_CMD="python ./xense_pub_node.py"
GRIPPER_ACTIVATE_CMD="python ./gripper_activate.py"
GRIPPER_OPEN_CMD="python ./gripper_open.py"
GRIPPER_CLOSE_CMD="python ./gripper_close.py"
ROS_CMD="roscore"

# Terminal-window title
MAIN_TERMINAL_TITLE="Deployment Console"

# ================== Colors ==================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================== Functions ==================

# Close all related terminals.
close_all_terminals() {
    cd $PROJ_MAIN_DIR && python gripper_open.py && sleep 0.5 && python init_pos.py
    sleep 0.5
    echo -e "${YELLOW}Closing all terminal windows...${NC}"
    pkill -f "gnome-terminal.*$MAIN_TERMINAL_TITLE" 2>/dev/null
    pkill -f "$CAMERA_CMD" 2>/dev/null
    pkill -f "$XARM_CMD" 2>/dev/null
    pkill -f "$TACTILE_CMD" 2>/dev/null
    pkill -f "$ROS_CMD" 2>/dev/null
    sleep 0.2
}


start_roscore() {
    echo -e "${YELLOW}[step] Starting RosCore...${NC}"
    pkill -f "gnome-terminal.*$MAIN_TERMINAL_TITLE" 2>/dev/null
    sleep 0.5
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "$ROS_CMD; exec bash"
    sleep 0.5
    return 0
}

start_inference() {
    XARM_CMD="python ./gello_trajectory_pub_node.py"
    CAMERA_CMD="python ./camera_pub_node.py"
    TACTILE_CMD="python ./xense_pub_node.py"

    ### STEP1: move the arm to initial position
    cd $PROJ_MAIN_DIR && python init_pos.py && python gripper_activate.py

    ### STEP2: Start all nodes
    # Start tactile publisher (new tab in the same terminal)
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}Error: directory not found: $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && \
        echo -e '${YELLOW}Publishing tactile data...${NC}' && \
        setsid $TACTILE_CMD & wait \$!; exec bash" &

    # Start arm trajectory publisher (new tab in the same terminal)
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}Error: directory not found: $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}Publishing trajectory data...${NC}' && setsid $XARM_CMD & wait \$!; exec bash" &


    # Start camera publisher (new tab in the same terminal)
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}Error: directory not found: $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}Publishing camera data...${NC}' && $CAMERA_CMD; exec bash" &

    ### STEP3: Place the object and close the gripper (interaction)
    read -p "Place the object at the end-effector, then press any key to continue..." -n1 -s
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}Closing gripper...${NC}' && $GRIPPER_CLOSE_CMD; exec bash" &

}

# ================== Main menu ==================
show_menu() {
    clear
    echo -e "${GREEN}=== Gello-xArm-Dexhand step-by-step control system ===${NC}"
    echo -e "1. Inference mode"
    echo -e "0. Exit and close all terminals"
    echo -n "Choice: "
}

# ================== Step dispatch ==================
execute_step() {
    case $1 in
        1)
            start_roscore
            start_inference ;;
        0)
            close_all_terminals
            echo -e "${GREEN}System exited safely${NC}"
            exit 0
            ;;
        *) echo -e "${RED}Invalid input, please try again${NC}"; sleep 0.5 ;;
    esac
}

# ================== Main loop ==================
while true; do
    show_menu
    read -r choice
    case $choice in
        [1]) execute_step "$choice" ;;
        0) execute_step 0 ;;
        *) echo -e "${RED}Invalid input, please try again${NC}"; sleep 0.5 ;;
    esac
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read -r
done
