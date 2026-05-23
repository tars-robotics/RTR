import os
current_path = os.path.dirname(os.path.abspath(__file__))
flexivrdk_root_path = os.path.join(current_path, '../', '../', '../', 'third_party', "flexiv_rdk-main")

print(flexivrdk_root_path)
import sys
sys.path.insert(0, flexivrdk_root_path+"/lib_py")
import flexivrdk
sys.path.insert(0, flexivrdk_root_path+"/example_py")

import time
from typing import List
from loguru import logger


class FlexivController():
    def __init__(self,
                 local_ip="192.168.2.187",
                 robot_ip="192.168.2.100",
                 ) -> None:
        self.DOF=7

        try:
            self.robot_states = flexivrdk.RobotStates()
            self.gripper_states = flexivrdk.GripperStates()
            self.log = flexivrdk.Log()
            self.mode = flexivrdk.Mode
            self.robot = flexivrdk.Robot(robot_ip, local_ip)
            self.gripper = flexivrdk.Gripper(self.robot)

            self.clear_fault()
            self.log.info("Enabling left robot ...")
            self.robot.enable()
            seconds_waited = 0
            while not self.robot.isOperational():
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited == 10:
                    self.log.warn(
                        "Still waiting for robot to become operational, please check that the robot 1) "
                        "has no fault, 2) is in [Auto (remote)] mode")
            self.log.info("Left robot is now operational")
            self.robot.setMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)
        except Exception as e:
            self.log.error("Error occurred while connecting to robot server: %s" % str(e))
            return None

    def clear_fault(self):
        # Fault Clearing
        # ==========================================================================================
        # Check if the robot has fault
        if self.robot.isFault():
            logger.warning("Fault occurred on robot server, trying to clear ...")
            self.log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            self.robot.clearFault()
            time.sleep(2)
            # Check again
            if self.robot.isFault():
                self.log.error("Fault cannot be cleared, exiting ...")
                return
            self.log.info("Fault on robot server is cleared")

    def get_current_robot_states(self) -> flexivrdk.RobotStates:
        # return current robot states via flexivAPI
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states

    def get_current_gripper_states(self) -> flexivrdk.GripperStates:
        # return current gripper states via flexivAPI
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states

    def get_current_gripper_force(self) -> float:
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.force

    def get_current_gripper_width(self) -> float:
        self.gripper.getGripperStates(self.gripper_states)
        return self.gripper_states.width

    def get_current_q(self) -> List[float]:
        # return current joint angles via flexivAPI
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states.q

    def get_current_tcp(self) -> List[float]:
        # return current TCP pose via flexivAPI
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states.tcpPose

    def move(self, target_q):
        v = [1.5]*self.DOF # velocity limit
        a = [0.8]*self.DOF # acceleration limit
        self.robot.sendJointPosition(
                target_q,
                [0.0]*self.DOF,
                [0.0]*self.DOF,
                v,
                a)
            
    def tcp_move(self, target_tcp):
        self.robot.sendCartesianMotionForce(
                target_tcp, 
                [0.0]*6, 
                0.5,
                1.0)

    def execute_primitive(self, primitive_command: str):
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
        self.robot.executePrimitive(primitive_command)
        while self.parse_pt_states(self.robot.getPrimitiveStates(), "reachedTarget") != "1":
            time.sleep(0.1)
        self.robot.setMode(self.mode.NRT_CARTESIAN_MOTION_FORCE)

    @staticmethod
    def parse_pt_states(pt_states, parse_target):
        """
        Parse the value of a specified primitive state from the pt_states string list.
        Parameters
        ----------
        pt_states : list of str
            Primitive states string list returned from Robot::getPrimitiveStates().
        parse_target : str
            Name of the primitive state to parse for.
        Returns
        ----------
        str
            Value of the specified primitive state in string format. Empty string is
            returned if parse_target does not exist.
        """
        for state in pt_states:
            words = state.split()
            if words[0] == parse_target:
                return words[-1]
        return ""
    