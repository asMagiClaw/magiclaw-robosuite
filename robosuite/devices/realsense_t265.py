"""
Driver class for Intel RealSense T265 camera.
"""

import threading
import copy
import numpy as np
import pyrealsense2 as rs
from typing import Tuple, Dict, Optional, List
from robosuite.controllers.composite.composite_controller import WholeBody, WholeBodyIK
from robosuite.devices import Device
from robosuite.utils import transform_utils
from pynput.keyboard import Key, Listener


class RealSenseT265(Device):
    """
    A driver class for the Intel RealSense T265 camera.
    
    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0, active_end_effector: Optional[str] = "right"):
        super().__init__(env)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.active_end_effector = active_end_effector

        self._pose = np.zeros(7, dtype=np.float32)  # translation and quaternion (x, y, z, qx, qy, qz, qw)
        self._reset_state = 0
        self._enabled = False

        # Create a pipeline and configure it to stream pose data
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.pose)
        self.pipeline.start(self.config)
        
        self._display_controls()
        self._reset_internal_state()
        
        self.robot_ee_init_poses = {}
        site_names = self._get_site_names()
        for site_name in site_names:
            pos = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_name)]
            mat = self.env.sim.data.site_xmat[self.env.sim.model.site_name2id(site_name)]
            self.robot_ee_init_poses[site_name] = {
                "pos": copy.deepcopy(pos),
                "mat": copy.deepcopy(mat),
            }
        
        # Wait for the first pose frame to initialize
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
        self.initial_pose = self._get_pose(frames)
        
        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """
        
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        
        print("")
        print_command("Control", "Command")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("Ctrl+C", "quit")
        print_command("", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
    
    def _get_site_names(self) -> List[str]:
        """
        Helper function to get the names of the sites used for robot initial poses.

        TODO: unify this logic to be controller independent.

        Returns:
            List[str]: A list of site names.
        """
        if isinstance(self.env.robots[0].composite_controller, WholeBody):  # input type passed to joint_action_policy
            site_names = self.env.robots[0].composite_controller.joint_action_policy.site_names
        else:
            site_name = f"gripper0_{self.active_arm}_grip_site"
            site_names = [site_name]
        return site_names
    
    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        # Reset pose
        self._pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # translation and quaternion (x, y, z, qx, qy, qz, qw)
        
        init_trans, init_rot = self._get_pose(self.pipeline.wait_for_frames())
        self.initial_pose = np.zeros(7, dtype=np.float32)
        self.initial_pose[:3] = init_trans
        self.initial_pose[3:] = transform_utils.mat2quat(init_rot)  # quaternion (qx, qy, qz, qw)
        print(f"Initial pose: {self.initial_pose}")

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
    
    def _get_pose(self, frames) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the pose data from the given frames.
        
        Args:
            frames: The frames from the RealSense pipeline containing pose data.
            
        Returns:
            pos (np.ndarray): Translation vector as a numpy array.
            rot (np.ndarray): Rotation as a scipy Rotation object.
        """
        
        # Get the pose frame and extract translation and rotation
        pose = frames.get_pose_frame().get_pose_data()
        pos = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
        rot = transform_utils.quat2mat(np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]))
        
        # Convert to z-up coordinate
        pos, rot = self._convert_to_z_up(pos, rot)
        
        return pos, rot

    def _convert_to_z_up(self, pos: np.ndarray, rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the pose from the T265 coordinate system to a z-up coordinate system.
        
        The T265 uses a x-right, y-down, z-forward coordinate system,
        while the z-up coordinate system uses x-right, y-forward, z-up.
        T265:                   z-up:
                z                   z   y
               /                    |  /
              /                     | /
             /                      |/
             ------- x              ------- x
            |
            |
            y
        This function rotates the translation and rotation accordingly.
        Specifically, it rotates the pose by -90 degrees around the x-axis.
        
        Args:
            pos (np.ndarray): Translation vector in T265 coordinate system.
            rot (np.ndarray): Rotation in T265 coordinate system.
        Returns:
            converted_trans (np.ndarray): Translation vector in z-up coordinate system.
            converted_rot (np.ndarray): Rotation in z-up coordinate system.
        """
        
        # Rotation matrix to convert T265 to z-up coordinate system
        rot_mat = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])
        
        # Convert T265 translation to z-up translation
        converted_pos = rot_mat @ pos
        
        # Convert T265 rotation to z-up rotation
        converted_rot = rot_mat @ rot
        
        return converted_pos, converted_rot
    
    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        
        return dict()
    
    def run(self):
        """
        Main loop for the RealSense T265 camera.
        Continuously reads pose data and updates the control state.
        """
        while True:
            if self._enabled:
                # Wait for a new frame
                frames = self.pipeline.wait_for_frames()
                
                # Get the pose data
                pos, rot = self._get_pose(frames)
                
                pos = pos - self.initial_pose[:3]
                
                self._pose[:3] = pos
                self._pose[3:] = transform_utils.mat2quat(rot)
    
    def input2action(self) -> Optional[Dict]:
        """
        Converts the current t265 pose into a control action for the robot.

        Returns:
            dict: A dictionary containing the control values for the robot.
        """
        
        if self._reset_state:
            return None
        
        action: Dict[str, np.ndarray] = {}
        gripper_dof = self.env.robots[0].gripper[self.active_end_effector].dof
        site_names = self._get_site_names()
        for site_name in site_names:
            target_name_prefix = "right" if "right" in site_name else "left"  # hardcoded for now
            robot_init_pose = self.robot_ee_init_poses[site_name]
            target_pos_world = robot_init_pose["pos"] + self._pose[:3] * self.pos_sensitivity
            target_ori_mat_world = transform_utils.quat2mat(self._pose[3:])

            if isinstance(self.env.robots[0].composite_controller, WholeBodyIK):
                assert (
                    self.env.robots[0].composite_controller.composite_controller_specific_config.get(
                        "ik_input_ref_frame", "world"
                    )
                    == "world"
                ), ("Only support world frame for MJGui teleop for now. " "Please modify the controller configs.")
                assert (
                    self.env.robots[0].composite_controller.composite_controller_specific_config.get(
                        "ik_input_type", "absolute"
                    )
                    == "absolute"
                ), ("Only support absolute actions for MJGui teleop for now. " "Please modify the controller configs.")
            # check if need to update frames
            # if isinstance(self.env.robots[0].composite_controller, WholeBody):
                # TODO: should be more general
                if (
                    self.env.robots[0].composite_controller.composite_controller_specific_config.get(
                        "ik_input_ref_frame", "world"
                    )
                    != "world"
                ):
                    target_pose = np.eye(4)
                    target_pose[:3, 3] = target_pos_world
                    target_pose[:3, :3] = target_ori_mat_world
                    target_pose = self.env.robots[0].composite_controller.joint_action_policy.transform_pose(
                        src_frame_pose=target_pose,
                        src_frame="world",  # mocap pose is world coordinates
                        dst_frame=self.env.robots[0].composite_controller.composite_controller_specific_config.get(
                            "ik_input_ref_frame", "world"
                        ),
                    )
                    target_pos, target_ori_mat = target_pose[:3, 3], target_pose[:3, :3]
                else:
                    target_pos, target_ori_mat = target_pos_world, target_ori_mat_world
            else:
                assert (
                    self.env.robots[0].part_controllers[self.active_end_effector].input_ref_frame == "world"
                    and self.env.robots[0].part_controllers[self.active_end_effector].input_type == "absolute"
                ), (
                    "Only support world frame and absolute actions for now. You can modify the controller configs "
                    "being used, e.g. in robosuite/controllers/config/robots/{robot_name}.json, "
                    "robosuite/controllers/config/default/composite/{}.json to enable other options."
                )
                target_pos, target_ori_mat = target_pos_world, target_ori_mat_world
            # convert ori mat to axis angle
            axis_angle_target = transform_utils.quat2axisangle(transform_utils.mat2quat(target_ori_mat))
            action[target_name_prefix + "_abs"] = np.concatenate([target_pos, axis_angle_target])
            grasp = 1 if self.grasp else -1  # hardcode grasp action for now
            action[f"{target_name_prefix}_gripper"] = np.array([grasp] * gripper_dof)
        
        return action
    
    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            print(f"Key released: {key}")
            # controls for grasping
            if key == Key.space:
                self.grasp_states[self.active_robot][self.active_arm_index] = not self.grasp_states[self.active_robot][
                    self.active_arm_index
                ]  # toggle gripper
            
            # controls for mobile base (only applicable if mobile base present)
            elif key.char == "b":
                self.base_modes[self.active_robot] = not self.base_modes[self.active_robot]  # toggle mobile base
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots
            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

        except AttributeError as e:
            pass
        
    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation