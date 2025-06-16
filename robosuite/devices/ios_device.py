"""
Driver for iOS Devices with MagiClaw app.
"""

import threading
import copy
import zmq
from . import phone_msg_pb2  # Import the generated protobuf message class
import numpy as np
from typing import Tuple, Dict, Optional, List
from robosuite.controllers.composite.composite_controller import WholeBody, WholeBodyIK
from robosuite.devices import Device
from robosuite.utils import transform_utils
from pynput.keyboard import Key, Listener

class PhoneSubscriber:
    def __init__(self, host, port, hwm: int = 1, conflate: bool = True, timeout: int = 100) -> None:
        """Subscriber initialization.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
            hwm (int): High water mark for the subscriber. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
        """

        print("{:-^80}".format(" Phone Subscriber Initialization "))
        print(f"Address: tcp://{host}:{port}")

        # Create a ZMQ context
        self.context = zmq.Context()
        # Create a ZMQ subscriber
        self.subscriber = self.context.socket(zmq.SUB)
        # Set high water mark
        self.subscriber.set_hwm(hwm)
        # Set conflate
        self.subscriber.setsockopt(zmq.CONFLATE, conflate)
        # Connect the address
        self.subscriber.connect(f"tcp://{host}:{port}")
        # Subscribe the topic
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout

        # Init the message
        self.phone = phone_msg_pb2.Phone()

        print("Package Phone")
        print("Message Phone")
        print(
            "{\n\tbytes img = 1;\n\trepeated float pose = 2;\n\trepeated float force = 3;\n\trepeated float node = 4;\n}"
        )

        print("Phone Subscriber Initialization Done.")
        print("{:-^80}".format(""))

    def subscribeMessage(self) -> Tuple[bytes, list, int, int, list, list]:
        """Subscribe the message.

        Args:
            timeout: Maximum time to wait for a message in milliseconds. Default is 100ms.

        Returns:
            color_img: The image captured by the camera.
            depth_img: The depth image captured by the camera.
            depth_width: The width of the depth image.
            depth_height: The height of the depth image.
            local_pose: The local pose of the phone.
            global_pose: The global pose of the phone.

        Raises:
            zmq.ZMQError: If no message is received within the timeout period.
        """

        # Receive the message
        
        if self.poller.poll(self.timeout):
            msg = self.subscriber.recv()
            # Parse the message
            self.phone.ParseFromString(msg)
        else:
            raise Exception("No message received within the timeout period.")
        return (
            self.phone.color_img,
            self.phone.depth_img,
            self.phone.depth_width,
            self.phone.depth_height,
            self.phone.local_pose,
            self.phone.global_pose,
        )

    def close(self):
        """Close ZMQ socket and context to prevent memory leaks."""
        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()
            
class IOSDevice(Device):
    """
    Device for iOS devices with MagiClaw app.
    """

    def __init__(self, env, host, port=8000, pos_sensitivity=1.0, rot_sensitivity=1.0, active_end_effector: Optional[str] = "right") -> None:
        """Initialize the iOS device.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
        """
        super().__init__(env)
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.active_end_effector = active_end_effector

        self._pose = np.zeros(7, dtype=np.float32)
        self._reset_state = 0
        self._enabled = False
        
        self.host = host
        self.port = port
        self.subscriber = PhoneSubscriber(host, port)
        
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
            
        # Get the initial pose of the phone
        self.init_pose = self._get_pose()
        
        
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
        
        init_pos, init_rot = self._get_pose()
        self.initial_pose = np.zeros(7, dtype=np.float32)
        self.initial_pose[:3] = init_pos
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
    
    def _get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the pose data from the given frames.
        
        Args:
            frames: The frames from the RealSense pipeline containing pose data.
            
        Returns:
            pos: Translation vector as a numpy array.
            rot: Rotation as a scipy Rotation object.
        """
        
        # Get the pose frame and extract translation and rotation
        _, _, _, _, _, global_pose = self.subscriber.subscribeMessage()
        pos = np.array(global_pose[:3], dtype=np.float32)
        rot = transform_utils.quat2mat(np.array(global_pose[3:], dtype=np.float32))
        
        # Convert to z-up coordinate
        pos, rot = self._convert_to_z_up(pos, rot)
        
        return pos, rot
    
    def _convert_to_z_up(self, pos: np.ndarray, rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the pose from the ARKit coordinate system to a z-up coordinate system.
        
        The T265 uses a x-right, y-down, z-forward coordinate system,
        while the z-up coordinate system uses x-right, y-forward, z-up.
        ARKit:                   z-up:
            y                       z
            |                       |
            |                       |
            |                       |
             ------- x              ------- y
           /                       /
          /                       /
         /                       /
        z                       x
        
        This function rotates the translation and rotation accordingly.
        Specifically, it rotates the pose by -90 degrees around the x-axis.
        
        Args:
            pos (np.ndarray): Translation vector in T265 coordinate system.
            rot (R): Rotation in T265 coordinate system.
        Returns:
            ur_pos (np.ndarray): Translation vector in z-up coordinate system.
            ur_rot (R): Rotation in z-up coordinate system.
        """
        
        # Rotation matrix to convert T265 to z-up coordinate system
        rot_mat = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
        
        # Convert T265 translation to z-up translation
        convert_pos = rot_mat @ pos
        
        # Convert T265 rotation to z-up rotation
        convert_rot = rot_mat @ rot
        
        return convert_pos, convert_rot
    
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
                # Get the pose data
                pos, rot = self._get_pose()
                
                pos = pos - self.initial_pose[:3]
                
                self._pose[:3] = pos
                self._pose[3:] = rot.as_quat()
                
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