"""
Driver for MagiClaw.
"""

import threading
import copy
import zmq
import numpy as np
from typing import Tuple, Dict, Optional, List
from robosuite.controllers.composite.composite_controller import WholeBody, WholeBodyIK
from robosuite.devices import Device
from robosuite.devices.protobuf import magiclaw_msg_pb2
from robosuite.utils import transform_utils
from pynput.keyboard import Listener


class MagiClawSubscriber:
    def __init__(self, host: str, port: int, hwm: int = 1, conflate: bool = True, timeout: int = 100) -> None:
        """Subscriber initialization.

        Args:
            host (str): The host address of the subscriber.
            port (int): The port number of the subscriber.
            hwm (int): High water mark for the subscriber. Default is 1.
            conflate (bool): Whether to conflate messages. Default is True.
        """

        print("{:-^80}".format(" MagiClaw Subscriber Initialization "))
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
        # Subscribe all messages
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set poller
        self.poller = zmq.Poller()
        self.poller.register(self.subscriber, zmq.POLLIN)
        self.timeout = timeout

        print("MagiClaw Subscriber Initialization Done.")
        print("{:-^80}".format(""))

    def subscribeMessage(self):
        """Subscribe the message.

        Returns:
            The message.
        """

        # Receive the message
        msg = self.subscriber.recv()
        # Parse the message
        magiclaw = magiclaw_msg_pb2.MagiClaw()
        magiclaw.ParseFromString(msg)

        # Unpack the message
        claw_angle = magiclaw.claw.angle
        motor_angle = magiclaw.claw.motor.angle
        motor_speed = magiclaw.claw.motor.speed
        magiclaw_pose = np.array(magiclaw.pose)

        return (
            claw_angle,
            motor_angle,
            motor_speed,
            magiclaw_pose,
        )

    def close(self):
        """Close ZMQ socket and context to prevent memory leaks."""
        if hasattr(self, "subscriber") and self.subscriber:
            self.subscriber.close()
        if hasattr(self, "context") and self.context:
            self.context.term()

class MagiClaw(Device):
    """
    Device for MagiClaw.
    """
    
    def __init__(
        self,
        env,
        host,
        port=6300,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        active_end_effector: Optional[str] = "right",
    ) -> None:
        """
        Initialize the MagiClaw device.

        Args:
            env: The environment in which the device operates.
            host (str): The host address of the MagiClaw server.
            port (int): The port number of the MagiClaw server.
            pos_sensitivity (float): Sensitivity for position control.
            rot_sensitivity (float): Sensitivity for rotation control.
            active_end_effector (Optional[str]): The active end effector to control, default is "right".
        """
        
        super().__init__(env)
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.active_end_effector = active_end_effector

        self._pose = np.zeros(7, dtype=np.float32)
        self._initial_pose = np.zeros(7, dtype=np.float32)
        self._claw_angle = 0.0
        self._reset_state = 0
        self._enabled = False

        self.host = host
        self.port = port
        self.subscriber = MagiClawSubscriber(host, port)
        
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
        
        # launch a thread to listen to MagiClaw messages
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        
        # also add a listener for keyboard events
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
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
        print_command("Move MagiClaw laterally", "move arm horizontally in x-y plane")
        print_command("Move MagiClaw vertically", "move arm vertically")
        print_command("Twist MagiClaw about an axis", "rotate arm about a corresponding axis")
        print_command("Move MagiClaw trigger", "open/close gripper")
        print_command("Ctrl+C", "quit")
        print_command("Ctrl+q", "reset simulation")
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

        # reset initial pose
        self._initial_pose = copy.deepcopy(self._pose)
        print(f"Initial pose: {self._initial_pose}")
        
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        
    def _get_state(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Get the current state of the MagiClaw.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: The claw angle, position and rotation matrix of the MagiClaw.
        """
        
        # Get the pose and extract translation and rotation
        claw_angle, _, _, magiclaw_pose = self.subscriber.subscribeMessage()
        pos = magiclaw_pose[:3]
        rot = magiclaw_pose[3:]
        claw_angle = claw_angle / 180.0 * np.pi  # convert to radians
        
        return claw_angle, pos, rot
    
    def get_controller_state(self):
        """
        Grabs the current state of the controller.

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
                # get the current state
                claw_angle, pos, rot = self._get_state()
                
                self._pose[:3] = pos
                self._pose[3:] = rot
                self._claw_angle = claw_angle
    
    def input2action(self) -> Optional[Dict]:
        """
        Converts the current pose into a control action for the robot.

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
            target_pos_world = robot_init_pose["pos"] + (self._pose[:3] - self._initial_pose[:3]) * self.pos_sensitivity
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
            grasp = np.clip(self._claw_angle / 1.1 * 2.0 - 1.0, -1.0, 1.0)
            action[f"{target_name_prefix}_gripper"] = np.array([grasp] * gripper_dof)

        return action
    
    
    def on_press(self, key):
        """
        Handle key press events.

        Args:
            key: The key that was pressed.
        """
        
        pass
    
    def on_release(self, key):
        """
        Handle key release events.

        Args:
            key: The key that was released.
        """
        
        try:
            print(f"Key released: {key}")
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
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
