"""
MagiClaw on robot
"""

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MagiClaw(GripperModel):
    """
    MagiClaw gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/magiclaw_on_robot.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps action to the gripper's joint positions.

        Args:
            action (np.array): Gripper-specific action, expected to be of size 1.

        Raises:
            AssertionError: If the action does not have the expected dimension size.
        """

        # Ensure the action is of the correct dimension
        assert len(action) == self.dof
        # self.current_action = np.clip(
        #     self.current_action + self.speed * np.sign(action), -1.0, 1.0
        # )
        self.current_action = np.clip(
            np.array(action), -1.0, 1.0
        )  # Directly use the action as it is already in the correct range
        return self.current_action

    @property
    def init_qpos(self):
        """
        Returns the initial joint positions of the gripper.
        """

        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def _important_geoms(self):
        """
        Returns a dictionary mapping finger names to their important geometries.
        Each finger has a list of geometries that are considered important for collision detection.
        """
        return {
            "left_finger": [
                "left_drive_beam_col",
                "left_follow_beam_col",
                "left_finger_base_col",
                "left_finger_col",
            ],
            "right_finger": [
                "right_drive_beam_col",
                "right_follow_beam_col",
                "right_finger_base_col",
                "right_finger_col",
            ],
            "left_fingerpad": [
                "left_finger_base_col",
                "left_finger_col",
            ],
            "right_fingerpad": [
                "right_finger_base_col",
                "right_finger_col",
            ],
        }

    @property
    def speed(self):
        """
        Returns the speed at which the gripper operates.
        """

        return 0.5

    @property
    def dof(self):
        """
        Returns the number of degrees of freedom (DOF) of the gripper.
        """

        return 1
