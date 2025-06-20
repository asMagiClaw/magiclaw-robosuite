from .device import Device
from .keyboard import Keyboard
from .ios_device import IOSDevice
try:
    from .spacemouse import SpaceMouse
    from .dualsense import DualSense
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse or DualSense.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
