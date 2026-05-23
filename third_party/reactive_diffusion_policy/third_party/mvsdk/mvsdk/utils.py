import os
import platform

_rules = """KERNEL=="*", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ACTION=="add", ATTR{idVendor}=="f622", MODE="666", TAG="mvusb_dev",  A"
KERNEL=="*", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ACTION=="add", ATTR{idVendor}=="080b", MODE="666", TAG="mvusb_dev",  A"
KERNEL=="*", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ACTION=="remove", TAG=="mvusb_dev", R"
"""
_rules_install_path = "/etc/udev/rules.d/88-mvusb.rules"


def install_driver() -> bool:
    is_win = (platform.system() == "Windows")
    if is_win:
        print("You should run the .msi installer")
        return True
    try:
        print(f"Writing to {_rules_install_path}")
        with open(_rules_install_path, 'w') as f:
            f.write(_rules)
    except Exception as e:
        print(e)
        return False


def uninstall_driver() -> bool:
    is_win = (platform.system() == "Windows")
    if is_win:
        print("You should uninstall from ControlPanel")
        return True
    else:
        try:
            os.remove(_rules_install_path)
            return True
        except Exception as e:
            print(e)
            return False
