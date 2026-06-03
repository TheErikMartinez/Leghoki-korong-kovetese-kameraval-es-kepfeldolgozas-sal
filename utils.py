import os
import sys


def resource_path(relative_path: str) -> str:
    """Returns the absolute path to a resource file.

    Works both during development and when bundled with PyInstaller.
    In a PyInstaller bundle, files are extracted to sys._MEIPASS.
    """
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS          # PyInstaller temp folder
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)
