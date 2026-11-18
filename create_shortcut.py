"""
Utility script to create a Windows Desktop Shortcut for DR Vision AI.
"""

import os
from pathlib import Path
import win32com.client

def create_shortcut():
    base_dir = Path(__file__).resolve().parent
    desktop = Path(os.environ.get('USERPROFILE')) / 'OneDrive' / 'Desktop'
    if not desktop.exists():
        desktop = Path(os.environ.get('USERPROFILE')) / 'Desktop'

    shortcut_path = desktop / 'DR Vision AI.lnk'
    bat_path = base_dir / 'launch.bat'
    ico_path = base_dir / 'static' / 'images' / 'app_icon.ico'

    shell = win32com.client.Dispatch('WScript.Shell')
    shortcut = shell.CreateShortcut(str(shortcut_path))
    shortcut.TargetPath = str(bat_path)
    shortcut.WorkingDirectory = str(base_dir)
    shortcut.Description = 'DR Vision AI - Rural Retinal Screening System'
    if ico_path.exists():
        shortcut.IconLocation = str(ico_path)
    shortcut.Save()
    print(f"[Shortcut] Successfully created at: {shortcut_path}")

if __name__ == '__main__':
    create_shortcut()
