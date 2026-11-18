"""
DR Vision AI - Quick Launcher
Starts the Flask server if not already running and automatically opens
the web application in the default browser.
"""

import os
import sys
import time
import socket
import webbrowser
import subprocess

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    url = "http://127.0.0.1:5000"

    if not is_port_in_use(5000):
        print("[Launcher] Starting DR Vision AI Server...")
        subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=base_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        
        # Wait for server to become responsive
        for _ in range(40):
            if is_port_in_use(5000):
                break
            time.sleep(0.1)

    print(f"[Launcher] Opening {url} in browser...")
    webbrowser.open(url)

if __name__ == '__main__':
    main()
