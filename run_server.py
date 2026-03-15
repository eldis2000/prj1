import os
import subprocess
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

with open("server_debug.log", "w") as f:
    process = subprocess.Popen(
        ["C:\\Users\\804\\miniforge3\\envs\\prj1\\python.exe", "main.py"],
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd()
    )
    print(f"Server started with PID {process.pid}")
    time.sleep(10) # Wait a bit to catch initial startup errors
    if process.poll() is not None:
        print(f"Process exited with code {process.returncode}")
    else:
        print("Process is still running.")

with open("server_debug.log", "r") as f:
    print("--- Server Output ---")
    print(f.read())
