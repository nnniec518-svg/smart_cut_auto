import subprocess
import sys
import os

os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
result = subprocess.run([sys.executable, "test_run.py"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
