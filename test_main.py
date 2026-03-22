import subprocess
import sys
import os

os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
# Run main.py with a simple test
result = subprocess.run(
    [sys.executable, "main.py", "-s", "test_full_flow_script.py"],
    capture_output=True,
    text=True,
    timeout=120
)
print("STDOUT:", result.stdout[:5000] if len(result.stdout) > 5000 else result.stdout)
print("STDERR:", result.stderr[:2000] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)
