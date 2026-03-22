import subprocess
import sys
import os

os.chdir(r"c:\Users\nnniec\Program\smart_cut_auto")
result = subprocess.run(
    [sys.executable, "test_simple.py"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# Write to file
with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\n\nReturn code: {result.returncode}")

print("Done. Check test_result.txt")
