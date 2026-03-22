@echo off
cd /d c:\Users\nnniec\Program\smart_cut_auto
python test_run.py > test_output.txt 2>&1
type test_output.txt
