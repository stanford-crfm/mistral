import os
import subprocess
import sys


tests = [x for x in os.listdir(".") if x.startswith("test") and x.endswith("py")]

errors = 0
for test in tests:
    # clean up if necessary
    for log_path in ["test.out", "test.err", "test.log"]:
        if os.path.exists(log_path):
            os.remove(log_path)
    # run tests
    try:
        print("Running test:", test)
        subprocess.check_call(
            f"CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 --num_nodes 1 {test}",
            shell=True,
        )
    except Exception:
        errors += 1
    if os.path.exists("test.log"):
        subprocess.call("cat test.log", shell=True)
        print("")

for log_path in ["test.out", "test.err", "test.log"]:
    if os.path.exists(log_path):
        os.remove(log_path)

if errors > 0:
    sys.exit(1)
