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
        subprocess.check_call(
            "CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 --num_nodes 1 {test} > test.out 2> test.err",
            shell=True,
        )
    except Exception:
        errors += 1
    subprocess.call("cat test.log ; rm test.out test.err test.log", shell=True)

if errors > 0:
    sys.exit(1)
