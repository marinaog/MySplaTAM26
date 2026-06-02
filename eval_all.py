import os
import signal
import subprocess

num_runs = 1
args = []
for run_num in range(num_runs):
    for scene in [
        #"bottles_5",
        #"bottles_6",
        #"bottles",
        "bottles_raw_logexp_3",
        #"boxes_5",
        #"boxes",
        #"boxes_raw_logexp_2",
        ##"cabin_2",
        #"cabin",
        #"cabin_raw_logexp_2",
        ##"candles_raw_logexp_2",
        ##"christmas_4",
        #"christmas",
        #"christmas_raw_logexp_3",
        #"coat_rack_3",
        #"coat_rack_2",
        #"coat_rack_raw_logexp",
        #"kitchen_3",
        "kitchen",
        #"kitchen_raw_logexp_5",
        #"nerdy_robot_3",
        #"nerdy_robot_logexp_3",
        #"small_city_logexp_2",
        #"coffee_raw_logexp"
        ]:
        script_args = [f"experiments/rawslam/{scene}"]
        log_file = f"prints/eval_{scene}_{run_num}"
        args.append((script_args, log_file))

script = "scripts/eval_only.py"
os.makedirs("prints", exist_ok=True)


def reap_finished_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break


def terminate_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


for script_args, log_file in args:
    command = ["python", script] + script_args
    print(f"Running {' '.join(command)}...")
    p1 = None
    p2 = None
    try:
        p1 = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        p2 = subprocess.Popen(["tee", log_file], stdin=p1.stdout)
        p1.stdout.close()
        p2.wait()
        ret = p1.wait()
    finally:
        if p1 is not None:
            terminate_process_group(p1)
        if p2 is not None and p2.poll() is None:
            p2.terminate()
            p2.wait()
        reap_finished_children()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, command)
    print(f"Finished {script}.")
    print("")
