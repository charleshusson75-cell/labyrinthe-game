"""Kaggle script kernel for Labyrinthe's RL training (P3, DECISIONS.md D24, D25).

drive.py fills SPEC below and pushes this file as a private script kernel on "GPU T4 x2" with internet
enabled. The kernel:
1. clones the public repository at SPEC["commit"] and installs jax[cuda12] (and optax, gymnasium);
2. unpacks the private dataset's bundle.tar (P2's designed maze pools, checkpoints of earlier
   sessions) into /kaggle/working;
3. runs SPEC["jobs"] in parallel, each on its own GPU (CUDA_VISIBLE_DEVICES), with `{hours}` in
   their commands replaced by the time left before the session's deadline (SPEC["hours"] after the
   start, safely under Kaggle's 12-hour cap). Jobs checkpoint on their own (every ~20 minutes) and
   stop cleanly at their deadline; a job still running 10 minutes later gets SIGTERM, then SIGKILL.
   When an "essential" job ends, the others get SIGTERM a minute later (no GPU left idling). Every
   15 minutes the last lines of each job's log are printed, so `drive.py logs` shows progress while
   the session runs;
4. leaves everything in /kaggle/working (runs, logs/, summary.json): the kernel's output, which
   drive.py downloads.
"""

import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time

START = time.time()
SPEC = json.loads(r"""__SPEC__""")
WORK = "/kaggle/working"
REPO = "/kaggle/repo"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def unpack_inputs() -> None:
    bundles = glob.glob("/kaggle/input/**/bundle.tar", recursive=True)
    for b in bundles:
        print(f"unpacking {b}", flush=True)
        with tarfile.open(b) as t:
            t.extractall(WORK, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    if not bundles:  # Kaggle may have extracted the archive itself
        for m in glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True):
            print(f"copying {os.path.dirname(m)}", flush=True)
            sh(f"cp -rn {os.path.dirname(m)}/. {WORK}/")


def main() -> int:
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    sh(f"git clone -q {SPEC['repo']} {REPO} && cd {REPO} && git checkout -q {SPEC['commit']} && git log -1 --oneline")
    try:
        sh(f"pip install -q 'jax[cuda12]=={SPEC['jax']}' 'optax=={SPEC['optax']}' gymnasium")
    except subprocess.CalledProcessError:  # the Mac's versions aren't on PyPI for this Python: take the latest
        sh("pip install -q -U 'jax[cuda12]' optax gymnasium")
    sh(f"pip install -q --no-deps --ignore-requires-python -e {REPO}/rl")
    sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
    sh("python -c 'import jax; print(jax.__version__, jax.devices())'")
    unpack_inputs()
    deadline = START + SPEC["hours"] * 3600
    procs = []
    for job in SPEC["jobs"]:
        hours = max(0.05, (deadline - time.time()) / 3600 - job.get("margin_hours", 0.1))
        cmds = [[part.replace("{hours}", f"{hours:.3f}") for part in c] for c in job.get("cmds", [job.get("cmd")])]
        cmd = " ; ".join(shlex.join(c) for c in cmds)  # one after the other
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(job["gpu"]), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
               "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
        log = open(f"{WORK}/logs/{job['name']}.log", "w")
        print(f"[{job['name']}] GPU {job['gpu']}: {cmd}", flush=True)
        procs.append((job, subprocess.Popen(cmd, shell=True, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log))
    killed = set()
    last_echo = time.time()
    ended_at = None
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(30)
        if time.time() - last_echo > 900:
            last_echo = time.time()
            for job, _, _ in procs:
                with open(f"{WORK}/logs/{job['name']}.log", errors="replace") as f:
                    lines = [line.rstrip() for line in f.readlines()[-3:]]
                for line in lines:
                    print(f"[{job['name']} {(time.time() - START) / 3600:.2f} h] {line[:400]}", flush=True)
        if ended_at is None and any(job.get("essential") and p.poll() is not None for job, p, _ in procs):
            ended_at = time.time()
            print("an essential job ended: stopping the others in a minute", flush=True)
        if ended_at is not None and time.time() - ended_at > 60:
            for job, p, _ in procs:
                if p.poll() is None and job["name"] not in killed:
                    os.killpg(p.pid, signal.SIGTERM)
                    killed.add(job["name"])
        late = time.time() - deadline
        for job, p, _ in procs:
            if p.poll() is None and late > 600 and job["name"] not in killed:
                print(f"[{job['name']}] past the deadline: SIGTERM", flush=True)
                os.killpg(p.pid, signal.SIGTERM)  # the shell and the Python under it
                killed.add(job["name"])
            elif p.poll() is None and late > 900:
                os.killpg(p.pid, signal.SIGKILL)
    for _, _, log in procs:
        log.close()
    summary = {"spec": SPEC, "elapsed_hours": (time.time() - START) / 3600,
               "jobs": {job["name"]: p.returncode for job, p, _ in procs}}
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for job, p, _ in procs:
        print(f"[{job['name']}] exit {p.returncode}; last lines:", flush=True)
        os.system(f"tail -n 15 {WORK}/logs/{job['name']}.log")
    print(f"done in {summary['elapsed_hours']:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
