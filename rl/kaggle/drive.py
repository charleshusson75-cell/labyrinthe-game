"""Drive Kaggle training sessions from the Mac (P3, DECISIONS.md D24, D25). Uses the kaggle CLI of
rl/.venv and the credentials in ~/.kaggle/kaggle.json.

    rl/.venv/bin/python rl/kaggle/drive.py upload [--runs NAME,...]      bundle → private dataset
    rl/.venv/bin/python rl/kaggle/drive.py push JOB [--hours H] [--commit SHA] [--dry-run]
    rl/.venv/bin/python rl/kaggle/drive.py status
    rl/.venv/bin/python rl/kaggle/drive.py logs [--seconds 60]
    rl/.venv/bin/python rl/kaggle/drive.py fetch [--to DIR]
    rl/.venv/bin/python rl/kaggle/drive.py quota

- upload: packs rl/runs/pools (P2's designed maze pools) and the named run directories of rl/runs/
  into bundle.tar, and creates the private dataset USER/labyrinthe-p3 or adds a version to it. Every
  session unpacks it into /kaggle/working, so a run resumes from the checkpoint uploaded here.
- push: fills kernel.py with a job (JOBS below), the commit (default: origin/main's head; the commit
  must be pushed, since the kernel clones the public repository) and the session's time budget, and
  pushes it as the private script kernel USER/labyrinthe-p3-train on "GPU T4 x2" (NvidiaTeslaT4) with
  internet on. Pushing starts the session.
- status: the session's state. Check it sparingly: a league session lasts ~11 hours.
- logs: the session's live log for `--seconds` (the kernel prints its jobs' last lines every 15
  minutes), then stops listening; after the session, its whole log.
- fetch: downloads the session's output (runs, logs, summary.json) into a new directory,
  rl/runs/kaggle/<time>/ by default. Downloads are data: nothing in them is executed.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
REPO_DIR = RL.parent
RUNS = RL / "runs"
KAGGLE = str(RL / ".venv" / "bin" / "kaggle")
SLUG = "labyrinthe-p3"  # the private dataset
KERNEL = "labyrinthe-p3-train"  # the script kernel (a kernel can't share a dataset's slug: HTTP 409)
REPO = "https://github.com/charleshusson75-cell/labyrinthe-game.git"
JAX, OPTAX = "0.12.0.1", "0.2.8"  # as in rl/.venv on the Mac
PY = "python"


def user() -> str:
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def run(cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


# Jobs: one per GPU, a command ("cmd") or commands run one after the other ("cmds"). Paths are relative
# to /kaggle/working; {hours} is the time left.
TRAIN_GPU = ["--envs", "1024", "--minibatch", "2048", "--adv-refresh", "256", "--save-minutes", "20", "--expert-ref", "0", "--pool-dir", "runs/pools"]


def jobs(name: str, args) -> list:
    if name == "calibrate":  # fp32 on one GPU, fp16 on the other, at the same time (as training will run)
        base = [PY, "-m", "labyrinthe_rl.calibrate", "--iterations", "3", "--out", "calibration.jsonl"]

        def seq(prec):
            return [base + ["--nets", "32x4,64x6,96x8", "--envs", "1024", "--minibatch", "2048", "--precision", prec, "--designer", "--d-mazes", "512"],
                    base + ["--nets", "64x6,96x8", "--envs", "2048", "--minibatch", "4096", "--precision", prec]]

        return [{"name": "calib-fp32", "gpu": 0, "cmds": seq("fp32")}, {"name": "calib-fp16", "gpu": 1, "cmds": seq("fp16")}]
    if name == "calibrate-alone":
        return [{"name": "calib-alone", "gpu": 0, "cmd": [PY, "-m", "labyrinthe_rl.calibrate", "--nets", args.net, "--envs", "1024,2048",
                                                          "--minibatch", "2048,4096", "--precision", args.precision, "--iterations", "3", "--out", "calibration.jsonl"]}]
    c, b = args.net.split("x")
    if name == "stage0":
        return [
            {"name": "stage0", "gpu": 0, "cmd": [PY, "-m", "labyrinthe_rl.train_explorer", "--run", "runs/stage0", "--size", "5", "--channels", c, "--blocks", b,
                                                "--iterations", "100000", "--hours", "{hours}", "--eval-every", "25", *TRAIN_GPU, *(["--resume"] if args.resume else [])]},
            {"name": "designer-vs-p2", "gpu": 1, "cmd": [PY, "-m", "labyrinthe_rl.league", "--run", "runs/designer-vs-p2", "--size", "5", "--designer-vs", "p2",
                                                        "--d-channels", c, "--d-blocks", b, "--d-mazes", "1024", "--hours", "{hours}", "--iterations", "100000", "--save-minutes", "20"]},
        ]
    if name == "league":
        main = [PY, "-m", "labyrinthe_rl.league", "--run", f"runs/{args.run}", "--hours", "{hours}"]
        if args.resume:
            main += ["--resume"]
        else:
            main += ["--size", "5", "--channels", c, "--blocks", b, "--d-channels", c, "--d-blocks", b, "--iterations", "1000000",
                     "--d-mazes", "1024", "--exploiters", "worker", "--explorer-init", args.explorer_init, *TRAIN_GPU, *args.extra]
            if args.designer_init:
                main += ["--designer-init", args.designer_init]
        return [
            {"name": "league", "gpu": 0, "cmd": main, "margin_hours": 0.1, "essential": True},
            {"name": "exploiters", "gpu": 1, "cmd": [PY, "-m", "labyrinthe_rl.league", "--run", f"runs/{args.run}", "--worker", "--hours", "{hours}"], "margin_hours": 0.15},
        ]
    raise SystemExit(f"unknown job {name!r}")


def git_head() -> str:
    run(["git", "-C", str(REPO_DIR), "fetch", "-q", "origin"])
    return run(["git", "-C", str(REPO_DIR), "rev-parse", "origin/main"], capture=True).stdout.strip()


def push(args) -> None:
    commit = args.commit or git_head()
    spec = {"repo": REPO, "commit": commit, "jax": JAX, "optax": OPTAX, "hours": args.hours, "job": args.job, "jobs": jobs(args.job, args)}
    folder = RUNS / "kaggle" / "push"
    folder.mkdir(parents=True, exist_ok=True)
    code = (Path(__file__).parent / "kernel.py").read_text().replace("__SPEC__", json.dumps(spec, indent=1))
    (folder / "kernel.py").write_text(code)
    meta = {
        "id": f"{user()}/{KERNEL}", "title": KERNEL, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{user()}/{SLUG}"],
        "competition_sources": [], "kernel_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(spec, indent=1))
    if args.dry_run:
        print(f"dry run: kernel in {folder}")
        return
    timeout = str(int(args.hours * 3600 + 1800))  # a hard cap above the kernel's own deadline
    run([KAGGLE, "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4", "-t", timeout])


def upload(args) -> None:
    stage = RUNS / "kaggle" / "dataset"
    stage.mkdir(parents=True, exist_ok=True)
    names = [n for n in (args.runs or "").split(",") if n]
    manifest = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": names}
    with tarfile.open(stage / "bundle.tar", "w") as t:
        t.add(RUNS / "pools", arcname="runs/pools")
        for n in names:
            t.add(RUNS / n, arcname=f"runs/{n}", filter=lambda ti: None if ti.name.endswith(".tmp") else ti)
        info = tarfile.TarInfo("MANIFEST.json")
        data = json.dumps(manifest).encode()
        info.size = len(data)
        import io

        t.addfile(info, io.BytesIO(data))
    (stage / "dataset-metadata.json").write_text(json.dumps({"title": SLUG, "id": f"{user()}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    print(f"bundle: {(stage / 'bundle.tar').stat().st_size / 1e6:.1f} MB ({', '.join(['pools'] + names)})")
    exists = run([KAGGLE, "datasets", "status", f"{user()}/{SLUG}"], check=False, capture=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        run([KAGGLE, "datasets", "version", "-p", str(stage), "-m", args.message or manifest["time"]])
    else:
        run([KAGGLE, "datasets", "create", "-p", str(stage)])  # private unless --public


def fetch(args) -> None:
    dest = Path(args.to) if args.to else RUNS / "kaggle" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=False)
    run([KAGGLE, "kernels", "output", f"{user()}/{KERNEL}", "-p", str(dest)])
    print(f"output in {dest}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("job", choices=["calibrate", "calibrate-alone", "stage0", "league"])
    p.add_argument("--hours", type=float, default=0.5, help="session budget, setup included")
    p.add_argument("--commit", default=None)
    p.add_argument("--net", default="32x4", help="network channels×blocks (both roles; D26: 32x4)")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--run", default="league-1", help="league run directory under runs/")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--explorer-init", default="runs/stage0/checkpoint.npz")
    p.add_argument("--designer-init", default="")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="more league options")
    p.add_argument("--dry-run", action="store_true")
    u = sub.add_parser("upload")
    u.add_argument("--runs", default="")
    u.add_argument("--message", default="")
    sub.add_parser("status")
    lg = sub.add_parser("logs")
    lg.add_argument("--seconds", type=int, default=60)
    f = sub.add_parser("fetch")
    f.add_argument("--to", default=None)
    sub.add_parser("quota")
    args = ap.parse_args(argv)
    if args.cmd == "push":
        push(args)
    elif args.cmd == "upload":
        upload(args)
    elif args.cmd == "status":
        run([KAGGLE, "kernels", "status", f"{user()}/{KERNEL}"])
    elif args.cmd == "logs":
        try:
            subprocess.run([KAGGLE, "kernels", "logs", "-f", f"{user()}/{KERNEL}"], timeout=args.seconds)
        except subprocess.TimeoutExpired:
            pass
    elif args.cmd == "fetch":
        fetch(args)
    elif args.cmd == "quota":
        run([KAGGLE, "quota"])


if __name__ == "__main__":
    main(sys.argv[1:])
