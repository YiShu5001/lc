from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def build_jobs() -> list[tuple[str, str, str]]:
    return [
        (
            "exp-v7ep500decay-slow",
            "lc.entrypoints.train_x_axis_rl_refline_v7_ep500_decay_slow",
            "outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2",
        ),
        (
            "exp-v7ep500decay-slow-tau02",
            "lc.entrypoints.train_x_axis_rl_refline_v7_ep500_decay_slow_tau02",
            "outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2",
        ),
        (
            "exp-v7ep500decay-slow-tau02-net768",
            "lc.entrypoints.train_x_axis_rl_refline_v7_ep500_decay_slow_tau02_net768",
            "outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02-net768__ep-500__v-7__noise-linear-0.1-to-0.04__net-768__drop-0.25",
        ),
    ]


def _launch(module_name: str, output_dir: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    print(f"[launch] module={module_name} output={output_dir}", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", module_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def main() -> int:
    jobs = build_jobs()
    processes = [(name, output_dir, _launch(module_name, output_dir)) for name, module_name, output_dir in jobs]
    failures: list[tuple[str, str, int]] = []
    for name, output_dir, proc in processes:
        if proc.stdout is not None:
            for line in proc.stdout:
                print(f"[{name}] {line}", end="")
        code = proc.wait()
        print(f"[done] name={name} exit_code={code} output={output_dir}", flush=True)
        if code != 0:
            failures.append((name, output_dir, code))
    if failures:
        print("[summary] failures detected", flush=True)
        for name, output_dir, code in failures:
            print(f"[failure] name={name} exit_code={code} output={output_dir}", flush=True)
        return 1
    print("[summary] all experiments completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
