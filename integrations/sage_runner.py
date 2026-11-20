"""Run optional LatticeLab Sage integrations through SageMath's supported -c interface.

Some packaged SageMath launchers reject arguments after a .sage filename.
This wrapper passes script arguments via sys.argv inside the Sage interpreter.
No shell interpolation or executable code from user input is used.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "crosscheck": ROOT / "sage_crosscheck.sage",
    "estimate": ROOT / "estimate_lwe.sage",
}


def build_sage_command(command, arguments=()):
    if command not in SCRIPTS:
        raise ValueError(f"unknown SageMath integration: {command}")
    path = SCRIPTS[command]
    # All dynamic strings are encoded as Python string repr literals.
    # runpy executes only the fixed, version-controlled integration script.
    argv = [str(path), *arguments]
    code = ("import runpy, sys; "
            f"sys.argv = {argv!r}; "
            f"runpy.run_path({str(path)!r}, run_name='__main__')")
    return ["sage", "-c", code]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(SCRIPTS))
    args, passthrough = parser.parse_known_args(argv)
    if shutil.which("sage") is None:
        parser.error("sage executable not found in PATH; install SageMath first")
    completed = subprocess.run(build_sage_command(args.command, passthrough),
                               check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
