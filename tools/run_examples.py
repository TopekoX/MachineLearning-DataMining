"""Jalankan skrip contoh dan bandingkan keluarannya dengan examples/expected/.

Pemakaian (dari root repo, dengan Python yang berisi pustaka ML):
    python tools/run_examples.py                 # semua contoh, bandingkan
    python tools/run_examples.py --update        # tulis ulang keluaran yang diharapkan
    python tools/run_examples.py fundamentals    # hanya satu folder
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
EXPECTED = EX / "expected"
HELPERS = {"customer_data", "forecast_utils", "segment_data", "basket_data"}
CHAPTERS = ["fundamentals", "classification", "estimation", "forecasting",
            "clustering", "association"]


def main(argv):
    update = "--update" in argv
    names = [a for a in argv if not a.startswith("--")] or CHAPTERS
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
    failed = 0
    for chapter in names:
        for script in sorted((EX / chapter).glob("*.py")):
            if script.stem in HELPERS:  # berkas pembantu, bukan contoh mandiri
                continue
            rel = script.relative_to(EX).with_suffix(".txt")
            target = EXPECTED / rel
            proc = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                print(f"GAGAL  {script.relative_to(EX)}\n{proc.stderr[-800:]}")
                failed += 1
                continue
            out = proc.stdout.replace("\r\n", "\n")
            if update:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(out, encoding="utf-8", newline="\n")
                print(f"TULIS  {rel}")
            elif not target.exists() or target.read_text(encoding="utf-8") != out:
                print(f"BEDA   {rel}")
                failed += 1
            else:
                print(f"OK     {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
