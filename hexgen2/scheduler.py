"""Run the supplied CPU scheduler from a location-independent entry point."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1] / "benchmark/scheduling_algorithm_documents"
if "--machine_config_path" in sys.argv:
    i = sys.argv.index("--machine_config_path") + 1
    sys.argv[i] = str(Path(sys.argv[i]).resolve())
if "--output-json" in sys.argv:
    i = sys.argv.index("--output-json") + 1
    sys.argv[i] = str(Path(sys.argv[i]).resolve())
elif os.environ.get("HEXGEN2_RUN_DIR"):
    sys.argv.extend(["--output-json", str(Path(os.environ["HEXGEN2_RUN_DIR"]) / "scheduler-result.json")])
sys.path.insert(0, str(root))
os.chdir(root)
sys.argv[0] = str(root / "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")
