import json
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from hexgen2.record import main, ROOT
from hexgen2.summarize import verify


class RecordTest(unittest.TestCase):
    def record(self, folder, run_id, command, timeout="10"):
        argv = ["record", "--run-id", run_id, "--node", "local", "--classification", "cpu-reference",
                "--placement", str(ROOT / "configs/local-reference-placement.json"),
                "--root", str(folder), "--timeout", timeout, "--", sys.executable, "-c", command]
        with patch.object(sys, "argv", argv), patch.dict("os.environ", {"FABRIC_ACCESS_TOKEN": "private-value"}):
            code = main()
        path = folder / run_id / "local"
        verify(path)
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertNotIn("FABRIC_ACCESS_TOKEN", manifest["environment"])
        self.assertTrue((path / "source/working-source.tar.gz").is_file())
        return code, path, manifest

    def test_failure_keeps_raw_logs_and_integrity_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, path, manifest = self.record(Path(temporary), "failure", "import sys;print('attempt retained');sys.exit(7)")
            self.assertEqual(code, 7)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("attempt retained", (path / "logs/stdout.log").read_text())
            with self.assertRaises(FileExistsError):
                self.record(Path(temporary), "failure", "pass")
            (path / "logs/stdout.log").write_text("tampered")
            with self.assertRaises(ValueError):
                verify(path)

    def test_timeout_terminates_child_and_finalizes_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, _, manifest = self.record(Path(temporary), "timeout", "import time;time.sleep(30)", timeout="0.1")
            self.assertEqual(code, 124)
            self.assertEqual(manifest["status"], "timed_out")
            self.assertIsNotNone(manifest["child_returncode"])

    def test_interrupt_preserves_bundle_and_terminates_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            command = [sys.executable, "-m", "hexgen2.record", "--run-id", "interrupt", "--node", "local",
                       "--classification", "cpu-reference", "--placement", str(ROOT / "configs/local-reference-placement.json"),
                       "--root", str(folder), "--timeout", "60", "--", sys.executable, "-c",
                       "import time;print('started',flush=True);time.sleep(30)"]
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            bundle = folder / "interrupt/local"
            deadline = time.monotonic() + 30
            try:
                while time.monotonic() < deadline:
                    log = bundle / "logs/stdout.log"
                    if log.exists() and "started" in log.read_text():
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Recorder child did not start")
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 130, stderr.decode())
                verify(bundle)
                manifest = json.loads((bundle / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "interrupted")
                self.assertIsNotNone(manifest["child_returncode"])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
