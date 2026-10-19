import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hexgen2.fixture import create
from hexgen2.validate import validate

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def test_reference_matches_full_context_and_two_rank_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            create(folder / "model")
            result = validate(folder / "model", "hf-reference", "cpu", "float32")
            self.assertEqual(result["status"], "passed")
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                       "-m", "hexgen2.run", "--model-dir", str(folder / "model"),
                       "--trace", str(ROOT / "configs/tiny-trace.jsonl"), "--output", str(folder / "output"),
                       "--engine", "hf-reference", "--device", "cpu", "--dtype", "float32", "--check-colocated"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(s) for s in (folder / "output/requests.jsonl").read_text().splitlines()]
            self.assertEqual([r["output_tokens"] for r in records], [1, 1, 4, 3])
            self.assertEqual([r["warmup"] for r in records], [True, False, False, False])
            self.assertTrue(all(r["colocated_matches"] for r in records))
            self.assertTrue(all(r["kv_packet_bytes"] > 0 for r in records))
            for rank in (0, 1):
                events = [json.loads(s) for s in (folder / f"output/rank-{rank}-events.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["event"], "complete")
            # A retry must not overwrite earlier request records.
            retry = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=40)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("Refusing to overwrite", retry.stderr)


if __name__ == "__main__":
    unittest.main()
