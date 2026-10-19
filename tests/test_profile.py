import json
from pathlib import Path
import tempfile
import unittest

from hexgen2.profile import read_profile


class ProfileTest(unittest.TestCase):
    def test_repair_is_limited_to_the_known_exporter_defect(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.json"
            raw = b'{"traceEvents":[{"cat":"kernel","name":"flash_fwd","args":{"Process Group Description": ,"rank":1}}]}'
            path.write_bytes(raw)
            document, report = read_profile(path)
            self.assertIsNone(document["traceEvents"][0]["args"]["Process Group Description"])
            self.assertEqual(report["missing_process_group_descriptions_replaced_with_null"], 1)
            self.assertEqual(report["flash_kernel_count"], 1)
            self.assertEqual(path.read_bytes(), raw)
            path.write_text('{"traceEvents": ,"other":1}')
            with self.assertRaises(json.JSONDecodeError):
                read_profile(path)
