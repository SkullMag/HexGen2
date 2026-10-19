from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hexgen2.transport import recv_data, send_data, validate_header


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=Path(rendezvous).as_uri(),
                            timeout=timedelta(seconds=20))
    try:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            cache = {7: torch.arange(-24, 24).reshape(1, 4, 2, 2, 3).to(dtype)[..., ::2],
                     2: torch.full((1, 3, 2, 1, 4), -1.25, dtype=dtype)}
            token = torch.tensor([42, 17], dtype=torch.long)
            if rank == 0:
                send_data(token, cache, dst=1)
                received, returned = recv_data(src=1, num_layers=2)
            else:
                received, returned = recv_data(src=0, num_layers=2)
                send_data(received, returned, dst=0)
            torch.testing.assert_close(received, token, rtol=0, atol=0)
            if list(returned) != [7, 2]:
                raise AssertionError("Layer IDs/order changed")
            for key in cache:
                torch.testing.assert_close(returned[key], cache[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TransportTest(unittest.TestCase):
    def test_lossless_real_process_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(worker, args=(str(Path(temporary) / "rendezvous"),), nprocs=2, join=True)

    def test_invalid_metadata_is_rejected_before_allocating_payload(self):
        for shape in ([-1], [True], [2**40], [1] * 9):
            with self.assertRaises(ValueError):
                validate_header({"version": 1, "tensors": [{"name": "x", "shape": shape, "dtype": "torch.float32"}]})
        with self.assertRaises(ValueError):
            validate_header({"version": 1, "tensors": [
                {"name": "x", "shape": [1], "dtype": "torch.float32"},
                {"name": "x", "shape": [1], "dtype": "torch.float32"}]})


if __name__ == "__main__":
    unittest.main()
