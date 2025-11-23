import unittest

import torch


class TestGPUAvailability(unittest.TestCase):
    def test_cuda_available(self) -> None:
        """Ensure at least one CUDA device is visible to PyTorch."""
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.assertTrue(has_cuda, "CUDA GPU not detected; required for gn_demo.py.")


if __name__ == "__main__":
    unittest.main()
