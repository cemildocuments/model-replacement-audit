import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy.stats import hypergeom
from bounds import hg_upper, paired_upper


class BoundsTests(unittest.TestCase):
    def test_exhaustive_coverage(self):
        for D in range(1, 41):
            for n in range(D + 1):
                x = np.arange(n + 1)
                up = hg_upper(D, n, x, 0.05)
                for K in range(D + 1):
                    bad = hypergeom.pmf(x, D, K, n)[up < K].sum()
                    self.assertLessEqual(bad, 0.05 + 1e-12)

    def test_census(self):
        for D in [0, 1, 10, 100]:
            k = np.arange(D + 1)
            np.testing.assert_array_equal(hg_upper(D, D, k, 0.001), k)

    def test_empty(self):
        self.assertEqual(float(paired_upper(0, 0, 0, 0, 100, 0.001)), 0)
        self.assertEqual(int(hg_upper(100, 0, 0, 0.001)), 100)

    def test_monotonic(self):
        for n in [1, 10, 50]:
            self.assertTrue(
                np.all(np.diff(hg_upper(100, n, np.arange(n + 1), 0.001)) >= 0)
            )


if __name__ == "__main__":
    unittest.main()
