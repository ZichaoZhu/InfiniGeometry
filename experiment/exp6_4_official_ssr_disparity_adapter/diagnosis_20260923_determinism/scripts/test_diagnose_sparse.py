import unittest
import torch
from diagnose_sparse import canonical, compare, delta


class DiagnosticChecks(unittest.TestCase):
    def test_coordinate_permutation_is_not_feature_difference(self):
        coords = torch.tensor([[0, 1, 0, 3], [0, 0, 2, 1], [0, 0, 1, 2]])
        feats = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        order = torch.tensor([2, 0, 1])
        a, b = canonical(feats, coords), canonical(feats[order], coords[order])
        self.assertTrue(torch.equal(a['coords'], b['coords']))
        self.assertTrue(torch.equal(a['features'], b['features']))
        self.assertNotEqual(a['order_sha'], b['order_sha'])

    def test_exact_tolerance_and_shape(self):
        a = torch.tensor([1., 2.])
        self.assertTrue(delta(a, a)['exact'])
        self.assertFalse(delta(a, a + .001)['close_1e6'])
        with self.assertRaises(AssertionError):
            delta(a, a[:1])

    def test_voxel_boundary_change_count(self):
        a, b = torch.tensor([.00249]), torch.tensor([.00251])
        result = compare({0: a}, {0: b}, {'input_d_1': a}, {'input_d_1': b})
        self.assertEqual(result['trace']['input_d_1']['different_voxel_bins'], 1)


if __name__ == '__main__':
    unittest.main()
