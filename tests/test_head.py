"""Behavioral checks for the independent trajectory-offset head."""

import unittest

import torch

from lanechange.components.head import TrajectoryPredictionHead


class TrajectoryPredictionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(197)

    def assertClose(self, actual, expected, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_default_and_custom_output_shapes(self):
        default = TrajectoryPredictionHead().eval()
        self.assertEqual(default(torch.randn(3, 17, 64)).shape, (3, 40, 2))
        custom = TrajectoryPredictionHead(d_model=5, horizon=7, output_dim=3, hidden_dim=11).eval()
        self.assertEqual(custom(torch.randn(2, 9, 5)).shape, (2, 7, 3))

    def test_outputs_are_direct_offsets_without_accumulation(self):
        head = TrajectoryPredictionHead(d_model=4, horizon=5, hidden_dim=8, dropout=0).eval()
        direct_offsets = torch.tensor([[0.5, -0.25]]).repeat(5, 1)
        with torch.no_grad():
            head.mlp[-1].weight.zero_()
            head.mlp[-1].bias.copy_(direct_offsets.flatten())
        output = head(torch.randn(2, 7, 4))
        self.assertClose(output, direct_offsets[None].expand(2, -1, -1))
        self.assertClose(output[:, -1], output[:, 0])

    def test_left_right_and_nonfinite_padding_do_not_change_output(self):
        head = TrajectoryPredictionHead(d_model=8, horizon=6, hidden_dim=12, dropout=0).eval()
        original = torch.randn(2, 9, 8)
        expected = head(original)
        for left, right in ((4, 0), (0, 5), (3, 4)):
            with self.subTest(left=left, right=right):
                padded = torch.cat((
                    torch.full((2, left, 8), float("nan")), original,
                    torch.full((2, right, 8), float("inf")),
                ), dim=1)
                valid = torch.zeros(2, left + 9 + right, dtype=torch.bool)
                valid[:, left:left + 9] = True
                self.assertClose(head(padded, valid), expected)

    def test_masked_interior_positions_are_excluded_from_pooling(self):
        head = TrajectoryPredictionHead(d_model=5, hidden_dim=9, horizon=3, dropout=0).eval()
        sequence = torch.randn(1, 4, 5)
        with_gap = torch.cat((sequence[:, :2], torch.full((1, 1, 5), float("nan")), sequence[:, 2:]), dim=1)
        valid = torch.tensor([[True, True, False, True, True]])
        self.assertClose(head(with_gap, valid), head(sequence))

    def test_gradients_reach_valid_features_and_head_parameters(self):
        head = TrajectoryPredictionHead(d_model=8, horizon=5, hidden_dim=16, dropout=0)
        features = torch.randn(2, 13, 8, requires_grad=True)
        valid = torch.arange(13)[None, :] < torch.tensor([[8], [13]])
        output = head(features, valid)
        (output - torch.randn_like(output)).square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(features.grad[~valid].abs().sum().item(), 0)
        self.assertGreater(features.grad[valid].abs().sum().item(), 0)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_short_histories_and_odd_channels_remain_finite(self):
        head = TrajectoryPredictionHead(d_model=5, hidden_dim=7, horizon=2, dropout=0).eval()
        for steps in (1, 3):
            output = head(torch.randn(2, steps, 5))
            self.assertEqual(output.shape, (2, 2, 2))
            self.assertTrue(torch.isfinite(output).all())

    def test_invalid_configurations_are_rejected(self):
        invalid = (
            {"d_model": 0}, {"d_model": True}, {"horizon": -1},
            {"horizon": 2.5}, {"output_dim": 0}, {"hidden_dim": 0},
            {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrajectoryPredictionHead(**kwargs)

    def test_invalid_masks_and_observed_values_are_rejected(self):
        head = TrajectoryPredictionHead(d_model=4)
        features = torch.randn(2, 5, 4)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            head(features, torch.zeros(2, 5, dtype=torch.bool))
        for mask in (torch.ones(2, 5), torch.ones(1, 5, dtype=torch.bool)):
            with self.subTest(mask_shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                head(features, mask)
        for nonfinite in (float("nan"), float("inf")):
            corrupt = features.clone()
            corrupt[0, 1, 0] = nonfinite
            with self.assertRaisesRegex(ValueError, "finite"):
                head(corrupt)
        with self.assertRaises(ValueError):
            head(torch.ones(2, 5, 4, dtype=torch.int64))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 5, 3))


if __name__ == "__main__":
    unittest.main()
