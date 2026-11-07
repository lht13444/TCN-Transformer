"""Behavioral tests for independent temporal feature components."""

import unittest

import torch

from lanechange.components import (
    AttentionDistillation, HistoryEmbedding, LocalGlobalFusion,
    MultiScaleCausalTCN, ProjectedAttention, SinusoidalPositionEncoding,
    TransformerBackbone, compact_padding,
)


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def assertClose(self, actual, expected, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_tcn_cannot_read_future_timesteps(self):
        tcn = MultiScaleCausalTCN(d_model=12, dropout=0).eval()
        original = torch.randn(2, 17, 12)
        changed = original.clone()
        changed[:, 9:] = torch.randn_like(changed[:, 9:]) * 20
        self.assertClose(tcn(original)[:, :9], tcn(changed)[:, :9])
        self.assertGreater((tcn(original)[:, 9:] - tcn(changed)[:, 9:]).abs().max().item(), 1e-4)

    def test_scale_weights_and_each_tcn_branch_receive_gradients(self):
        tcn = MultiScaleCausalTCN(d_model=12, dropout=0)
        x = torch.randn(2, 19, 12, requires_grad=True)
        target_features = torch.randn_like(x)
        (tcn(x) - target_features).square().mean().backward()
        self.assertClose(tcn.scale_weights.sum(), torch.tensor(1.0))
        self.assertTrue((tcn.scale_weights > 0).all())
        self.assertEqual(tcn.receptive_fields, (5, 9, 17))
        for parameter in [tcn.scale_logits, *(branch.conv1.weight for branch in tcn.branches)]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_attention_depends_on_keys_for_both_backends(self):
        query, key, value = (torch.randn(2, 19, 12) for _ in range(3))
        for mode in ("dense", "probsparse"):
            with self.subTest(mode=mode):
                attention = ProjectedAttention(12, 3, dropout=0, mode=mode, sparse_factor=1).eval()
                baseline = attention(query, key, value)
                changed_key = key.clone()
                changed_key[:, :8] *= -4
                changed = attention(query, changed_key, value)
                self.assertGreater((baseline - changed).abs().max().item(), 1e-4)

    def test_attention_masks_queries_keys_and_padding_gradients(self):
        for mode in ("dense", "probsparse"):
            with self.subTest(mode=mode):
                attention = ProjectedAttention(12, 3, dropout=0, mode=mode, sparse_factor=1).eval()
                query = torch.randn(2, 13, 12, requires_grad=True)
                key = torch.randn(2, 17, 12, requires_grad=True)
                value = torch.randn(2, 17, 12, requires_grad=True)
                q_mask = torch.arange(13)[None, :] < torch.tensor([[9], [13]])
                k_mask = torch.arange(17)[None, :] < torch.tensor([[11], [17]])
                output = attention(query, key, value, q_mask, k_mask)
                corrupt_q, corrupt_k, corrupt_v = query.detach().clone(), key.detach().clone(), value.detach().clone()
                corrupt_q[~q_mask] = float("nan")
                corrupt_k[~k_mask] = float("nan")
                corrupt_v[~k_mask] = float("inf")
                self.assertClose(output, attention(corrupt_q, corrupt_k, corrupt_v, q_mask, k_mask))
                self.assertEqual(output[~q_mask].abs().sum().item(), 0)
                (output - torch.randn_like(output)).square().sum().backward()
                self.assertEqual(query.grad[~q_mask].abs().sum().item(), 0)
                self.assertEqual(key.grad[~k_mask].abs().sum().item(), 0)
                self.assertEqual(value.grad[~k_mask].abs().sum().item(), 0)
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                    self.assertTrue(torch.isfinite(projection.weight.grad).all())
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_sparse_evaluation_is_deterministic_and_batch_independent(self):
        attention = ProjectedAttention(12, 3, dropout=0.2, mode="probsparse", sparse_factor=1).eval()
        x = torch.randn(1, 37, 12)
        first = attention(x, x, x)
        torch.randn(1000)
        self.assertClose(first, attention(x, x, x), atol=0, rtol=0)
        companion = torch.randn_like(x)
        batch = torch.cat((companion, x), dim=0)
        self.assertClose(first, attention(batch, batch, batch)[1:])

    def test_auto_backend_routing_is_per_sample(self):
        attention = ProjectedAttention(12, 3, dropout=0, mode="auto", sparse_threshold=8).eval()
        short, long = torch.randn(1, 7, 12), torch.randn(1, 19, 12)
        padded_short = torch.cat((short, torch.zeros(1, 12, 12)), dim=1)
        batch = torch.cat((padded_short, long), dim=0)
        mask = torch.arange(19)[None, :] < torch.tensor([[7], [19]])
        together = attention(batch, batch, batch, mask, mask)
        self.assertEqual(attention.last_mode, "mixed")
        self.assertClose(together[:1, :7], attention(short, short, short))
        self.assertEqual(attention.last_mode, "dense")
        self.assertClose(together[1:], attention(long, long, long))
        self.assertEqual(attention.last_mode, "probsparse")

    def test_distillation_odd_lengths_and_tail_masks(self):
        distill = AttentionDistillation(12).eval()
        x = torch.randn(2, 9, 12)
        valid = torch.arange(9)[None, :] < torch.tensor([[4], [9]])
        output, mask = distill(x, valid)
        self.assertEqual(output.shape, (2, 5, 12))
        self.assertEqual(mask.sum(1).tolist(), [2, 5])
        alone, _ = distill(x[:1, :4])
        self.assertClose(alone, output[:1, :2])
        corrupt = x.clone()
        corrupt[~valid] = float("nan")
        self.assertClose(output, distill(corrupt, valid)[0])
        self.assertEqual(output[~mask].abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, "right padding"):
            distill(x[:1], torch.tensor([[False, False, True, True, True, True, True, True, True]]))

    def test_distillation_receives_finite_gradients(self):
        distill = AttentionDistillation(12)
        x = torch.randn(2, 11, 12, requires_grad=True)
        features, mask = distill(x)
        (features - torch.randn_like(features)).square().mean().backward()
        self.assertTrue(mask.all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(distill.conv.weight.grad.abs().sum().item(), 0)

    def test_fusion_features_gate_mask_and_gradients(self):
        for mode in ("cross", "gated_cross", "sum"):
            with self.subTest(mode=mode):
                fusion = LocalGlobalFusion(12, 3, dropout=0, mode=mode).eval()
                local = torch.randn(2, 13, 12, requires_grad=True)
                global_features = torch.randn(2, 7, 12, requires_grad=True)
                local_valid = torch.arange(13)[None, :] < torch.tensor([[9], [13]])
                global_valid = torch.arange(7)[None, :] < torch.tensor([[5], [7]])
                output, gate = fusion(local, global_features, local_valid, global_valid)
                self.assertEqual(output.shape, local.shape)
                self.assertEqual(output[~local_valid].abs().sum().item(), 0)
                alone, _ = fusion(local[:1, :9], global_features[:1, :5])
                self.assertClose(alone, output[:1, :9])
                if gate is not None:
                    self.assertTrue(((gate[local_valid] > 0) & (gate[local_valid] < 1)).all())
                (output - torch.randn_like(output)).square().sum().backward()
                self.assertGreater(local.grad.abs().sum().item(), 0)
                self.assertGreater(global_features.grad.abs().sum().item(), 0)
                if mode == "gated_cross":
                    self.assertGreater(fusion.gate.weight.grad.abs().sum().item(), 0)

    def test_default_cross_fusion_key_dependence_and_projection_gradients(self):
        fusion = LocalGlobalFusion(12, 3, dropout=0).eval()
        self.assertEqual(fusion.mode, "cross")
        self.assertEqual(fusion.cross_attention.mode, "dense")
        local = torch.randn(2, 13, 12, requires_grad=True)
        global_features = torch.randn(2, 7, 12, requires_grad=True)
        output, gate = fusion(local, global_features)
        self.assertIsNone(gate)
        context = fusion.cross_attention(local, global_features, global_features)
        self.assertClose(output, fusion.output_norm(local + context))
        # Hold values fixed while changing keys to exercise attention scores.
        changed_keys = global_features.detach().clone()
        changed_keys[:, :3] *= -5
        changed_context = fusion.cross_attention(local, changed_keys, global_features)
        self.assertGreater((context - changed_context).abs().max().item(), 1e-4)
        (output - torch.randn_like(output)).square().mean().backward()
        for projection in (fusion.cross_attention.q_proj, fusion.cross_attention.k_proj, fusion.cross_attention.v_proj):
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_embedding_padding_canonicalization_and_odd_dimensions(self):
        embedding = HistoryEmbedding(6, 15, dropout=0).eval()
        history = torch.randn(1, 9, 6)
        expected, _ = embedding(history)
        padded = torch.cat((torch.full((1, 3, 6), float("nan")), history, torch.randn(1, 4, 6)), dim=1)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[:, 3:12] = True
        actual, actual_mask = embedding(padded, mask)
        self.assertClose(actual[:, :9], expected)
        self.assertEqual(actual_mask.sum().item(), 9)
        self.assertEqual(actual[~actual_mask].abs().sum().item(), 0)
        position = SinusoidalPositionEncoding(15)
        self.assertTrue(torch.isfinite(position(torch.zeros(1, 1, 15))).all())

    def test_history_backbone_features_are_batch_and_padding_independent(self):
        for mode in ("dense", "probsparse"):
            with self.subTest(mode=mode):
                backbone = TransformerBackbone(6, 12, 3, 24, 2, dropout=0, attention=mode).eval()
                first, companion = torch.randn(1, 8, 6), torch.randn(1, 23, 6)
                alone, alone_mask = backbone(first)
                padded = torch.cat((first, torch.zeros(1, 15, 6)), dim=1)
                mask = torch.arange(23)[None, :] < torch.tensor([[8], [23]])
                together, together_mask = backbone(torch.cat((padded, companion)), mask)
                self.assertEqual(alone.shape, (1, 4, 12))
                self.assertEqual(alone_mask.sum().item(), together_mask[0].sum().item())
                self.assertClose(alone, together[:1, :4])
                self.assertEqual(backbone.attention_modes, (mode, mode))

    def test_backbone_short_histories_and_feature_gradients(self):
        backbone = TransformerBackbone(6, 15, 3, 24, 2, dropout=0)
        for length in (1, 3, 11):
            features, valid = backbone(torch.randn(2, length, 6))
            self.assertEqual(features.shape, (2, (length + 1) // 2, 15))
            self.assertTrue(valid.all())
            self.assertTrue(torch.isfinite(features).all())
        x = torch.randn(2, 21, 6, requires_grad=True)
        features, _ = backbone(x)
        (features - torch.randn_like(features)).square().mean().backward()
        self.assertGreater(backbone.encoder.layers[0].attention.q_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(backbone.encoder.distillers[0].conv.weight.grad.abs().sum().item(), 0)

    def test_invalid_masks_and_component_configuration_rejected(self):
        with self.assertRaises(ValueError):
            ProjectedAttention(d_model=13, n_heads=4)
        with self.assertRaises(ValueError):
            MultiScaleCausalTCN(dilations=())
        x = torch.randn(1, 5, 6)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            compact_padding(x, torch.zeros(1, 5, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "interior time gaps"):
            compact_padding(x, torch.tensor([[True, True, False, True, True]]))


if __name__ == "__main__":
    unittest.main()
