# Component guide

Feature sequences use `[batch, time, channels]`. In boolean validity masks, `True` denotes an observed timestep. Padding is excluded from attention and pooling.

## Global temporal features

`ProjectedAttention` projects queries, keys, and values, computes attention, and projects the output. The default `probsparse` backend samples keys to estimate query importance; selected queries attend to all valid keys, while other queries use the mean of valid values as their initial context. Inference sampling is fixed. Read `last_mode` to identify the active backend. `dense` selects standard attention; `auto` selects a backend separately for each sample using its valid length.

`AttentionDistillation` applies convolution, ELU, and stride-2 masked max pooling, returning compressed features and their validity mask. It requires right padding. Each sample retains `ceil(valid_length / 2)` valid output tokens.

`TransformerBackbone` embeds observed features, adds positional encoding, and encodes history. It accepts left or right padding and returns `(features, valid_mask)` with right padding. The default two-layer example distills a 40-step history to 20 feature tokens. `GlobalEncoder` accepts already embedded features with right padding.

The sampled-query implementation uses a max-minus-sample-mean sparsity estimate. It is an independent implementation of the method's core idea. Sparse self-attention and dense cross-attention have separate computational costs.

## Multi-scale local features

`CausalTemporalBlock` uses two left-padded dilated convolutions, per-timestep LayerNorm, and a residual connection. Its feature at time `t` depends on inputs at or before `t`.

`MultiScaleCausalTCN` runs branches with different dilation rates in parallel and combines them using trainable softmax-normalized scale weights. Each branch has receptive field `1 + 2 * (kernel_size - 1) * dilation`. The `receptive_fields` and `scale_weights` properties expose these values.

## Feature fusion

`LocalGlobalFusion` uses local features as queries and global features as keys and values. The default `cross` mode applies dense cross-attention, a local residual connection, and normalization. `gated_cross` adds an optional learned gate; `sum` uses additive feature fusion. Both inputs require right padding and may have different temporal lengths.

The module returns `(fused_features, gate)`, with `gate=None` for `cross` and `sum`. Cross-attention reads global context into local queries and forms dense scores of size `local_length * global_length`.

## Prediction head

`TrajectoryPredictionHead` accepts `[batch, time, d_model]` features and an optional validity mask. It combines the last valid feature with the masked temporal mean, then applies normalization and an MLP to produce `[batch, horizon, output_dim]` direct position offsets. Each point is relative to the last observed position; outputs are not cumulative step increments. The caller defines target units and restores the coordinate origin when needed.

The default head returns 40 future points with two coordinates each. `examples/04_prediction_head.py` demonstrates it independently. `examples/05_network_assembly.py` contains a fully commented assembly example with a configuration dataclass, a multi-stage local tower, a distilled global tower, repeated context fusion, feature-return options, and the prediction head.

## Data interface

Input features are `[relative_x, relative_y, vx, vy, ax, ay]` in meters and seconds. The x-axis follows the driving direction; the y-axis points to the driver's left. Relative positions use the last observed position as the origin.

The HighD loader converts bounding-box corners to vehicle centers, aligns coordinates with the driving direction, and reads the frame rate from recording metadata, defaulting to 25 Hz. The NGSIM loader converts feet to meters. NGSIM source records are sampled at 10 Hz; the observation-window interface interpolates within the observed history to produce the default 40-step, 25 Hz input.

Validity masks describe padding. Interior missing timesteps must be imputed using observed history before calling `HistoryEmbedding` or `TransformerBackbone`. `compact_padding` prepares a contiguous observed sequence for components that require right padding.

## References

- Original project: [wanglong803/TCN-Transformer](https://github.com/wanglong803/TCN-Transformer), reference commit `1e087dbcc7d6d02b306c6cfd7028e33468182e31`.
- Sparse-query attention: [Informer](https://github.com/zhouhaoyi/Informer2020).
- Tensor interfaces: [PyTorch](https://pytorch.org/).
