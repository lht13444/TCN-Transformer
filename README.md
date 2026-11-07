# TCN-Transformer

Temporal feature components for high-speed lane-change trajectory research.

| Component | Implementation |
|---|---|
| ProbSparse attention | `components/attention.py` |
| Attention distillation | `components/distillation.py` |
| History Transformer backbone | `components/backbone.py` |
| Multi-scale causal TCN | `components/tcn.py` |
| Local/global cross-attention fusion | `components/fusion.py` |
| Trajectory prediction head | `components/head.py` |

Code is located in `src/lanechange/`. See the [component guide](docs/components.md) for interfaces and masking conventions.

## Run

Use Python 3.10 or later:

```bash
pip install -r requirements.txt
python run.py components
python run_tests.py
```

`examples/04_prediction_head.py` runs the prediction head independently.
`examples/05_network_assembly.py` shows the network assembly as commented code.

For CPU-only PyTorch, install this wheel before the requirements:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## Data

The paper uses **HighD** for its main experiments and **VTD** for cross-dataset generalization. Download a small HighD sample from the TUM CommonRoad tutorial:

```bash
python scripts/fetch_highd_sample.py
python scripts/inspect_sample.py --input data/highd_sample/03_tracks.csv --format highd
python run.py components
```

An additional **NGSIM US-101** interface example contains 12 vehicles and 6,783 source records:

```bash
python scripts/inspect_sample.py --input data/raw_samples/ngsim_us101_12vehicles.csv --format ngsim
```

Sources, units, and file checksums are recorded in the data directories. See the [HighD source notice](data/highd_sample/SOURCE_NOTICE.md) and [NGSIM data notice](data/raw_samples/DATA_LICENSE.md).

The software is licensed under the [MIT License](LICENSE). Third-party datasets retain the terms stated in their data directories.
