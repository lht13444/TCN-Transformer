# HighD sample

Source: the [TUM CommonRoad tutorial](https://commonroad.in.tum.de/tutorials/static_docs_tutorials_data_preprocessing.html) and its [CSV directory](https://gitlab.lrz.de/tum-cps/commonroad-rl/-/tree/master/commonroad_rl/tutorials/data/highD/raw/data).

The sample contains complete tracks for six vehicles from recording 03: 1,781 rows and six lane changes at 25 Hz. Track values retain their source values. The tracks metadata selects the same vehicles; the recording metadata retains the source recording summary.

The pinned source revision, download locations, units, and checksums are listed in [PROVENANCE.json](PROVENANCE.json). Download the sample with:

```bash
python scripts/fetch_highd_sample.py
```

The data is subject to the [HighD usage terms](https://levelxdata.com/highd-dataset/). Dataset paper: [Krajewski et al., IEEE ITSC 2018](https://doi.org/10.1109/ITSC.2018.8569552).
