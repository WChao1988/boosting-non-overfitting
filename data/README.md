# Data sources

The seven benchmark datasets in this folder are unmodified copies downloaded
from the UCI Machine Learning Repository (https://archive.ics.uci.edu) on
October 17, 2026. They are redistributed here unmodified so that the
repository is self-contained; usage follows the terms of the UCI Machine
Learning Repository (most datasets are released under Creative Commons
licenses — see the per-dataset pages at archive.ics.uci.edu for details and
attribution requirements).

| File | UCI dataset | Size | Classes | Notes |
|---|---|---|---|---|
| `sonar.all-data` | Connectionist Bench (Sonar) | 208 x 60 | 2 | labels `R`/`M` in the last column |
| `ionosphere.data` | Ionosphere | 351 x 34 | 2 | the first two columns (constant 1/0) are dropped by the loader |
| `breast-cancer.data` | Breast Cancer (recurrence) | 286 x 10 | 2 | class is the FIRST column; the 9 categorical attributes are ordinal-encoded |
| `waveform.data` | Waveform Database Generator (Version 1) | 5000 x 22 | 3 | decompressed from UCI's `waveform.data.Z`; class label in the last column |
| `glass.data` | Glass Identification | 214 x 11 | 6 | id column (first) dropped by the loader |
| `segmentation.data` | Image Segmentation | 210 x 20 | 7 | contains Lisp-style `;;;` header lines and an attribute-name header, both skipped by the loader |
| `letter-recognition.data` | Letter Recognition | 20000 x 17 | 26 | letter label in the first column |

The synthetic datasets of the paper (two-class unit-ball Gaussians) are
generated on the fly by `common.py::make_gaussian_two_class` /
`make_separable` and require no storage.
