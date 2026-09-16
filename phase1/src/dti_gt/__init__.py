"""dti_gt: Graph-Transformer drug-target affinity prediction on KIBA.

Phase 1 of a two-phase thesis. The package is organised as

* :mod:`dti_gt.data`   - download, featurisation (SMILES -> graph, protein encodings), datasets, splits
* :mod:`dti_gt.models` - edge-aware graph transformer, GNN baselines, protein encoders, fusion, DTI model
* :mod:`dti_gt.train`  - configuration-driven training loop
* :mod:`dti_gt.evaluate` - evaluation of saved runs
* :mod:`dti_gt.utils`  - seeding, metrics, logging helpers
"""

__version__ = "1.0.0"
