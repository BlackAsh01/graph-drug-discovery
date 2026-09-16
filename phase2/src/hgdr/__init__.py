"""HGDR: Heterogeneous-Graph Drug Recommendation with DDI awareness.

Phase 2 of the thesis *Safe medication recommendation over heterogeneous
EHR graphs*.  The package turns MIMIC-III admissions into a heterogeneous
entity graph (diagnoses, procedures, drugs), enriches drug nodes with
molecular (SMILES) features and a TWOSIDES drug-drug-interaction graph,
and trains a DDI-aware multi-label recommender.
"""

__version__ = "1.0.0"
