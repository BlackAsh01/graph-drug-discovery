import torch

from dti_gt.data.featurize import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS, smiles_to_graph
from dti_gt.data.proteins import AA_VOCAB, format_for_protbert, sequence_to_composition, sequence_to_tokens


def test_smiles_to_graph_shapes():
    g = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin: 13 heavy atoms, 13 bonds
    assert g is not None
    assert g.x.shape == (13, len(ATOM_FEATURE_DIMS))
    assert g.edge_index.shape == (2, 26)
    assert g.edge_attr.shape == (26, len(BOND_FEATURE_DIMS))
    assert g.degree.shape == (13,)
    assert g.x.dtype == torch.long and g.edge_attr.dtype == torch.long
    # every categorical index must be inside its embedding table
    for col, n in enumerate(ATOM_FEATURE_DIMS):
        assert int(g.x[:, col].max()) < n
    for col, n in enumerate(BOND_FEATURE_DIMS):
        assert int(g.edge_attr[:, col].max()) < n
    assert int(g.edge_attr[:, 0].min()) >= 1  # bond type 0 is reserved for self loops


def test_invalid_and_single_atom_smiles():
    assert smiles_to_graph("not_a_smiles") is None
    g = smiles_to_graph("[Na+]")
    assert g is not None and g.num_nodes == 1 and g.edge_index.shape == (2, 0)


def test_protein_encodings():
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    tok = sequence_to_tokens(seq, max_len=30)
    assert tok.shape == (30,) and int(tok[len(seq)]) == 0 and int(tok[:len(seq)].min()) >= 1
    comp = sequence_to_composition(seq)
    assert comp.shape == (len(AA_VOCAB) + 1,)
    assert abs(float(comp[:-1].sum()) - 1.0) < 1e-5
    assert format_for_protbert("MKZ") == "M K X"
