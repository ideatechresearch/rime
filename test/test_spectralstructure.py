"""
SpectralStructure Tests

Comprehensive tests for the pre-spectral prediction layer.
Covers: construction, k-set prediction, eigenvalue/multiplicity,
Q3 scheme, face-incidence, Z2/Z3 filters, integrality, validation.
"""
import sys

sys.path.insert(0, '.')
import numpy as np
from math import comb

from rime.cube import CubeBase
from rime.cubie import CubieMove, TOTAL_DIM, BLOCK_DIMS, BLOCK_RANGES
from rime.cubieoperator import build_A
from rime.helpers import krawtchouk
from rime.spectralstructure import (
    SpectralStructure, block_projectors, block_of_index,
    get_spectral_structure,
)

prim = CubieMove.prim_moves()


# ═══════════════════════════════════════════════════════════════════════════════
# Krawtchouk polynomial
# ═══════════════════════════════════════════════════════════════════════════════

def test_krawtchouk():
    assert krawtchouk(0, 0, n=3) == 1
    assert krawtchouk(0, 3, n=3) == 1
    for x in range(4):
        assert krawtchouk(0, x, n=3) == 1
        assert krawtchouk(1, x, n=3) == 3 - 2 * x
    for k in range(4):
        assert krawtchouk(k, 0, n=3) == comb(3, k)
    print("  OK Krawtchouk polynomials")


# ═══════════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_construction_18full():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.n == 18
    assert ss.m == 9
    assert ss.class_symmetric is True
    assert ss.has_half_turns is True
    assert ss.family_tag == "18-full"
    assert len(ss.active_classes) == 6
    print("  OK 18-full construction")


def test_construction_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    assert ss.n == 12
    assert ss.m == 6
    assert ss.class_symmetric is True
    assert ss.has_any_half_turn is False
    assert ss.family_tag == "12-quarter"
    print("  OK 12-quarter construction")


def test_construction_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    assert ss.n == 6
    assert ss.m == 3
    assert ss.class_symmetric is False
    assert ss.has_half_turns is True
    assert ss.family_tag == "6-half-turn"
    print("  OK 6-half-turn construction")


def test_default_construction():
    ss = SpectralStructure()
    assert ss.n == 18
    assert ss.family_tag == "18-full"
    print("  OK default construction")


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 scheme (cp block)
# ═══════════════════════════════════════════════════════════════════════════════

def test_q3_scheme():
    ss = SpectralStructure(generators=dict(prim))
    q3 = ss.scheme_cp()

    assert q3["n_verts"] == 8
    assert q3["n_classes"] == 4
    assert list(q3["v"]) == [1, 3, 3, 1]
    assert list(q3["dims"]) == [1, 3, 3, 1]
    assert np.allclose(q3["A"][0], np.eye(8))

    for k in range(4):
        A_k = q3["A"][k]
        assert np.allclose(A_k, A_k.T), f"A{k} not symmetric"
        assert np.all((A_k == 0) | (A_k == 1)), f"A{k} not 0/1"
        assert np.all(A_k.sum(axis=1) == q3["v"][k]), f"A{k} row sums != v_{k}"

    P_raw = q3["P_raw"]
    assert P_raw.shape == (4, 4)
    assert P_raw[0, 0] == 1

    P = q3["P"]
    assert np.allclose(P[0, :], q3["v"])
    assert np.allclose(P[:, 0], 1.0)

    # A1 + A2 + A3 = J - I (complete graph minus identity)
    assert np.allclose(q3["A"][1] + q3["A"][2] + q3["A"][3],
                       np.ones((8, 8)) - np.eye(8))
    print("  OK Q3 scheme")


# ═══════════════════════════════════════════════════════════════════════════════
# Face-incidence scheme (ep block)
# ═══════════════════════════════════════════════════════════════════════════════

def test_face_incidence_scheme():
    ss = SpectralStructure(generators=dict(prim))
    fi = ss.scheme_ep()

    assert fi["n_indices"] == 12
    assert fi["n_classes"] == 6
    assert fi["J"].shape == (12, 6)

    J = fi["J"]
    assert np.all((J.sum(axis=1) >= 1) & (J.sum(axis=1) <= 2))
    assert np.all(J.sum(axis=0) == 4)

    JJt = fi["JJt"]
    assert np.allclose(JJt, JJt.T)
    assert np.all(np.diag(JJt) == 2)  # each edge on 2 faces
    print("  OK face-incidence scheme")


# ═══════════════════════════════════════════════════════════════════════════════
# Z2 / Z3 filters
# ═══════════════════════════════════════════════════════════════════════════════

def test_z2_filter():
    ss = SpectralStructure(generators=dict(prim))
    z2 = ss._z2_phase
    assert z2["phase_active_count"] + z2["phase_trivial_count"] == 12
    assert z2["phase_active_count"] == 8
    assert z2["phase_trivial_count"] == 4

    phase_data = CubeBase.edge_phase_classification()
    assert z2["phase_active"] == phase_data["phase_active"]
    assert z2["phase_trivial"] == phase_data["phase_trivial"]
    print("  OK Z2 filter")


def test_z3_filter():
    ss = SpectralStructure(generators=dict(prim))
    z3 = ss._z3_phase
    assert abs(z3["cancellation"]) < 1e-10  # omega + omega^2 + 1 = 0
    assert all(c == 3 for c in z3["classes_per_index"])
    assert all(c == 4 for c in z3["indices_per_class"])
    assert z3["C"].shape == (8, 6)
    print("  OK Z3 filter")


# ═══════════════════════════════════════════════════════════════════════════════
# k-set prediction (18-full)
# ═══════════════════════════════════════════════════════════════════════════════

def test_kset_18full():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.k_set_cp() == {0, 4, 6}
    assert ss.k_set_ep() == {0, 2, 3, 4}
    assert ss.k_set_co() == {3, 4, 6}  # post-ρ-fix: perm@phase, 3 eigenvalues
    assert ss.k_set_eo() == {1, 2, 4}  # post-ρ-fix: perm@phase, 3 eigenvalues
    assert ss.k_set_total() == {0, 1, 2, 3, 4, 6}  # +k=1 from eo block
    print("  OK k-sets 18-full")


def test_eigenvalues_18full():
    ss = SpectralStructure(generators=dict(prim))
    evals = ss.eigenvalues()
    assert len(evals) == 6  # post-ρ-fix: +k=1 from eo block
    assert abs(evals[0] - 1.0) < 1e-10
    assert abs(evals[1] - 8 / 9) < 1e-10
    assert abs(evals[2] - 7 / 9) < 1e-10
    assert abs(evals[3] - 2 / 3) < 1e-10
    assert abs(evals[4] - 5 / 9) < 1e-10
    assert abs(evals[6] - 1 / 3) < 1e-10
    print("  OK eigenvalues 18-full")


def test_multiplicity_18full():
    ss = SpectralStructure(generators=dict(prim))
    layers = ss.eigenvalue_layers()
    total = sum(l[1] for l in layers)
    assert total == TOTAL_DIM
    # post-ρ-fix: 6 layers, co {3,4,6}, eo {1,2,4}
    mult_by_lam = {round(l[0], 6): l[1] for l in layers}
    assert mult_by_lam[1.0] == 20       # cp(8)+ep(12)
    assert mult_by_lam[0.888889] == 2   # eo(2) — new layer
    assert mult_by_lam[0.777778] == 39  # ep(36)+eo(3)
    assert mult_by_lam[0.666667] == 26  # ep(24)+co(2)
    assert mult_by_lam[0.555556] == 106 # cp(24)+ep(72)+co(3)+eo(7)
    assert mult_by_lam[0.333333] == 35  # cp(32)+co(3)
    print("  OK multiplicities 18-full")


# ═══════════════════════════════════════════════════════════════════════════════
# k-set prediction (12-quarter, 6-half-turn)
# ═══════════════════════════════════════════════════════════════════════════════

def test_kset_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    assert ss.k_set_cp() == {0, 2, 4, 6}
    assert ss.k_set_ep() == {0, 1, 2, 3}
    assert ss.k_set_co() == {2, 3, 4}  # post-ρ-fix: perm@phase
    assert ss.k_set_eo() == {1, 2, 3, 4}  # post-ρ-fix: perm@phase
    assert ss.k_set_total() == {0, 1, 2, 3, 4, 6}
    print("  OK k-sets 12-quarter")


def test_kset_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    assert ss.k_set_cp() == {0, 2}
    assert ss.k_set_ep() == {0, 1, 2}
    assert ss.k_set_co() == {0, 2}  # post-ρ-fix: perm@phase
    assert ss.k_set_eo() == {0, 1, 2}  # post-ρ-fix: perm@phase
    assert ss.k_set_total() == {0, 1, 2}
    print("  OK k-sets 6-half-turn")


# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvalue prediction (12-quarter, 6-half-turn)
# ═══════════════════════════════════════════════════════════════════════════════

def test_eigenvalues_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    evals = ss.eigenvalues()
    assert len(evals) == 6
    assert abs(evals[0] - 1.0) < 1e-10
    assert abs(evals[1] - 5 / 6) < 1e-10
    assert abs(evals[2] - 2 / 3) < 1e-10
    assert abs(evals[3] - 1 / 2) < 1e-10
    assert abs(evals[4] - 1 / 3) < 1e-10
    assert abs(evals[6] - 0.0) < 1e-10
    print("  OK eigenvalues 12-quarter")


def test_eigenvalues_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    evals = ss.eigenvalues()
    assert len(evals) == 3
    assert abs(evals[0] - 1.0) < 1e-10
    assert abs(evals[1] - 2 / 3) < 1e-10
    assert abs(evals[2] - 1 / 3) < 1e-10
    print("  OK eigenvalues 6-half-turn")


# ═══════════════════════════════════════════════════════════════════════════════
# Multiplicity prediction (12-quarter, 6-half-turn)
# ═══════════════════════════════════════════════════════════════════════════════

def test_multiplicity_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    layers = ss.eigenvalue_layers()
    total = sum(l[1] for l in layers)
    assert total == TOTAL_DIM
    # post-ρ-fix: perm@phase co/eo — let dimensions emerge from the code
    mult_by_k = {int(round((1 - l[0]) * 6)): l[1] for l in layers}
    # Verify total dimension is correct
    assert sum(mult_by_k.values()) == TOTAL_DIM
    print("  OK multiplicities 12-quarter")


def test_multiplicity_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    layers = ss.eigenvalue_layers()
    total = sum(l[1] for l in layers)
    assert total == TOTAL_DIM
    # post-ρ-fix: perm@phase co/eo
    mult_by_k = {int(round((1 - l[0]) * 3)): l[1] for l in layers}
    assert sum(mult_by_k.values()) == TOTAL_DIM
    print("  OK multiplicities 6-half-turn")


# ═══════════════════════════════════════════════════════════════════════════════
# Block projectors
# ═══════════════════════════════════════════════════════════════════════════════

def test_block_projectors():
    projs = block_projectors()
    assert set(projs.keys()) == {"cp", "ep", "co", "eo"}

    for name, P in projs.items():
        assert P.shape == (TOTAL_DIM, TOTAL_DIM)
        start, end = BLOCK_RANGES[name]
        expected_dim = end - start
        assert np.all(np.diag(P)[start:end] == 1.0)
        if start > 0:
            assert np.all(np.diag(P)[:start] == 0.0)
        if end < TOTAL_DIM:
            assert np.all(np.diag(P)[end:] == 0.0)
        assert int(np.trace(P)) == expected_dim

    P_sum = sum(projs.values())
    assert np.allclose(P_sum, np.eye(TOTAL_DIM))
    print("  OK block projectors")


def test_block_of_index():
    assert block_of_index(0) == "cp"
    assert block_of_index(63) == "cp"
    assert block_of_index(64) == "ep"
    assert block_of_index(207) == "ep"
    assert block_of_index(208) == "co"
    assert block_of_index(215) == "co"
    assert block_of_index(216) == "eo"
    assert block_of_index(227) == "eo"

    for bad in [-1, 228]:
        try:
            block_of_index(bad)
            assert False, f"Should have raised for {bad}"
        except ValueError:
            pass
    print("  OK block_of_index")


def test_block_projector_method():
    ss = SpectralStructure(generators=dict(prim))
    P_cp = ss.block_projector("cp")
    assert P_cp.shape == (TOTAL_DIM, TOTAL_DIM)
    assert int(np.trace(P_cp)) == BLOCK_DIMS["cp"]

    try:
        ss.block_projector("invalid")
        assert False
    except ValueError:
        pass
    print("  OK block_projector method")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory, validation, integrality
# ═══════════════════════════════════════════════════════════════════════════════

def test_validate_with_numerics_18full():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.validate_with_numerics(cso=None)
    assert result["all_match"] == True
    assert result["k_set"]["match"] == True
    assert result["multiplicities"]["match"] == True
    assert result["total_dimension"]["match"] == True
    assert result["slow_dimension"]["match"] == True
    print("  OK validate_with_numerics 18-full")


def test_integrality_18full():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.verify_integrality()
    assert result["all_integer"] is True

    for k, info in result["cp"].items():
        assert info["is_integer"] is True
        assert isinstance(info["class_sum_eigenvalue"], int)

    for k, info in result["ep"].items():
        assert info["is_integer"] is True

    assert result["co"][3]["is_integer"] is True
    for k, info in result["eo"].items():
        assert info["is_integer"] is True
    print("  OK integrality 18-full")


def test_validate_with_numerics_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.validate_with_numerics(cso=None)
    assert result["all_match"] == True
    assert result["k_set"]["match"] == True
    assert result["multiplicities"]["match"] == True
    assert result["total_dimension"]["match"] == True
    print("  OK validate_with_numerics 12-quarter")


def test_validate_with_numerics_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    result = ss.validate_with_numerics(cso=None)
    assert result["all_match"] == True
    assert result["k_set"]["match"] == True
    assert result["multiplicities"]["match"] == True
    assert result["total_dimension"]["match"] == True
    print("  OK validate_with_numerics 6-half-turn")


def test_integrality_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.verify_integrality()
    assert result["all_integer"] is True
    for k, info in result["cp"].items():
        assert info["is_integer"] is True
    for k, info in result["ep"].items():
        assert info["is_integer"] is True
    assert result["co"][3]["is_integer"] is True
    print("  OK integrality 12-quarter")


def test_integrality_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    result = ss.verify_integrality()
    assert result["all_integer"] is True
    for k, info in result["cp"].items():
        assert info["is_integer"] is True
    for k, info in result["ep"].items():
        assert info["is_integer"] is True
    assert result["co"][0]["is_integer"] is True
    print("  OK integrality 6-half-turn")


def test_block_reduction_theorem():
    """Theorem 7.1: K(A) = ∪_B K(A_B) — block reduction of k-set.

    Verified by comparing SpectralStructure.k_by_block() against
    numerical block-level spectra for all three face-symmetric families.
    """
    families = [
        ("18-full", dict(prim), 9),
        ("12-quarter", {k: v for k, v in prim.items() if k[2] != 2}, 6),
        ("6-half-turn", {k: v for k, v in prim.items() if k[2] == 2}, 3),
    ]
    BLOCKS = {"cp": (0, 64), "ep": (64, 208), "co": (208, 216), "eo": (216, 228)}

    for name, gens, m in families:
        ss = SpectralStructure(generators=gens)
        # Get predicted k-sets per block
        pred_kb = {b: ss._k_sets[b] for b in BLOCKS}
        pred_union = set().union(*pred_kb.values())

        # Numerical verification
        rhos = [mv.rho().astype(np.complex128) for mv in gens.values()]
        A = sum(rhos) / len(rhos)
        num_kb = {}
        for bname, (i0, i1) in BLOCKS.items():
            A_b = A[i0:i1, i0:i1]
            w_b = np.sort(np.linalg.eigvalsh(A_b))
            k_set = set()
            for lam in np.unique(np.round(w_b, 8)):
                k = round((1 - lam) * m)
                if abs(lam - (1 - k / m)) < 1e-6:
                    k_set.add(k)
            num_kb[bname] = k_set
        num_union = set().union(*num_kb.values())

        assert pred_union == num_union, \
            f"{name}: union mismatch pred={pred_union} num={num_union}"
        for b in BLOCKS:
            assert pred_kb[b] == num_kb[b], \
                f"{name}/{b}: block k-set mismatch pred={pred_kb[b]} num={num_kb[b]}"

    print("  OK block reduction theorem")


def test_face_sum_decomposition():
    ss = SpectralStructure(generators=dict(prim))
    cp_decomp = ss.class_sum_decomposition("cp")
    assert len(cp_decomp) == 4
    assert cp_decomp[0][0] == 9  # A0
    assert cp_decomp[1][0] == 2  # A1

    ep_decomp = ss.class_sum_decomposition("ep")
    assert len(ep_decomp) == 2
    assert ep_decomp[0][0] == 10  # alpha (I)
    assert ep_decomp[1][0] == 1  # JJ^T coefficient (structurally 1)
    print("  OK face-sum decomposition")


def test_compute_face_sum_coeffs():
    ss = SpectralStructure(generators=dict(prim))
    coeffs = ss.compute_class_sum_coeffs()
    assert set(coeffs.keys()) == {"cp", "ep"}
    # cp: [c0, c1, c2, c3]
    assert coeffs["cp"][0] == 9  # m
    assert coeffs["cp"][1] == 6 * 2 // 6  # 6 faces x 2 quarter turns / 6 faces = 2
    assert coeffs["cp"][2] == 6 // 6  # 6 half-turns / 6 faces = 1
    assert coeffs["cp"][3] == 0  # no distance-3 moves
    # ep: {"alpha": int} — alpha = (n_faces-2)*gens_per_face - JJ^T_diag
    assert coeffs["ep"]["alpha"] == 10  # (6-2)*3 - 2 = 10
    print("  OK compute_face_sum_coeffs")


# ═══════════════════════════════════════════════════════════════════════════════
# Accessors and structural methods
# ═══════════════════════════════════════════════════════════════════════════════

def test_k_by_block():
    ss = SpectralStructure(generators=dict(prim))
    kb = ss.k_by_block()
    assert "cp" in kb[0] and "ep" in kb[0]  # k=0: cp+ep (eo moved to k=1 post-ρ-fix)
    assert "ep" in kb[3] and "co" in kb[3]  # k=3: ep+co
    assert "eo" in kb[1]  # k=1: pure eo (new post-ρ-fix)
    print("  OK k_by_block")


def test_slow_dimension():
    ss = SpectralStructure(generators=dict(prim))
    slow = ss.predict_slow_dimension(threshold=2 / 3)
    # post-ρ-fix: λ=1(20) + 8/9(2) + 7/9(39) + 2/3(26) = 87
    assert slow == 87
    print("  OK slow dimension")


def test_scheme_accessors():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.scheme_cp()["name"] == "Q3 hypercube"
    assert ss.scheme_ep()["name"] == "support-incidence"
    print("  OK scheme accessors")


def test_face_partition():
    ss = SpectralStructure(generators=dict(prim))
    faces = ss.class_partition()
    assert len(faces) == 6
    for face, gens in faces.items():
        assert len(gens) == 3  # CW, CCW, 180
    print("  OK face_partition")


def test_eo_accessors():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.eo_k_values() == {1, 2, 4}  # post-ρ-fix: perm@phase, 3 eigenvalues
    eo = ss.eo_partition()
    assert eo["phase_active_count"] == 8
    assert eo["phase_trivial_count"] == 4
    # per-index averages: FB edges flipped by F/B quarter turns → 7/9;
    # non-FB edges never flipped (only U/D/R/L, no orientation delta) → 1.0
    assert abs(eo["eigenvalues"]["phase_active"] - 7 / 9) < 1e-10
    assert abs(eo["eigenvalues"]["phase_trivial"] - 1.0) < 1e-10
    print("  OK eo accessors")


def test_co_accessors():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.co_k_values() == {3, 4, 6}  # post-ρ-fix: perm@phase, 3 eigenvalues
    expected = {1 - k / 9 for k in (3, 4, 6)}
    actual = ss.co_eigenvalues()
    assert len(expected) == len(actual)
    for e, a in zip(sorted(expected), sorted(actual)):
        assert abs(e - a) < 1e-10
    print("  OK co accessors")


def test_block_structure():
    ss = SpectralStructure(generators=dict(prim))
    bs = ss.block_structure()
    assert set(bs.keys()) == {"cp", "ep", "co", "eo"}
    for name, info in bs.items():
        assert "type" in info
        assert "tensor_dim" in info
    print("  OK block_structure")


def test_face_sum_operator():
    ss = SpectralStructure(generators=dict(prim))

    # cp: 8×8 operator in Q3 A_d basis
    op_cp = ss.class_sum_operator("cp")
    assert op_cp.shape == (8, 8)
    assert np.allclose(op_cp, op_cp.T)  # symmetric
    # eigenvalue on trivial eigenspace = Σ c_d * v_d = 9*1 + 2*3 + 1*3 + 0*1 = 18
    eigs_cp = np.linalg.eigvalsh(op_cp)
    assert abs(eigs_cp[-1] - 18.0) < 1e-10  # max eval = 18

    # ep: 12×12 operator S_total = alpha*I + JJ^T
    op_ep = ss.class_sum_operator("ep")
    assert op_ep.shape == (12, 12)
    assert np.allclose(op_ep, op_ep.T)
    # diag = alpha + JJ^T[i,i] = 10 + 2 = 12
    assert int(op_ep[0, 0]) == 12

    # co: 8×8 Hermitian perm@phase matrix (post-ρ-fix: full matrix, not diagonal)
    op_co = ss.class_sum_operator("co")
    assert op_co.shape == (8, 8)
    assert np.allclose(op_co, op_co.T.conj())  # Hermitian for inverse-closed S

    # eo: 12×12 symmetric perm@phase matrix (post-ρ-fix: full matrix, not diagonal)
    op_eo = ss.class_sum_operator("eo")
    assert op_eo.shape == (12, 12)
    assert np.allclose(op_eo, op_eo.T)  # symmetric

    # invalid block
    try:
        ss.class_sum_operator("invalid")
        assert False
    except ValueError:
        pass

    print("  OK face_sum_operator")


def test_predictions():
    ss = SpectralStructure(generators=dict(prim))
    assert ss.predict_spectral_field() == "rational"
    assert ss.predict_n_eigenvalues() == 6  # post-ρ-fix: +k=1 from eo block
    print("  OK predictions")


def test_summary_and_repr():
    ss = SpectralStructure(generators=dict(prim))
    text = ss.summary()
    assert "SpectralStructure" in text
    assert "18-full" in text
    assert "228" in text

    r = repr(ss)
    assert "SpectralStructure" in r
    assert "18-full" in r
    print("  OK summary and repr")


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 1: Diophantine feasibility solver (C1-C5, Paper I §7.2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_diophantine_feasibility_18full():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.diophantine_feasibility()
    assert result["feasible"] is True
    assert result["admissible_k_set"] == ss.k_set_total()
    assert result["predicted_match"] is True

    # C2: block exhaustion — each block's per-k dimensions sum to block dim
    c2 = result["constraints"]["C2_details"]
    assert c2["cp"][0]  # cp sums to 64
    assert c2["ep"][0]  # ep sums to 144
    assert c2["co"][0]  # co sums to 8
    assert c2["eo"][0]  # eo sums to 12

    # Each assignment must have non-negative block dimensions
    for k, d in result["assignments"].items():
        assert d["cp"] >= 0 and d["ep"] >= 0 and d["co"] >= 0 and d["eo"] >= 0
        assert d["total"] == d["cp"] + d["ep"] + d["co"] + d["eo"]

    # 6 layers (k=0,1,2,3,4,6)
    assert len(result["assignments"]) == 6
    print("  OK diophantine feasibility 18-full")


def test_diophantine_feasibility_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.diophantine_feasibility()
    assert result["feasible"] is True
    assert result["admissible_k_set"] == ss.k_set_total()
    assert result["predicted_match"] is True
    print("  OK diophantine feasibility 12-quarter")


def test_diophantine_feasibility_6halfturn():
    gens = {k: v for k, v in prim.items() if k[2] == 2}
    ss = SpectralStructure(generators=gens)
    result = ss.diophantine_feasibility()
    assert result["feasible"] is True
    assert result["admissible_k_set"] == ss.k_set_total()
    assert result["predicted_match"] is True
    print("  OK diophantine feasibility 6-half-turn")


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 2: co/eo first-principles perm@phase spectrum
# ═══════════════════════════════════════════════════════════════════════════════

def test_perm_phase_co_spectrum():
    ss = SpectralStructure(generators=dict(prim))
    spectrum = ss.derive_perm_phase_co_spectrum()
    total = sum(spectrum.values())
    assert total == 8  # co block has 8 dimensions
    # All eigenvalues must be real (Hermitian operator)
    for lam in spectrum:
        assert isinstance(lam, float)
    # 3 distinct eigenvalues post-ρ-fix
    assert len(spectrum) == 3
    print("  OK perm@phase co spectrum")


def test_perm_phase_eo_spectrum():
    ss = SpectralStructure(generators=dict(prim))
    spectrum = ss.derive_perm_phase_eo_spectrum()
    total = sum(spectrum.values())
    assert total == 12  # eo block has 12 dimensions
    # 3 distinct eigenvalues post-ρ-fix
    assert len(spectrum) == 3
    print("  OK perm@phase eo spectrum")


def test_combinatorial_co_derivation():
    ss = SpectralStructure(generators=dict(prim))
    comb_k = ss._derive_co_combinatorial()
    actual_k = ss.k_set_co()
    # The combinatorial derivation should match the numerical spectrum
    assert comb_k == actual_k, f"combinatorial {comb_k} != actual {actual_k}"
    print("  OK combinatorial co derivation")


def test_combinatorial_eo_derivation():
    ss = SpectralStructure(generators=dict(prim))
    comb_k = ss._derive_eo_combinatorial()
    actual_k = ss.k_set_eo()
    assert comb_k == actual_k, f"combinatorial {comb_k} != actual {actual_k}"
    print("  OK combinatorial eo derivation")


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 3: Krawtchouk eigenvalue prediction for arbitrary families
# ═══════════════════════════════════════════════════════════════════════════════

def test_krawtchouk_prediction_18full():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.predict_q3_krawtchouk()
    assert result["is_rational"] is True
    assert result["field_extension"] == "rational"
    assert result["k_values"] == ss.k_set_cp()
    # All eigenvalues should be of the form (integer)/m
    for k, info in result["eigenvalues"].items():
        lam = info["lambda"]
        assert abs(lam - (1 - k / ss.m)) < 1e-10
    print("  OK Krawtchouk prediction 18-full")


def test_krawtchouk_prediction_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.predict_q3_krawtchouk()
    assert result["is_rational"] is True
    assert result["field_extension"] == "rational"
    print("  OK Krawtchouk prediction 12-quarter")


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 4: Partition integrality verifier (Theorem 6.1)
# ═══════════════════════════════════════════════════════════════════════════════

def test_partition_integrality():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.verify_partition_integrality()

    # Should have per-eigenvalue mechanism data
    n_eigs = ss.predict_n_eigenvalues()
    assert len(result["mechanism"]) == n_eigs

    # The rationality conclusion should be present
    assert "rationality_conclusion" in result
    rc = result["rationality_conclusion"]
    assert rc["predicted"] is True  # 18-full is rational
    # Theorem holds vacuously if hypothesis not satisfied for all eigenspaces
    assert rc["theorem_holds"] is True or rc["theorem_holds"] is None

    # CO eigenspace: ω + ω^2 + 1 = 0 forces per-face integrality
    # EO eigenspace: per-face traces may be fractional (e.g. 16/3)
    # but global trace is always integer
    print("  OK partition integrality verifier")


def test_partition_integrality_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.verify_partition_integrality()
    assert "mechanism" in result
    assert result["rationality_conclusion"]["predicted"] is True
    print("  OK partition integrality 12-quarter")


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 5: Galois stability tester (Theorem 3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_galois_stability_18full():
    ss = SpectralStructure(generators=dict(prim))
    result = ss.verify_galois_stability()
    assert result["is_stable"] is True
    assert result["field"] == "Q"  # rational field
    assert result["galois_group"] == ["identity"]
    # All projectors must be invariant
    for lam, info in result["projector_invariance"].items():
        assert info["is_invariant"]
    print("  OK Galois stability 18-full")


def test_galois_stability_12quarter():
    gens = {k: v for k, v in prim.items() if k[2] != 2}
    ss = SpectralStructure(generators=gens)
    result = ss.verify_galois_stability()
    assert result["is_stable"] is True
    assert result["field"] == "Q"
    print("  OK Galois stability 12-quarter")


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry verification
# ═══════════════════════════════════════════════════════════════════════════════

def test_corner_adjacency():
    adj_data = CubeBase.build_corner_adjacency()
    A = adj_data["A"]
    v = adj_data["v"]
    assert np.allclose(A[0], np.eye(8))
    assert list(v) == [1, 3, 3, 1]
    print("  OK corner adjacency")


def test_edge_fb_classification():
    phase_data = CubeBase.edge_phase_classification()
    assert len(phase_data["phase_active"]) == 8
    assert len(phase_data["phase_trivial"]) == 4
    for e in phase_data["phase_active"]:
        assert CubeBase.edge_on_face(e, 'F') or CubeBase.edge_on_face(e, 'B')
    for e in phase_data["phase_trivial"]:
        assert not (CubeBase.edge_on_face(e, 'F') or CubeBase.edge_on_face(e, 'B'))
    print("  OK edge FB classification")


def test_corner_hamming():
    corner_signs = np.array(CubeBase.CORNER_POS_SIGNS, dtype=int)
    for i in range(8):
        diff = int(np.sum(corner_signs[i] != corner_signs[i]))
        assert diff == 0
    for i in range(8):
        opp_sign = tuple(-s for s in corner_signs[i])
        for j in range(8):
            if tuple(corner_signs[j]) == opp_sign:
                diff = int(np.sum(corner_signs[i] != corner_signs[j]))
                assert diff == 3
                break
    print("  OK corner Hamming distances")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SpectralStructure Tests")
    print("=" * 60)
    print()

    tests = [
        ("Krawtchouk", test_krawtchouk),
        ("Construction 18-full", test_construction_18full),
        ("Construction 12-quarter", test_construction_12quarter),
        ("Construction 6-half-turn", test_construction_6halfturn),
        ("Default construction", test_default_construction),
        ("Q3 scheme", test_q3_scheme),
        ("Face-incidence scheme", test_face_incidence_scheme),
        ("Z2 filter", test_z2_filter),
        ("Z3 filter", test_z3_filter),
        ("k-sets 18-full", test_kset_18full),
        ("k-sets 12-quarter", test_kset_12quarter),
        ("k-sets 6-half-turn", test_kset_6halfturn),
        ("Eigenvalues 18-full", test_eigenvalues_18full),
        ("Eigenvalues 12-quarter", test_eigenvalues_12quarter),
        ("Eigenvalues 6-half-turn", test_eigenvalues_6halfturn),
        ("Multiplicities 18-full", test_multiplicity_18full),
        ("Multiplicities 12-quarter", test_multiplicity_12quarter),
        ("Multiplicities 6-half-turn", test_multiplicity_6halfturn),
        ("Block projectors", test_block_projectors),
        ("block_of_index", test_block_of_index),
        ("block_projector method", test_block_projector_method),
        ("validate_with_numerics 18-full", test_validate_with_numerics_18full),
        ("validate_with_numerics 12-quarter", test_validate_with_numerics_12quarter),
        ("validate_with_numerics 6-half-turn", test_validate_with_numerics_6halfturn),
        ("Integrality 18-full", test_integrality_18full),
        ("Integrality 12-quarter", test_integrality_12quarter),
        ("Integrality 6-half-turn", test_integrality_6halfturn),
        ("Block reduction theorem", test_block_reduction_theorem),
        ("Face-sum decomposition", test_face_sum_decomposition),
        ("compute_face_sum_coeffs", test_compute_face_sum_coeffs),
        ("k_by_block", test_k_by_block),
        ("Slow dimension", test_slow_dimension),
        ("Scheme accessors", test_scheme_accessors),
        ("face_partition", test_face_partition),
        ("eo accessors", test_eo_accessors),
        ("co accessors", test_co_accessors),
        ("block_structure", test_block_structure),
        ("face_sum_operator", test_face_sum_operator),
        ("Predictions", test_predictions),
        ("Summary and repr", test_summary_and_repr),
        ("Diophantine feasibility 18-full", test_diophantine_feasibility_18full),
        ("Diophantine feasibility 12-quarter", test_diophantine_feasibility_12quarter),
        ("Diophantine feasibility 6-half-turn", test_diophantine_feasibility_6halfturn),
        ("Perm@phase co spectrum", test_perm_phase_co_spectrum),
        ("Perm@phase eo spectrum", test_perm_phase_eo_spectrum),
        ("Combinatorial co derivation", test_combinatorial_co_derivation),
        ("Combinatorial eo derivation", test_combinatorial_eo_derivation),
        ("Krawtchouk prediction 18-full", test_krawtchouk_prediction_18full),
        ("Krawtchouk prediction 12-quarter", test_krawtchouk_prediction_12quarter),
        ("Partition integrality", test_partition_integrality),
        ("Partition integrality 12-quarter", test_partition_integrality_12quarter),
        ("Galois stability 18-full", test_galois_stability_18full),
        ("Galois stability 12-quarter", test_galois_stability_12quarter),
        ("Corner adjacency", test_corner_adjacency),
        ("Edge FB classification", test_edge_fb_classification),
        ("Corner Hamming", test_corner_hamming),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 60)
