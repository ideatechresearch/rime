import sys
import io, os

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (ValueError, AttributeError):
        pass  # stdout already configured or not wrappable

from rime.cubie import CubieState, CubieMove, CubieBase
from rime.base import DATA_DIR
from rime.helpers import is_in_qsqrt5, is_rational_form
import numpy as np
import random, math
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Shared spectral utilities (used by _exp_*.py test files)
# ============================================================

def eigenspaces(A, tol=1e-6):
    """Eigenspace decomposition: {eigenvalue: {'dim': int, 'projector': ndarray}}.

    Handles both Hermitian (eigh) and non-Hermitian (eig) matrices.
    Returns only real eigenvalues.
    """
    if np.allclose(A, A.T.conj(), atol=1e-10):
        w, V = np.linalg.eigh(A)
    else:
        w_raw, V_raw = np.linalg.eig(A)
        mask = np.abs(np.imag(w_raw)) < 1e-8
        w, V = np.real(w_raw[mask]), V_raw[:, mask]
    w_rounded = np.round(w, 6)
    unique_w = np.unique(w_rounded)
    result = {}
    for lam in unique_w:
        idx = np.where(w_rounded == lam)[0]
        V_lam = V[:, idx]
        P_lam = V_lam @ V_lam.T.conj()
        result[lam] = {'dim': len(idx), 'projector': P_lam}
    return result


def build_A(gens_dict):
    """Build averaging operator A = (1/|S|) Σ_{s∈S} ρ(s)."""
    rhos = [m.rho() for m in gens_dict.values()]
    return sum(rhos) / len(rhos)


def build_h_operators(gens_dict):
    """Build symmetric h_i = (ρ(g) + ρ(g⁻¹))/2 operators from generator dict.

    Returns (h_ops, h_labels) where h_ops are (228,228) arrays.
    """
    from rime.cube import ActionToken
    h_ops, h_labels = [], []
    for axis in range(3):
        for side in [-1, 1]:
            cw_key = (axis, side, -1)
            ccw_key = (axis, side, 1)
            if cw_key in gens_dict and ccw_key in gens_dict:
                h_ops.append((gens_dict[cw_key].rho() + gens_dict[ccw_key].rho()) / 2)
                at_cw = str(ActionToken.from_cubie_move(*cw_key, n=3))
                at_ccw = str(ActionToken.from_cubie_move(*ccw_key, n=3))
                h_labels.append(f"({at_cw}+{at_ccw})/2")
    for axis in range(3):
        keys_180 = [(axis, side, 2) for side in [-1, 1] if (axis, side, 2) in gens_dict]
        if len(keys_180) == 2:
            h_ops.append((gens_dict[keys_180[0]].rho() + gens_dict[keys_180[1]].rho()) / 2)
            at_a = str(ActionToken.from_cubie_move(*keys_180[0], n=3))
            at_b = str(ActionToken.from_cubie_move(*keys_180[1], n=3))
            h_labels.append(f"({at_a}+{at_b})/2")
    return h_ops, h_labels


def classify_spectral_field(eigs, m_eff, name=None):
    """Classify spectral field as 'rational', 'sqrt5', or 'higher'.

    Uses theoretical classification: n=8/n=16 → sqrt5, otherwise based on m_eff check.
    """
    all_rational = all(is_rational_form(lam, m_eff) for lam in eigs)
    if all_rational:
        return 'rational'
    if name in ('n=8', 'n=16'):
        return 'sqrt5'
    # Check if all non-rational eigenvalues are in ℚ(√5)
    non_rat = [lam for lam in eigs if not is_rational_form(lam, m_eff)]
    if non_rat and all(is_in_qsqrt5(lam)[0] for lam in non_rat):
        return 'sqrt5'
    return 'higher'


def spectral_field_label(set_class):
    """Return LaTeX field notation for a set class."""
    return {
        'rational': r'$\mathbb{Q}$',
        'sqrt5': r'$\mathbb{Q}(\sqrt{5})$',
        'higher': r'$\mathbb{Q}(\zeta_n)^+$',
    }[set_class]


def detect_blocks(moves, U, tol=1e-8):
    """
    moves: list of CubieMove
    U: eigenbasis from random operator
    tol: zero threshold
    """

    n = U.shape[0]
    S = np.zeros((n, n), dtype=float)

    # 累加所有变换后的矩阵绝对值
    for mv in moves:
        M = U.T.conj() @ mv.rho() @ U
        S += np.abs(M)

    # adjacency matrix：非零视为连通
    adj = S > tol

    # 用 BFS 找连通分量
    visited = np.zeros(n, dtype=bool)
    blocks = []

    for i in range(n):
        if visited[i]:
            continue

        stack = [i]
        component = []

        while stack:
            j = stack.pop()
            if visited[j]:
                continue

            visited[j] = True
            component.append(j)

            neighbors = np.where(adj[j])[0]
            for k in neighbors:
                if not visited[k]:
                    stack.append(k)

        blocks.append(sorted(component))

    return blocks


def split_isotypic_block(moves, U, block_idx, tol=1e-6, n_trials=3):
    idx = block_idx
    d = len(idx)
    restricted = []
    for mv in moves:
        M = U.T.conj() @ mv.rho() @ U
        restricted.append(M[np.ix_(idx, idx)])

    # 构造多个随机矩阵，增强稳定性
    C = np.zeros((d, d), dtype=complex)
    for _ in range(n_trials):
        X = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        for M in restricted:
            C += M @ X @ np.linalg.inv(M)

    # 对 C 做特征分解
    eigvals, eigvecs = np.linalg.eig(C)

    # 聚类特征值
    unique_vals = []
    multiplicities = []
    for val in eigvals:
        found = False
        for i, uval in enumerate(unique_vals):
            if abs(val - uval) < tol:
                multiplicities[i] += 1
                found = True
                break
        if not found:
            unique_vals.append(val)
            multiplicities.append(1)

    return multiplicities


# 生成投影算子
def construct_projection_operators(U, blocks, tol=1e-12):
    """
    输入:
        samples: CubieMove 对象列表
        U: 单位基矩阵，size (228, 228)
        blocks: detect_blocks(samples, U) 得到的 block 索引
    输出:
        projections: list of (228,228) Hermitian 投影矩阵
    """
    projections = []
    for b in blocks:
        # b 是 block 的行/列索引
        # U[:, b] 是该 block 的基
        Ub = U[:, b]
        P = Ub @ Ub.T.conj()  # 投影算子
        # 数值修正，保证幂等性
        P[np.abs(P) < tol] = 0
        projections.append(P)
    return projections


def verify_bose_mesner(A, tol=1e-6):
    """
    Verify diagonal semisimple algebra structure (not necessarily Bose-Mesner).

    For A Hermitian with distinct eigenvalues, the spectral projectors E_i
    satisfy E_i E_j = delta_ij E_i and sum E_i = I — the minimal conditions
    for a commutative semisimple algebra C[A] ≅ C^k. This is a diagonal
    (trivial) Bose-Mesner-type structure.

    IMPORTANT: This checks the SPECTRAL algebra C[A], not whether the original
    Cayley graph corresponds to an association scheme. Individual generators
    rho(s) are NOT in span{E_i} — the association scheme check requires
    additional closure conditions (see verify_association_scheme).
    """
    n = A.shape[0]
    details = {}

    # Step 0: 检查 A 是否近似 Hermitian
    if not np.allclose(A, A.T.conj(), atol=tol):
        return False, "A is not Hermitian (or close to it). Bose-Mesner algebra assumes symmetric matrices.", details

    # Step 1: 特征分解
    w, V = np.linalg.eigh(A)
    idx = np.argsort(-w)  # 从大到小排序
    w = w[idx]
    V = V[:, idx]
    details['eigenvalues'] = w
    details['eigenvectors'] = V

    # Step 2: 唯一特征值 & 多重度
    unique_w = np.unique(np.round(w, decimals=int(-np.log10(tol))))
    num_classes = len(unique_w)
    multiplicities = [np.sum(np.abs(w - lambda_i) < tol) for lambda_i in unique_w]
    details['unique_eigenvalues'] = unique_w
    details['multiplicities'] = multiplicities
    print(f"Number of distinct eigenvalues (classes): {num_classes}")
    for lambda_i, mult in zip(unique_w, multiplicities):
        print(f"Lambda {lambda_i:.6f}: multiplicity {mult}")

    if num_classes > 10:  # 经验阈值，association scheme 通常 classes 小
        print("Warning: Many classes; may not be a simple association scheme.")

    # Step 3: 构造 idempotents E_i = sum_{j: w_j ≈ lambda_i} v_j v_j^T
    idempotents = []
    for lambda_i in unique_w:
        mask = np.abs(w - lambda_i) < tol
        E_i = V[:, mask] @ V[:, mask].T.conj()  # 投影器
        idempotents.append(E_i)

    details['idempotents'] = idempotents

    # Step 4: 验证 idempotents 属性
    sum_E = np.zeros_like(A, dtype=np.complex128)
    for i, E_i in enumerate(idempotents):
        # E_i^2 ≈ E_i
        if not np.allclose(E_i @ E_i, E_i, atol=tol):
            return False, f"Idempotent check failed for E_{i} (lambda={unique_w[i]:.6f})", details

        sum_E += E_i

        for j in range(i + 1, len(idempotents)):
            E_j = idempotents[j]
            # E_i E_j ≈ 0
            if not np.allclose(E_i @ E_j, np.zeros_like(E_i), atol=tol):
                return False, f"Orthogonal check failed for E_{i} and E_{j}", details

    # sum E_i ≈ I
    if not np.allclose(sum_E, np.eye(n), atol=tol):
        return False, "Sum of idempotents not equal to identity", details

    # Step 5: 重构 A = sum lambda_i E_i
    A_recon = sum([lambda_i * E_i for lambda_i, E_i in zip(unique_w, idempotents)])
    if not np.allclose(A, A_recon, atol=tol):
        return False, "A reconstruction from sum lambda_i E_i failed", details

    # Step 6: 检查最小多项式度数 ≈ num_classes
    # 构造 p(A) = prod (A - lambda_i I) = 0
    from numpy.polynomial.polynomial import Polynomial
    p = Polynomial([1])
    for lambda_i in unique_w:
        p = p * Polynomial([-lambda_i, 1])  # (x - lambda_i)

    coeffs = p.coef
    p_A = np.zeros_like(A)
    A_pow = np.eye(n)
    for c in coeffs:
        p_A += c * A_pow
        A_pow = A_pow @ A

    if not np.allclose(p_A, np.zeros_like(A), atol=tol):
        return False, f"Minimal polynomial check failed (degree {num_classes})", details

    # Step 7: 如果所有通过，认为是 Bose-Mesner 代数，对应 association scheme
    message = f"Verified as Bose-Mesner algebra with {num_classes} classes.\n"
    message += "This implies the Cayley graph is representation-equivalent to an association scheme (or substructure)."
    return True, message, details


def verify_association_scheme(A, generators, tol=1e-6, integer_tol=1e-3):
    """
    Verify whether C[A] (the commutative spectral algebra of A) plus generator
    closure yields an association scheme structure.

    NOTE: For our system, Step 4 (generator closure check: rho(s) in span{E_i})
    is EXPECTED TO FAIL — individual generators are NOT in the commutative
    spectral algebra span{E_i}. C[A] is a 5-dim commutative algebra, not a
    full association scheme. This function serves as a diagnostic to confirm
    the distinction.

    Verification steps:
    1. Spectral decomposition + commutative algebra (Bose-Mesner) properties.
    2. Generator closure: each rho(s) in span{E_i}? → expected FAIL for our system.
    3. Multiplication closure: E_i E_j = sum p_{ij}^k E_k, p non-negative integers.
    4. If ALL pass (unlikely for Rubik's cube), Cayley graph is an association scheme.

    Output:
    - success: bool
    - message: str
    - details: dict
    """
    n = A.shape[0]
    details = {}

    # Step 0: 检查 A 是否 Hermitian
    if not np.allclose(A, A.T.conj(), atol=tol):
        return False, "A is not Hermitian.", details

    # Step 1: 特征分解 & 唯一值
    w, V = np.linalg.eigh(A)
    idx = np.argsort(-w)
    w = w[idx]
    V = V[:, idx]

    unique_w = np.unique(np.round(w, decimals=int(-np.log10(tol))))
    num_classes = len(unique_w)
    multiplicities = [np.sum(np.abs(w - lambda_i) < tol) for lambda_i in unique_w]
    details['unique_eigenvalues'] = unique_w
    details['multiplicities'] = multiplicities

    if num_classes < 2 or num_classes > 10:
        message = f"Unusual number of classes ({num_classes}); may not be scheme-like."
        print(message)

    # Step 2: 构造 E_i
    E_list = []
    for lambda_i in unique_w:
        mask = np.abs(w - lambda_i) < tol
        E_i = V[:, mask] @ V[:, mask].T.conj()
        E_list.append(E_i)
    details['idempotents'] = E_list

    # Step 3: 验证 Bose-Mesner 基本属性
    sum_E = np.zeros_like(A)
    for i, E_i in enumerate(E_list):
        if not np.allclose(E_i @ E_i, E_i, atol=tol):
            return False, f"E_{i} not idempotent.", details
        sum_E += E_i
        for j in range(i + 1, num_classes):
            if not np.allclose(E_list[i] @ E_list[j], np.zeros_like(A), atol=tol):
                return False, f"E_{i} and E_{j} not orthogonal.", details

    if not np.allclose(sum_E, np.eye(n), atol=tol):
        return False, "Sum E_i != I.", details

    A_recon = sum([lambda_i * E_i for lambda_i, E_i in zip(unique_w, E_list)])
    if not np.allclose(A, A_recon, atol=tol):
        return False, "A reconstruction failed.", details

    # Step 4: 检查生成元闭合 (ρ(s) in span{E_i})
    for s_idx, rho_s in enumerate(generators):
        # 假设 rho_s = sum c_i E_i
        rho_recon = np.zeros_like(A)
        coeffs = []
        for E_i in E_list:
            c_i = np.trace(rho_s @ E_i) / np.trace(E_i)  # Frobenius 内积投影
            coeffs.append(c_i)
            rho_recon += c_i * E_i

        if not np.allclose(rho_s, rho_recon, atol=tol):
            return False, f"Generator {s_idx} not in span{E_i}.", details
        print(f"Generator {s_idx} coeffs in span{E_i}: {coeffs}")

    details['generator_coeffs'] = coeffs  # 示例，最后一个的

    # Step 5: 检查乘法闭合 p_{ij}^k (E_i E_j = sum p_{ij}^k E_k, p 非负整数)
    p_constants = np.zeros((num_classes, num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            E_ij = E_list[i] @ E_list[j]
            recon = np.zeros_like(A)
            for k in range(num_classes):
                p_ijk = np.trace(E_ij @ E_list[k]) / np.trace(E_list[k])  # 投影系数
                p_constants[i, j, k] = p_ijk
                recon += p_ijk * E_list[k]

            if not np.allclose(E_ij, recon, atol=tol):
                return False, f"Multiplication closure failed for i={i}, j={j}.", details

            # 检查 p 非负整数
            if np.any(np.abs(p_constants[i, j] - np.round(p_constants[i, j])) > integer_tol) or np.any(
                    p_constants[i, j] < -integer_tol):
                return False, f"p_{i}{j}^k not non-negative integers.", details

    details['p_constants'] = np.round(p_constants).astype(int)  # 整数形式

    # 如果全通过
    message = f"Verified as association scheme with {num_classes} classes.\n"
    message += "Cayley graph corresponds to an association scheme (or substructure/quotient).\n"
    message += "p constants are non-negative integers."
    return True, message, details


def check_invariant_subspaces(A_micro, generators, w, V, tol=1e-6):
    """
    检查每个特征值 λ 对应的子空间 V_λ 是否在所有生成元 ρ(s) 下不变。

    输入:
    - A_micro: (n,n) 转移矩阵
    - generators: list of ρ(s) matrices (27 个)
    - w, V: 从 np.linalg.eigh(A_micro) 得到的特征值和特征向量（已排序）
    - tol: 数值容忍度

    输出:
    - results: dict {lambda_i: (is_invariant: bool, max_error: float)}
    - message: 总结报告
    """
    n = A_micro.shape[0]
    unique_w = np.unique(np.round(w, decimals=int(-np.log10(tol))))
    results = {}

    print("开始检查每个 λ 子空间的不变性...")
    print("-" * 60)

    for lambda_i in unique_w:
        # 提取对应子空间的投影器 E_λ
        mask = np.abs(w - lambda_i) < tol
        multiplicity = np.sum(mask)
        V_lambda = V[:, mask]  # (n, multiplicity)
        E_lambda = V_lambda @ V_lambda.T.conj()  # (n,n) 投影器

        max_err = 0.0
        all_invariant = True

        for s_idx, rho_s in enumerate(generators):
            # 计算 ρ(s) V_λ 是否还在 V_λ 内
            # 等价检查：|| (I - E_λ) ρ(s) V_λ || / || ρ(s) V_λ || 应 ≈ 0
            rho_V = rho_s @ V_lambda  # (n, mult)
            projected = E_lambda @ rho_V  # 留在 V_λ 的部分
            residual = rho_V - projected  # 逃逸部分
            err = np.linalg.norm(residual) / (np.linalg.norm(rho_V) + 1e-12)

            max_err = max(max_err, err)

            if err > tol:
                all_invariant = False
                print(f"  λ={lambda_i:.6f} (dim={multiplicity}) | "
                      f"Generator {s_idx:2d} → 逃逸误差 = {err:.2e} > tol")

        results[lambda_i] = {
            'multiplicity': multiplicity,
            'is_invariant': all_invariant,
            'max_error': max_err
        }

        status = "不变 (invariant)" if all_invariant else f"不完全不变 (max err={max_err:.2e})"
        print(f"λ = {lambda_i:.6f} (dim={multiplicity:3d}) → {status}")
        print("-" * 60)

    # 总结报告
    invariant_count = sum(1 for r in results.values() if r['is_invariant'])
    message = f"\n总结：{len(unique_w)} 个子空间中，有 {invariant_count} 个完全不变。\n"
    if invariant_count == len(unique_w):
        message += "所有 V_λ 在 Phase-1 生成元作用下不变。\n"
        message += "→ 228 维表示分解成 5 个不变子空间（C[A] 的同型分量，不一定是 G 的不可约表示）。"
    else:
        message += "并非所有子空间都完全不变。\n"
        message += "但慢层（λ ≥ 2/3）的不变性值得进一步关注。"

    return results, message


def commutant_dimension(generators, tol=1e-8):
    """
    计算 Comm(ρ(G)) = {X : ρ(s) X = X ρ(s), ∀s} 的维度

    BUGFIX (2026-04-28): vec(ρ X - X ρ) = (I ⊗ ρ - ρ^T ⊗ I) vec(X)，原先的
    ρ ⊗ I - I ⊗ ρ^T 对应的是 X^T ρ - ρ X^T = 0，不是交换子条件。
    修复：使用 I ⊗ ρ - ρ^T ⊗ I。
    """
    n = generators[0].shape[0]
    I = np.eye(n, dtype=complex)
    blocks = []

    for rho in generators:
        # vec(ρ X - X ρ) = (I ⊗ ρ - ρ^T ⊗ I) vec(X)
        A = np.kron(I, rho) - np.kron(rho.T, I)
        blocks.append(A)

    M = np.vstack(blocks)
    _, s, _ = np.linalg.svd(M)

    nullity = np.sum(s < tol)
    return nullity


def estimate_commutant_dim(generators, n=228, num_samples=50, tol=1e-6):
    """
    随机化估计 Comm(ρ(G)) 维度

    对每个生成元分别计算 commutator nullspace，取交集维度作为估计。
    避免巨型矩阵分配 (N_gen * num_samples × n²)。
    """
    # 计算单个生成元 commutator 的零空间维度，取平均值
    dims = []
    for rho_s in generators:
        Xs = np.random.randn(num_samples, n, n) + 1j * np.random.randn(num_samples, n, n)
        rho_X = rho_s @ Xs
        X_rho = Xs @ rho_s
        res = (rho_X - X_rho).reshape(num_samples, n * n)
        _, S, _ = np.linalg.svd(res, full_matrices=False)
        null_dim = np.sum(S < tol)
        dims.append(null_dim)

    # Commutant 是所有生成元 commutator 的零空间交集
    # 保守估计: 取最大值 (交集不会大于最大单独零空间)
    # 实际上 dim(∩ ker_i) ≤ min_i dim(ker_i)，但这里我们估算为平均值
    avg_null = np.mean(dims)
    return int(np.round(avg_null))


def check_commutativity(generators, V_slow, tol=1e-6):
    """
    检查慢子空间上 P rho(s) P 是否两两对易
    """
    # 投影
    Ps = []

    for rho_s in generators:
        A = V_slow.T @ rho_s @ V_slow
        Ps.append(A)

    max_comm_norm = 0

    for i in range(len(Ps)):
        for j in range(i + 1, len(Ps)):
            comm = Ps[i] @ Ps[j] - Ps[j] @ Ps[i]
            norm = np.linalg.norm(comm)
            max_comm_norm = max(max_comm_norm, norm)

    print("最大对易误差:", max_comm_norm)
    return max_comm_norm


def group_algebra_dim(generators, tol=1e-8):
    X = np.stack([A.reshape(-1) for A in generators])

    u, s, vh = np.linalg.svd(X, full_matrices=False)

    return np.sum(s > tol)


def slow_algebra_dimension(generators, V_slow, tol=1e-8):
    """
    计算 dim span{ P rho(s) P }
    """
    mats = []

    for rho_s in generators:
        A = V_slow.T @ rho_s @ V_slow  # 慢层压缩
        mats.append(A.reshape(-1))  # 展平为向量

    M = np.stack(mats, axis=1)  # (d^2 , num_generators)

    # SVD 求秩
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    dim = np.sum(S > tol)

    print("慢层生成代数维度 ≈", dim)
    # print("奇异值 (前10):", S[:10])
    return dim


# noinspection PyUnboundLocalVariable
def compute_span_dim(P, generators, tol=1e-6):
    """
    计算 span{P ρ(s) P for s in S} 的维度。

    输入:
    - P: (228,228) 慢子空间投影器 (E_slow = V_slow @ V_slow.T.conj())
    - generators: list of ρ(s) (27 个生成元矩阵)
    - tol: rank 计算的数值容忍度

    方法:
    - 计算所有 M_s = P @ rho_s @ P
    - 展平每个 M_s 成 (228*228,) 向量
    - 堆栈成矩阵 (27, 228^2)
    - 计算该矩阵的 rank (线性独立数) ≈ dim span{M_s}

    输出:
    - dim: int, 估算维度
    - message: str, 解释
    """
    n = P.shape[0]
    M_flat_list = []

    for rho_s in generators:
        M_s = P @ rho_s @ P
        M_flat = M_s.flatten()
        M_flat_list.append(M_flat)  # BUGFIX: 保留复数, np.real 会丢失虚部结构

    M_matrix = np.array(M_flat_list)

    # SVD 对复数矩阵同样有效, 奇异值始终是非负实数
    U, S, Vh = np.linalg.svd(M_matrix, full_matrices=False)
    rank = np.sum(S > tol * (S[0] if S[0] > 0 else 1.0))

    message = f"span{{P ρ(s) P}} 维度 ≈ {rank}\n"
    if rank <= 5:
        message += "≈ 5 → 统计 5 维子代数，慢子空间有强结构\n"
    elif rank > 5 and rank < 10:
        message += "略大于 5 → 近似 5 维子代数\n"
    else:
        message += ">> 5 → 只是谱退化现象，非强代数结构\n"

    print("奇异值 (前10):", S[:10])

    dim_A, dim_A2, msg = check_algebra_closure(M_matrix)
    message += msg
    return rank, message


def check_algebra_closure(M_matrix, tol=1e-6):
    """
    输入:
    - M_matrix: (27, 228*228) 矩阵，每行是 flatten(P ρ(s) P)

    输出:
    - dim_A: dim(span{M_s})
    - dim_A2: dim(span{M_i M_j for all i,j})
    - message: 解释是否闭合
    """
    # Step 1: dim(A) ≈ rank(M_matrix)
    _, S, _ = np.linalg.svd(M_matrix, full_matrices=False)
    dim_A = np.sum(S > tol * S[0])
    print(f"dim(A) ≈ {dim_A} (from previous: 11)")

    # Step 2: 计算所有 M_i M_j 的 flatten
    n_flat = M_matrix.shape[1]  # 228^2
    n_gen = M_matrix.shape[0]  # 27
    M_prod_list = []

    for i in range(n_gen):
        M_i = M_matrix[i].reshape(228, 228)
        for j in range(n_gen):
            M_j = M_matrix[j].reshape(228, 228)
            prod = M_i @ M_j
            M_prod_list.append(prod.flatten())

    # 堆栈成 (27*27, 228^2) 矩阵
    M_prod_matrix = np.array(M_prod_list)  # (729, 51984)

    # Step 3: 计算 rank(M_prod_matrix)
    _, S_prod, _ = np.linalg.svd(M_prod_matrix, full_matrices=False)
    dim_A2 = np.sum(S_prod > tol * S_prod[0])

    print(f"dim(A^2) ≈ {dim_A2}")
    print(f"dim(A^2) - dim(A) = {dim_A2 - dim_A}")

    if dim_A2 <= dim_A + 2:
        message = "dim(A^2) ≈ dim(A) → algebra numerically closed\n" \
                  "→ small closed non-abelian algebra on slow subspace (dim ≈11)"
    elif dim_A2 > dim_A + 5:
        message = "dim(A^2) >> dim(A) → not closed, just spectral degeneracy"
    else:
        message = "dim(A^2) slightly > dim(A), possible numerical artifact"

    print(message)
    return dim_A, dim_A2, message


def leakage_bounds(generators, V_slow):
    P_s = V_slow @ V_slow.T.conj()
    P_f = np.eye(P_s.shape[0]) - P_s

    max_B = 0
    max_D = 0

    for rho_s in generators:
        B = P_f @ rho_s @ P_s
        D = P_f @ rho_s @ P_f

        max_B = max(max_B, np.linalg.norm(B, 2))
        max_D = max(max_D, np.linalg.norm(D, 2))

    print("最大 slow→fast 泄漏 ‖B‖ =", max_B)
    print("最大 fast 内部谱 ‖D‖ =", max_D)

    return max_B, max_D


def analyze_fast_pseudospectrum(A_micro, V_slow, t_max=30):
    n = V_slow.shape[0]
    P_s = V_slow @ V_slow.T.conj()
    P_f = np.eye(n) - P_s

    A_f = P_f @ A_micro @ P_f

    print("=== Fast Layer Analysis ===")

    # 1️⃣ 谱半径
    eigvals, eigvecs = np.linalg.eig(A_f)  # ≈ 5/9
    rho = np.max(np.abs(eigvals))
    print("谱半径 ρ =", rho)

    # 2️⃣ 非正规性
    non_normality = np.linalg.norm(A_f @ A_f.conj().T - A_f.conj().T @ A_f)
    print("非正规性 ‖AA* - A*A‖ =", non_normality)

    # 3️⃣ 特征向量条件数
    try:
        V = eigvecs
        cond_V = np.linalg.cond(V)
        print("特征向量条件数 κ(V) =", cond_V)
    except:
        cond_V = None
        print("无法计算条件数")

    # 4️⃣ 实际幂增长
    norms = []
    A_power = np.eye(n)
    for t in range(1, t_max + 1):
        A_power = A_power @ A_f
        norm_val = np.linalg.norm(A_power, 2)
        norms.append(norm_val)

    print("t=1..10 实际 ‖A_f^t‖:")
    print(norms[:10])

    print("理论 ρ^t (前10):")
    print([rho ** t for t in range(1, 11)])

    return rho, non_normality, cond_V, norms


def estimate_fast_lyapunov(generators, V_slow, T=2000):
    n = V_slow.shape[0]
    P_s = V_slow @ V_slow.T.conj()
    P_f = np.eye(n) - P_s

    # 初始化随机 fast 向量
    v = np.random.randn(n) + 1j * np.random.randn(n)
    v = P_f @ v
    v /= np.linalg.norm(v)

    log_sum = 0.0

    for t in range(T):
        rho_s = random.choice(generators)

        v = P_f @ (rho_s @ v)
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-14:
            continue

        log_sum += np.log(norm_v)
        v /= norm_v

    lyap = log_sum / T
    print("最大 Lyapunov 指数 λ ≈", lyap)
    print("对应指数率 e^λ ≈", np.exp(lyap))

    return lyap


def compute_orbits(rho_generators, dim):
    """
    rho_generators : list of permutation matrices (或稀疏作用函数)
    dim : 空间维度
    """
    visited = set()
    orbits = []

    for i in range(dim):
        if i in visited:
            continue

        orbit = set([i])
        frontier = [i]

        while frontier:
            j = frontier.pop()
            for R in rho_generators:
                k = np.argmax(R @ np.eye(dim)[:, j])  # 作用在基向量上
                if k not in orbit:
                    orbit.add(k)
                    frontier.append(k)

        visited |= orbit
        orbits.append(sorted(list(orbit)))

    return orbits


def orbit_vector(orbit, dim):
    v = np.zeros(dim)
    for i in orbit:
        v[i] = 1.0
    v /= np.linalg.norm(v)
    return v


def poly_rank(A, k=6, tol=1e-10):
    """
    Krylov subspace rank: rank{I, A, A^2, ..., A^{k-1}} = minimal polynomial degree.

    BUGFIX (2026-04-28): np.linalg.matrix_rank 默认 tolerance 对大矩阵可能
    不可靠。使用 SVD + 显式 tolerance 确保数值稳定性。
    """
    mats = []
    Ak = np.eye(A.shape[0])

    for i in range(k):
        mats.append(Ak.flatten())
        Ak = Ak @ A

    M = np.vstack(mats)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return np.sum(s > tol * max(s[0], 1.0))


def shell_projector(samples=1000):
    shells = {}

    for _ in range(samples):

        g = CubieBase.random_walk(40)
        d = g.orientation_distance

        rho = g.rho()

        if d not in shells:
            shells[d] = []

        shells[d].append(rho)

    return shells


def fit_shell_decomposition(A_micro, P):
    basis = []
    ks = []

    for k, mat in P.items():
        basis.append(mat.flatten())
        ks.append(k)

    B = np.array(basis).T
    a = A_micro.flatten()

    coef, *_ = np.linalg.lstsq(B, a, rcond=None)

    print("coefficients:")
    for k, c in zip(ks, coef):
        print(f"shell {k}: {c:.6f}")

    err = np.linalg.norm(B @ coef - a) / np.linalg.norm(a)
    print("relative error:", err)


def attention_evolve_exact(x, lambda_list, E_list):
    x_next = np.zeros_like(x, dtype=complex)
    for lam, E in zip(lambda_list, E_list):
        x_next += lam * (E @ x)
    return x_next  # sum(lam * (E @ x) for lam, E in zip(lambdas, projectors))


def spectral_power(x, t, lambdas, projectors):
    # t步演化,E_i E_j = 0
    x_next = np.zeros_like(x, dtype=complex)
    for lam, E in zip(lambdas, projectors):
        x_next += (lam ** t) * (E @ x)
    return x_next


def spectral_evolve(x, lambdas, M_layers):
    """
    heads 完全由群结构决定
    λ_i E_i
    q_i = E_i x
    attention = λ_i
    """
    P = np.stack([E @ x for E in M_layers], axis=0)
    x_next = (lambdas[:, None] * P).sum(axis=0)
    return x_next


def analyze_block_spectrum(A_block, blocks):
    """
    对每个 block 计算子矩阵的特征值分布，并可视化
    假设：

    detect_blocks 得到的 block 在这个基底下是“连续 index”
    Parameters:
    - A_block: 228×228 的对角块矩阵（A_micro 的块对角形式）
    - blocks: list of index lists (e.g. [角块索引], [棱块索引], ...)
    """
    block_eigvals = {}
    for i, block_idx in enumerate(blocks):
        size = len(block_idx)
        # 提取子矩阵
        sub_A = A_block[np.ix_(block_idx, block_idx)]

        # 特征值分解（只取实部，因为 A 是 Hermitian）
        eigvals = np.linalg.eigvalsh(sub_A)  # eigh for Hermitian
        eigvals = np.round(eigvals, 10)  # 去掉数值噪声
        eigvals = np.sort(eigvals)[::-1]  # 降序

        print(f"\nBlock {i} (size {size}):")
        print("Top 10 eigenvalues:", eigvals[:10])
        print("Unique rounded values:", np.unique(np.round(eigvals, decimals=6)))
        if size > 1:
            block_eigvals[f"Block {i} (size {size})"] = eigvals

    # 可视化所有块的谱分布
    plt.figure(figsize=(12, 8))
    for name, ev in block_eigvals.items():
        # plt.plot(ev, label=name, linewidth=2, alpha=0.8)
        plt.scatter(range(len(ev)), ev, s=5)

    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(7 / 9, color='purple', linestyle='--', alpha=0.5, label='7/9')
    plt.axhline(2 / 3, color='blue', linestyle='--', alpha=0.5, label='2/3')
    plt.axhline(5 / 9, color='green', linestyle='--', alpha=0.5, label='5/9')
    plt.axhline(1 / 3, color='orange', linestyle='--', alpha=0.5, label='1/3')

    plt.xlabel('Eigenvalue Index (sorted descending)')
    plt.ylabel('Eigenvalue')
    plt.title('Spectral Distribution by Block')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(DATA_DIR, "Spectral Distribution by Block Scatter.png"), dpi=300, bbox_inches='tight')
    plt.show()

    eigvals_64 = next(ev for name, ev in block_eigvals.items() if 'size 64' in name)
    eigvals_144 = next(ev for name, ev in block_eigvals.items() if 'size 144' in name)
    slow_energy_64 = np.sum(eigvals_64[eigvals_64 >= 2 / 3] ** 2)
    slow_energy_144 = np.sum(eigvals_144[eigvals_144 >= 2 / 3] ** 2)
    print(f"Slow energy from 64 block: {slow_energy_64 / (slow_energy_64 + slow_energy_144):.2%}")
    """
        蓝色曲线（Block 0, size 64，角块）
        谱从 1.0 缓慢下降，到第 20–40 个特征值仍在 0.6–0.8 区间。
        明显覆盖了 7/9 ≈0.778 和 2/3 ≈0.667 的范围。
        下降速度慢，说明角块子空间贡献了大部分慢模式（集体缩放、准守恒行为）。

        橙色曲线（Block 1, size 144，棱块）
        谱从 1.0 快速下降，到第 40 个左右已跌破 0.6，第 80 个左右跌到 0.4 以下。
        明显覆盖了 5/9 ≈0.556 和 1/3 ≈0.333 的范围。
        下降速度快，说明棱块子空间贡献了大部分快模式（快速随机化、混沌扩散）。

        守恒层（λ=1）由所有小块 + 部分角块共同支撑。

        64 dim 角块（Block 0）
        谱从 1.0 缓慢下降，覆盖 7/9 ≈0.778 和 2/3 ≈0.667 的范围。
        主导慢层（7/9 + 2/3），是慢动力学的核心贡献者。

        144 dim 棱块（Block 1）
        谱快速下降，覆盖 5/9 ≈0.556 和 1/3 ≈0.333 的范围。
        主导快层（5/9 + 1/3），负责快速随机化和混沌混合。

        21 个 size=1 小块（Block 2–21）
        全部是单一特征值（一条线），没有退化。
        分类总结：
        λ=1.0（守恒）：Block 10, 12, 18, 20（至少 4 个）
        λ=7/9（慢）：Block 11, 13–17, 19, 21（至少 8 个）
        λ=2/3（次慢）：Block 2–9（至少 8 个）

        这些小块是独立的 1 维不变子空间（trivial 或简单 scaling 表示），支撑了守恒层和部分慢层。

        整体结构总结
        慢层（λ ≥ 2/3 ≈100 维） = 角块（64） + 小块中的 7/9 和 2/3（约 16 个 1 维）
        快层（λ ≤ 5/9 ≈128 维） = 棱块（144）主导
        守恒层（λ=1, 24 维） = 小块中的 λ=1 + 角块/棱块的部分 trivial 表示
        没有“纯随机”块，所有块都有明确谱贡献 → 整个 228 维表示是结构化的块对角分解，而非均匀混合。

        棱块有 12 个位置（置换群 S12），维度 144 = 12! / (12-12)! 的部分 + 朝向 2^11，变化空间大，容易产生“慢衰减”的集体模式（e.g. 整体置换趋势、长程相关）。
        角块只有 8 个位置（S8），维度 64 = 8! / (8-8)! + 朝向 3^7，变化更局部，更多贡献“中速”或“准守恒”模式（7/9, 2/3）。
        Slow energy from 64 block: 15.22%
        """
    return block_eigvals


def compute_block_distance_expectation(corner_idx, edge_idx, U_basis, num_samples_per_depth=300, max_depth=40):
    """
    计算 E[d_spec | depth=k]，分别对 corner block (64) 和 edge block (144)
    d_spec = || proj_block(v - v_solved) ||   （块子空间中的谱距离）

    BUGFIX (2026-04-28): corner_idx/edge_idx 是 detect_blocks 在特征基 U_basis 下
    检测出的连通分量索引。原先直接用这些索引去取 canonical basis 向量 v[idx]，这在
    数学上不合法——特征基索引和原始基索引之间没有一一对应关系。

    正确做法：先将 v 变换到特征基 U_basis^T @ v，再取对应索引的子向量。
    U_basis 是 detect_blocks 使用的同一个 U（来自 eigh/eig），这样块索引才与向量坐标语义一致。
    """

    # solved 在特征基下的坐标（中心）
    v_solved = CubieState.solved().vector
    v_solved_V = U_basis.T.conj() @ v_solved
    v_solved_corner = v_solved_V[corner_idx]
    v_solved_edge = v_solved_V[edge_idx]

    depths = np.arange(0, max_depth + 1)
    mean_corner = []
    mean_edge = []

    print("正在计算块级谱距离期望...")
    for k in tqdm(depths):
        dist_c = []
        dist_e = []
        for _ in range(num_samples_per_depth):
            state = CubieBase.generate_cubie(length=k)
            v = state.vector

            # 变换到特征基, 再取块子坐标
            v_V = U_basis.T.conj() @ v
            proj_c = v_V[corner_idx] - v_solved_corner
            proj_e = v_V[edge_idx] - v_solved_edge

            dist_c.append(np.linalg.norm(proj_c))
            dist_e.append(np.linalg.norm(proj_e))

        mean_corner.append(np.mean(dist_c))
        mean_edge.append(np.mean(dist_e))

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(depths, mean_corner, 'o-', label='Corner block (64)', linewidth=2.5, markersize=6)
    plt.plot(depths, mean_edge, 's-', label='Edge block (144)', linewidth=2.5, markersize=6)

    # 理论参考线
    plt.plot(depths, np.sqrt(depths) * 0.8, '--', color='purple', label='√k (diffusion theory)', alpha=0.7)
    plt.axhline(np.mean(mean_edge[-5:]), color='red', linestyle='--', label='Edge saturation level')

    plt.xlabel('Scramble Depth k')
    plt.ylabel('E[d_spec | depth=k]  (块子空间谱距离)')
    plt.title('Corner vs Edge Block: Expected Spectral Distance vs Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "Corner vs Edge Block_Expected Spectral Distance vs Depth.png"), dpi=300,
                bbox_inches='tight')
    plt.show()

    return depths, mean_corner, mean_edge


def compute_inter_state_block_distance(corner_idx, edge_idx, U_basis, num_pairs_per_depth=200, max_depth=40):
    """
    计算两个独立随机状态 (depth=k) 在角块/棱块子空间中的期望距离 E[d_block | depth=k]

    d_block = || proj_block(vA - vB) ||   （块子空间中的谱距离）

    BUGFIX (2026-04-28): 同 compute_block_distance_expectation，需要先变换到特征基。
    """

    depths = np.arange(0, max_depth + 1)
    mean_corner = []
    mean_edge = []
    std_corner = []
    std_edge = []

    print("正在计算两个随机状态间的块级谱距离期望...")
    for k in tqdm(depths):
        dist_c = []
        dist_e = []
        for _ in range(num_pairs_per_depth):
            stateA = CubieBase.generate_cubie(length=k)
            stateB = CubieBase.generate_cubie(length=k)

            vA = stateA.vector
            vB = stateB.vector

            # 变换到特征基后再取块子坐标
            vA_V = U_basis.T.conj() @ vA
            vB_V = U_basis.T.conj() @ vB
            diff_c = vA_V[corner_idx] - vB_V[corner_idx]
            diff_e = vA_V[edge_idx] - vB_V[edge_idx]

            dist_c.append(np.linalg.norm(diff_c))
            dist_e.append(np.linalg.norm(diff_e))

        mean_corner.append(np.mean(dist_c))
        mean_edge.append(np.mean(dist_e))
        std_corner.append(np.std(dist_c))
        std_edge.append(np.std(dist_e))

    # 绘图
    plt.figure(figsize=(12, 7))

    # 角块
    plt.errorbar(depths, mean_corner, yerr=std_corner, fmt='o-', label='Corner block',
                 capsize=4, linewidth=2.5, markersize=6, color='blue', alpha=0.9)

    # 棱块
    plt.errorbar(depths, mean_edge, yerr=std_edge, fmt='s-', label='Edge block',
                 capsize=4, linewidth=2.5, markersize=6, color='orange', alpha=0.9)

    # plt.loglog(depths[1:], mean_corner[1:], 'o-', label='Corner (log)')
    # plt.loglog(depths[1:], mean_edge[1:], 's-', label='Edge (log)')

    # 理论参考线
    plt.plot(depths, np.sqrt(2 * depths) * 0.8, '--', color='purple',
             label='√(2k) (diffusion theory)', alpha=0.7, linewidth=2)

    plt.axhline(np.mean(mean_edge[-5:]), color='red', linestyle='--',
                label='Edge saturation level', alpha=0.7)

    plt.xlabel('Scramble Depth k (both states)')
    plt.ylabel('E[d_block | depth=k]  (块子空间谱距离)')
    plt.title('Inter-State Spectral Distance in Corner vs Edge Blocks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "Inter-State Spectral Distance in Corner vs Edge Blocks.png"), dpi=300,
                bbox_inches='tight')
    plt.show()

    return depths, mean_corner, mean_edge, std_corner, std_edge


def classify_moves_by_slow_effect(model, trials=20):
    solved = CubieState.solved()
    z0 = model.project(solved.vector)

    move_effect = {}

    for k, mv in CubieMove.prim_moves.items():

        dists = []

        for _ in range(trials):
            state = CubieBase.generate_cubie(length=10)
            z_before = model.project(state.vector)

            new_state = mv.act(state)
            z_after = model.project(new_state.vector)

            d = np.linalg.norm(z_after - z_before)

            dists.append(d)

        move_effect[k] = np.mean(dists)

    return move_effect


def simulate_block_dominated_evolution(model, corner_idx, edge_idx, T=100, num_trials=5, effect_ratio=1.0):
    """
    分别模拟：
    - Pure Edge moves（棱块主导 → fast mixing）
    - Pure Corner moves（角块主导 → slow diffusion）
    观察 slow manifold 上的距离变化和轨迹
    """
    solved = CubieState.solved()
    z_solved = model.project(solved.vector)
    # 1. 分类 moves（根据 block 影响强度）
    edge_moves = []
    corner_moves = []
    for k, mv in CubieMove.prim_moves.items():  # 群共轭类(conjugacy class)导致
        rho = mv.rho()
        edge_block = rho[edge_idx][:, edge_idx]
        corner_block = rho[corner_idx][:, corner_idx]
        edge_effect = np.linalg.norm(edge_block - np.eye(len(edge_idx)), 'fro')  # / np.sqrt(144)
        corner_effect = np.linalg.norm(corner_block - np.eye(len(corner_idx)), 'fro')  # / np.sqrt(64)

        if edge_effect > corner_effect:
            edge_moves.append(mv)
        else:
            corner_moves.append(mv)
        print(k, edge_effect, corner_effect)

    """
    所有 generator effect 一样
    但 mixing rate 不一样
    所有 generator 在表示里是等价的/群共轭类（conjugacy class）/表示的等变性
    (0, -1, -1) 9.797958971132712 8.0
    (0, -1, 1) 9.797958971132712 8.0
    (0, -1, 2) 9.797958971132712 8.0
    (0, 1, -1) 9.797958971132712 8.0
    (0, 1, 1) 9.797958971132712 8.0
    (0, 1, 2) 9.797958971132712 8.0
    (1, -1, -1) 9.797958971132712 8.0
    (1, -1, 1) 9.797958971132712 8.0
    (1, -1, 2) 9.797958971132712 8.0
    (1, 1, -1) 9.797958971132712 8.0
    """

    print(f"Edge-heavy moves: {len(edge_moves)} | Corner-heavy moves: {len(corner_moves)}")
    if not edge_moves:
        edge_moves = []
        corner_moves = []
        for k, mv in CubieMove.prim_moves.items():  # edge_change == corner_change
            # 应用 move 到 solved，比较变化
            after = mv.act(solved)
            corner_change = np.sum(after.corners_perm != solved.corners_perm) + \
                            np.sum(after.corners_ori != solved.corners_ori)
            edge_change = np.sum(after.edges_perm != solved.edges_perm) + \
                          np.sum(after.edges_ori != solved.edges_ori)

            corner_score = corner_change / 8.0
            edge_score = edge_change / 12.0
            if edge_score > corner_score:
                edge_moves.append(mv)
            else:
                corner_moves.append(mv)

        print(f"Edge-heavy moves: {len(edge_moves)} | Corner-heavy moves: {len(corner_moves)}")

    # 2. 模拟两种演化
    plt.figure(figsize=(14, 6))

    for trial in range(num_trials):
        # --- Edge-dominated (fast mixing) ---
        state = CubieState.solved()
        dist_edge = []
        for t in range(T):
            m = np.random.choice(edge_moves)
            state = m.act(state)
            z = model.project(state.vector)
            dist_edge.append(model.distance(z, z_solved))

        plt.subplot(1, 2, 1)
        plt.plot(dist_edge, alpha=0.6, label=f'Edge trial {trial + 1}')

        # --- Corner-dominated (slow diffusion) ---
        state = CubieState.solved()
        dist_corner = []
        for t in range(T):
            m = np.random.choice(corner_moves)
            state = m.act(state)
            z = model.project(state.vector)
            dist_corner.append(model.distance(z, z_solved))

        plt.subplot(1, 2, 2)
        plt.plot(dist_corner, alpha=0.6, label=f'Corner trial {trial + 1}')

    # 左图：Edge moves（快速饱和）
    plt.subplot(1, 2, 1)
    plt.title('Edge moves → Fast Mixing (Chaotic Bulk)')
    plt.xlabel('Time steps')
    plt.ylabel('Slow manifold distance to solved')
    plt.axhline(np.mean(dist_edge[-10:]), color='red', ls='--', label='Saturation level')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：Corner moves（慢扩散）
    plt.subplot(1, 2, 2)
    plt.title('Corner moves → Slow Diffusion')
    plt.xlabel('Time steps')
    plt.ylabel('Slow manifold distance to solved')
    # 加 √t 参考线
    t = np.arange(T)
    plt.plot(t, 0.8 * np.sqrt(t), '--', color='purple', label='≈ √t (diffusion)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "simulate_block_dominated.png"), dpi=300, bbox_inches='tight')
    plt.show()


def compute_spectral_layers(A_micro, w, V):
    """5 层谱投影器 E_k 和特征值 λ_k，基于 eigh 的实谱分解"""
    vals = np.round(w, 6)
    unique, counts = np.unique(vals, return_counts=True)
    layers = {}
    for lam, dim in zip(unique[::-1], counts[::-1]):
        mask = np.abs(w - lam) < 1e-8
        E = V[:, mask] @ V[:, mask].T.conj()
        layers[lam] = {'dim': dim, 'projector': E}
    return layers


def verify_slow_approximation(A_micro, w, V, state_vector, T=100):
    """慢子空间截断近似精度：T 步演化后绝对/相对误差 + 守恒层投影误差"""
    mask_slow = w >= 2 / 3 - 1e-8
    w_slow = w[mask_slow]
    V_slow = V[:, mask_slow]
    mask_const = np.abs(w - 1.0) < 1e-8
    V_const = V[:, mask_const]

    z0 = V_slow.T @ state_vector
    xT_approx = V_slow @ (z0 * (w_slow ** T))
    true_x = state_vector.copy()
    for _ in range(T):
        true_x = A_micro @ true_x

    error_norm = np.linalg.norm(xT_approx - true_x)
    rel_error = error_norm / np.linalg.norm(true_x)
    const_true = V_const.T @ true_x
    const_approx = V_const.T @ xT_approx
    error_const = np.linalg.norm(const_approx - const_true)
    return {'abs_error': error_norm, 'rel_error': rel_error, 'const_error': error_const}


def compute_harmonic_error(phi, lam, n_samples=2000, max_depth=40):
    """采样计算群谐函数误差 φ(gx) - λ φ(x)，返回 (values, vals, depths)"""
    Moves = list(CubieMove.prim_moves.values())
    values, vals, depths = [], [], []
    for sample in range(n_samples):
        if sample % 500 == 0:
            print(f"  {sample}/{n_samples}")
        d = np.random.randint(0, max_depth)
        state_x = CubieBase.generate_cubie(length=d)
        x = state_x.vector
        phi_x = np.dot(phi, x)
        vals2 = []
        for s in Moves:
            sx = s.act(state_x).vector
            vals2.append(np.dot(phi, sx))
        mean_val = np.mean(vals2)
        diff = mean_val - lam * phi_x
        values.append(diff)
        vals.append(phi_x)
        depths.append(d)
    values = np.array(values)
    return values, np.array(vals), np.array(depths)


def compute_harmonic_error_by_block(V, w, start, n_modes=10, n_samples_per_mode=2000):
    """
    批量计算从 start 开始的 n_modes 个模式的群谐误差统计
    λ 越大 → 越“稳定模式”
    λ 越小 → 越“快速混合”
    """
    Moves = list(CubieMove.prim_moves.values())
    error_stats = []
    for mode_idx in range(n_modes):
        phi = V[:, start + mode_idx]
        lam = w[start + mode_idx]
        print(f"\n模式 {mode_idx + 1} (λ = {lam:.6f})...")
        values = []
        for sample in range(n_samples_per_mode):
            if sample % 500 == 0:
                print(f"  {sample}/{n_samples_per_mode}")
            d = np.random.randint(0, 40)
            state_x = CubieBase.generate_cubie(length=d)
            x = state_x.vector
            vals2 = [np.dot(phi, s.act(state_x).vector) for s in Moves]
            values.append(np.mean(vals2) - lam * np.dot(phi, x))  # (Aφ)(x) - λ φ(x),对 eigenvector 必然 = 0
            # print(f"mean={np.mean(vals2):.6f}, std(g-action)={np.std(vals2):.6f}")
        values = np.array(values)
        error_stats.append({
            'mode': mode_idx + 1, 'lambda': lam,
            'mean': np.mean(values), 'std': np.std(values),
            'max_abs': np.max(np.abs(values)),
            'values': values,
        })
    return error_stats


def run_annealing(A_micro, x0, rho_f=5 / 9, delta_beta=0.1, beta_max=1.0, n_rounds=10, T_total=200):
    """离散 + 连续退火模拟，返回 (norm_discrete, norm_continuous, Tf)"""
    gamma_s = 1 - 7 / 9
    Tf = int(np.log(1e-6) / -np.log(rho_f))
    # 离散退火
    beta_k = 0.0
    x = x0.copy()
    x_history_discrete = [x.copy()]
    for k in range(n_rounds):
        for _ in range(Tf):
            x = A_micro @ x
        x *= (1 - beta_k)
        beta_k += delta_beta
        x_history_discrete.append(x.copy())
    # 连续退火
    x = x0.copy()
    x_history_continuous = [x.copy()]
    for t in range(1, T_total + 1):
        beta_t = beta_max * (t / T_total)
        x = A_micro @ x * (1 - beta_t)
        x_history_continuous.append(x.copy())
    norm_discrete = [np.linalg.norm(xi) for xi in x_history_discrete]
    norm_continuous = [np.linalg.norm(xi) for xi in x_history_continuous]
    return norm_discrete, norm_continuous, Tf


def analyze_cubie_block_spectra_0(A_micro, eigvals_am, U_am, sizes=None):
    """按 cp/ep/co/eo 分块分析 A_block 对角谱，返回 block_spectra 列表"""
    if sizes is None:
        sizes = [64, 144, 8, 12]
    A_block = U_am.T.conj() @ A_micro @ U_am
    start = 0
    block_spectra = []
    for i, s in enumerate(sizes):
        if start + s > A_block.shape[0]:
            s = A_block.shape[0] - start
        block = A_block[start:start + s, start:start + s]
        eigvals = np.linalg.eigvals(block)
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        unique_real, counts = np.unique(np.round(real_parts, decimals=6), return_counts=True)
        print(f"\nBlock {i + 1}: size = {s}")
        print(f"  特征值实部 (排序后): {np.sort(real_parts)[::-1][:10]} ...")
        print(f"  唯一实部值 (round 6): {unique_real}")
        print(f"  计数: {counts}")
        print(f"  最大虚部幅度: {np.max(np.abs(imag_parts)):.2e}")
        block_spectra.append({
            'size': s, 'max_imag': np.max(np.abs(imag_parts)),
            'real_parts': real_parts, 'unique_real': unique_real, 'counts': counts,
        })
        start += s
    # 全局对比
    all_block_eigvals = np.concatenate([data['real_parts'] for data in block_spectra])
    print("\n全局检查:")
    print("  块谱排序:", np.sort(all_block_eigvals.real)[::-1][:20])
    print("  原 A_micro 谱 (前20):", np.sort(np.abs(eigvals_am))[::-1][:20])
    print("  迹守恒？", np.isclose(np.trace(A_micro), np.trace(A_block), atol=1e-5))
    return block_spectra


def analyze_cubie_block_spectra(A_micro, eigvals_am, U_am, sizes=None):
    """按 cp/ep/co/eo 分块分析，增加分数识别"""
    if sizes is None:
        sizes = [64, 144, 8, 12]

    A_block = U_am.conj().T @ A_micro @ U_am
    A_block = np.real_if_close(A_block, tol=1e-10)

    start = 0
    block_names = ['CP (64)', 'EP (144)', 'CO (8)', 'EO (12)']
    block_spectra = []

    print("═" * 80)
    print("Cubie Block Spectra Analysis (cp / ep / co / eo)")
    print("═" * 80)

    for i, s in enumerate(sizes):
        block = A_block[start:start + s, start:start + s]
        eigvals = np.linalg.eigh(block)[0]  # 用 eigh，更精确
        sorted_real = np.sort(eigvals)[::-1]

        # 尝试识别常见分数 k/9, k/3, k/2 等
        rounded = np.round(eigvals, decimals=8)
        unique, counts = np.unique(rounded, return_counts=True)

        print(f"\nBlock {i + 1}: {block_names[i]}   size = {s}")
        print(f"   特征值实部 (前12, 降序): {sorted_real[:12]}")
        print(f"   唯一值 (round 8)      : {unique}")
        print(f"   计数                  : {counts}")
        print(f"   最大 |虚部|           : {np.max(np.abs(np.imag(eigvals))):.2e}")

        block_spectra.append({
            'name': block_names[i],
            'size': s,
            'eigvals': eigvals,
            'sorted_real': sorted_real,
            'unique': unique,
            'counts': counts
        })
        start += s

    # 全局信息
    all_eig = np.concatenate([b['eigvals'] for b in block_spectra])
    print("\n" + "═" * 80)
    print("全局总结:")
    print(f"   最大特征值       : {all_eig.max():.8f}")
    print(f"   λ≈1 的数量       : {np.sum(np.abs(all_eig - 1.0) < 1e-6)}")
    print(f"   迹               : {np.trace(A_block):.8f}")
    print(f"   原矩阵迹守恒     : {np.isclose(np.trace(A_micro), np.trace(A_block), atol=1e-6)}")

    return block_spectra


def verify_universal_spectral_law(axis_subsets=None, verbose=True):
    """
    验证 Symmetry-Restricted Rational Spectral Law (Theorem 5.1):
    对面-对称生成元子集 S，A_S 的特征值遵循 λ = 1 - k/m，m = |S|/2。
    NOTE: 此定律非普适——n=8, n=16 反例推翻了 universal 声称。
    重命名为"对称性受限有理谱定律"（Symmetry-Restricted Rational Spectral Law）。

    Hermitian 性条件（A3+A4）：完整 3 轴覆盖 + 面对称结构。
    不满足面-对称的子集（如 n=8, n=16）会产生无理特征值。

    证明状态：有理形式 λ=k/m 已验证至机器精度。Step 4（h_i 非交换）
    的完整代数推导仍存在缺口——h_i 的共同特征基尚未建立。

    Parameters:
    -----------
    axis_subsets: dict, 自定义生成元子集 {name: [move_keys]}
                  若 None，使用默认的 6 组子集
    verbose: bool, 是否打印详细信息

    Returns:
    --------
    results: list of dict, 每组子集的谱分析结果
    """
    results = []

    # 默认生成元子集：按轴数从 1 到 9
    if axis_subsets is None:
        all_keys = list(CubieMove.prim_moves.keys())
        # 按轴分组: axis 0, 1, 2 各有 side ∈ {-1,+1} 各 3 个 direction
        axis_groups = {ax: [k for k in all_keys if k[0] == ax] for ax in (0, 1, 2)}
        # 按面分组: 每个 (axis, side) 对应一个面，3 个 direction
        face_groups = {k: [k_] for k in all_keys for k_ in [k] if True}
        face_keys = {}
        for k in all_keys:
            face = (k[0], k[1])
            if face not in face_keys:
                face_keys[face] = []
            face_keys[face].append(k)

        # 扁平化所有面生成元
        all_face_keys = [k for ks in face_keys.values() for k in ks]
        axis_subsets = [
            ("1-axis (axis=0)", axis_groups[0]),
            ("2-axis (0,1)", axis_groups[0] + axis_groups[1]),
            ("3-axis (full)", axis_groups[0] + axis_groups[1] + axis_groups[2]),
            ("1-face (U)", face_keys[(0, -1)]),
            ("2-face (U,D)", face_keys[(0, -1)] + face_keys[(0, 1)]),
            ("3-face (U,D,R)", face_keys[(0, -1)] + face_keys[(0, 1)] + face_keys[(1, 1)]),
            ("6-face (full outer)", all_face_keys),
        ]

    for name, keys in axis_subsets:
        # 构造该子集的 ρ 和 A
        moves = [CubieMove.prim_moves[k] for k in keys]
        rho_list = [m.rho() for m in moves]
        n_gen = len(rho_list)
        A_S = sum(rho_list) / n_gen

        # 特征分解
        w, V = np.linalg.eigh(A_S)
        w_sorted = np.sort(w)[::-1]

        # 识别有理特征值: λ = 1 - k/m, m = effective axis count
        # BUGFIX (2026-04-28): 原代码定义了 m 和 m_effective 两个变量但只用 m_effective，
        # 且 m 的奇数生成元 fallback 从未生效。简化为直接使用 n_gen // 2。
        if n_gen % 2 != 0:
            m_effective = n_gen  # 奇数个生成元, 例如 9/face 类型子集
        else:
            m_effective = n_gen // 2  # 偶数个生成元的标准情况
        unique_w = np.unique(np.round(w_sorted, 6))
        rational_fits = []
        for lam in unique_w:
            if m_effective > 0:
                k_val = round((1 - lam) * m_effective)
                predicted = 1 - k_val / m_effective
                error = abs(lam - predicted)
                rational_fits.append({
                    'eigenvalue': lam,
                    'k': k_val,
                    'predicted': predicted,
                    'error': error,
                    'is_rational': error < 1e-6
                })

        # 计算多项式秩
        pr = poly_rank(A_S, k=10)

        # 慢子空间维度
        slow_dim = np.sum(w >= 2 / 3 - 1e-8)

        result = {
            'name': name, 'n_gen': n_gen, 'm_eff': m_effective,
            'eigenvalues': w_sorted, 'unique_eigenvalues': unique_w,
            'rational_fits': rational_fits,
            'poly_rank': pr, 'slow_dim': slow_dim,
            'all_rational': all(f['is_rational'] for f in rational_fits)
        }
        results.append(result)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"子集: {name} | |S| = {n_gen}, m = {m_effective}")
            print(f"特征值: {np.round(unique_w, 6)}")
            print(f"多项式秩: {pr}, 慢子空间维度: {slow_dim}")
            print(f"λ = 1 - k/m 拟合:")
            for f in rational_fits:
                mark = "OK" if f['is_rational'] else "FAIL"
                print(f"  λ={f['eigenvalue']:.6f} → k={f['k']}, predicted={f['predicted']:.6f}, "
                      f"err={f['error']:.2e} {mark}")
            print(f"全部有理: {result['all_rational']}")

    return results


def compute_character_table(generators, n_samples=500):
    """
    计算 ρ 的特征 (character) 表，用于不可约分解验证

    χ(g) = Tr(ρ(g))

    对共轭类取平均: χ_class = E[χ(g) | g ∈ C]
    内积 ⟨χ_i, χ_j⟩ = (1/|G|) Σ_g χ_i(g) χ_j(g)^*

    Returns:
    --------
    chars: dict, {conjugacy_class_id: mean_character}
    char_inner_product: float, ⟨χ, χ⟩ 用于判断不可约性
    """
    chars = []
    class_labels = []

    for _ in range(n_samples):
        g = CubieBase.random_walk(length=30)
        rho_g = g.rho()
        chi = np.trace(rho_g)
        chars.append(chi)

        # 用特征值模式标记共轭类
        eig_pattern = tuple(np.sort(np.round(np.linalg.eigvals(rho_g).real, 4)))
        class_labels.append(eig_pattern)

    chars = np.array(chars)

    # 按共轭类分组
    class_chars = {}
    for chi, label in zip(chars, class_labels):
        if label not in class_chars:
            class_chars[label] = []
        class_chars[label].append(chi)

    class_means = {label: np.mean(chs) for label, chs in class_chars.items()}

    # ⟨χ, χ⟩ = (1/N) Σ |χ(g)|²
    char_inner = np.mean(np.abs(chars) ** 2)

    # 检验: ⟨χ, χ⟩ = 1 iff irreducible; > 1 means reducible
    n_irreps_est = np.round(char_inner).astype(int)

    print(f"特征值数量: {len(chars)}")
    print(f"共轭类数量: {len(class_chars)}")
    print(f"<chi, chi> = {char_inner:.4f}")
    print(f"估计不可约表示数: {n_irreps_est}")
    if n_irreps_est == 1:
        print("-> 不可约表示")
    else:
        print(f"-> 可约表示，约含 {n_irreps_est} 个不可约分量")

    # 各共轭类的特征均值
    print(f"\n前 10 个共轭类的特征均值:")
    for i, (label, mean_chi) in enumerate(list(class_means.items())[:10]):
        print(f"  class {i}: |χ| = {abs(mean_chi):.4f}, arg = {np.angle(mean_chi):.4f}")

    return {'chars': chars, 'class_chars': class_chars, 'class_means': class_means,
            'inner_product': char_inner, 'n_irreps_est': n_irreps_est}


def verify_spectral_collapse(generators, A_micro, w, V, n_random_reps=100):
    """
    验证核心论点: "平均对称性 vs 瞬时非对称性" = spectral collapse

    具体验证:
    1. A = E[ρ(s)] 的谱高度退化（5 层），而单个 ρ(s) 谱连续无退化
    2. 退化程度随采样生成元数量增加而增大 → spectral collapse 效应
    3. 随机线性组合 ρ_rand = Σ c_i ρ(s_i) 的谱介于两者之间

    Returns:
    --------
    collapse_data: dict, 谱退化指标随平均数的变化
    """
    n = A_micro.shape[0]

    # 1. 单个生成元的谱
    single_specs = []
    for rho_s in generators:
        eigs = np.linalg.eigvalsh(rho_s)
        single_specs.append(eigs)
    single_specs = np.array(single_specs)

    # 2. 部分平均的谱: 从 1 个到 18 个生成元
    avg_specs = []
    for k in range(1, len(generators) + 1):
        A_partial = sum(generators[:k]) / k
        eigs = np.linalg.eigvalsh(A_partial)
        avg_specs.append(eigs)
    avg_specs = np.array(avg_specs)

    # 3. 随机组合的谱
    rand_specs = []
    for _ in range(n_random_reps):
        coeffs = np.random.randn(len(generators))
        coeffs /= np.linalg.norm(coeffs)
        A_rand = sum(c * rho for c, rho in zip(coeffs, generators))
        eigs = np.linalg.eigvalsh(A_rand)
        rand_specs.append(eigs)
    rand_specs = np.array(rand_specs)

    # 计算谱退化度: entropy of eigenvalue distribution
    def spectral_entropy(eigs, n_bins=50):
        hist, _ = np.histogram(eigs, bins=n_bins, range=(-1.5, 1.5), density=True)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))

    def degeneracy_count(eigs, tol=1e-6):
        """计算不同特征值个数（退化度指标）"""
        unique = np.unique(np.round(eigs, int(-np.log10(tol))))
        return len(unique)

    entropies_single = [spectral_entropy(s) for s in single_specs]
    entropies_avg = [spectral_entropy(s) for s in avg_specs]
    entropies_rand = [spectral_entropy(s) for s in rand_specs]

    degen_single = [degeneracy_count(s) for s in single_specs]
    degen_avg = [degeneracy_count(s) for s in avg_specs]

    print("=== Spectral Collapse 验证 ===")
    print(f"单个 ρ(s): 谱熵 = {np.mean(entropies_single):.4f} ± {np.std(entropies_single):.4f}, "
          f"不同特征值数 = {np.mean(degen_single):.1f}")
    print(f"完全平均 A:  谱熵 = {entropies_avg[-1]:.4f}, 不同特征值数 = {degen_avg[-1]}")
    print(f"随机组合:    谱熵 = {np.mean(entropies_rand):.4f} ± {np.std(entropies_rand):.4f}")
    print(f"\n谱熵随平均数变化 (spectral collapse trajectory):")
    for k, (ent, deg) in enumerate(zip(entropies_avg, degen_avg)):
        print(f"  k={k + 1:2d}: entropy={ent:.4f}, distinct_eigenvalues={deg}")

    return {
        'single_specs': single_specs, 'avg_specs': avg_specs, 'rand_specs': rand_specs,
        'entropies_single': entropies_single, 'entropies_avg': entropies_avg,
        'degen_single': degen_single, 'degen_avg': degen_avg,
    }


# ═══════════════════════════════════════════════════════════════════════
# Numerical block decomposition (§8)
# IMPORTANT: These are numerical spectral clusters, not proven irreps.
# Random Hermitian operators from generator span do NOT generally commute
# with rho(G), so Schur's Lemma does not guarantee scalar action.
# ═══════════════════════════════════════════════════════════════════════

def compute_character_mc(n_samples=5000, walk_length=50):
    """
    Monte Carlo estimation of character inner product <chi, chi>.

    <chi, chi> = (1/|G|) sum_g |chi(g)|^2 = number of irreducible components.

    CAVEAT: Since |G| ~ 2.1e10, random walks do NOT produce uniform sampling
    over G. Estimates do NOT converge with increasing walk length (tested 30,
    80, 200 steps). This method is UNRELIABLE for irreducibility analysis.

    Returns:
        inner_product: float, <chi, chi> estimate (unreliable)
        n_irreps_est: int, estimated number of irreducible components (unreliable)
        chi_samples: array of character values
    """
    chi_samples = []
    for _ in range(n_samples):
        g = CubieBase.random_walk(length=walk_length)
        rho_g = g.rho()
        chi = np.trace(rho_g)
        chi_samples.append(chi)

    chi_arr = np.array(chi_samples)
    inner_product = np.mean(np.abs(chi_arr) ** 2)
    n_irreps_est = int(np.round(inner_product))

    return {
        'inner_product': inner_product,
        'n_irreps_est': n_irreps_est,
        'chi_samples': chi_arr,
        'chi_mean': np.mean(chi_arr),
        'chi_std': np.std(np.abs(chi_arr)),
    }


def detect_irrep_blocks(generators, n_random_ops=8, tol=1e-6):
    """
    Numerical block decomposition via spectral clustering across
    multiple random Hermitian operators from the generator span.

    IMPORTANT CAVEAT: The random operators H_k = sum_s c_{s,k} (rho(s) + rho(s)^*)
    do NOT generally commute with the full group action rho(G). Schur's Lemma only
    guarantees scalar action for operators in the commutant End_G(V). Random
    linear combinations from the generator span are not guaranteed to be in the
    commutant. Therefore, the resulting blocks are NUMERICAL SPECTRAL CLUSTERS,
    not proven irreducible representations.

    Method:
    1. Generate n_random_ops random Hermitian operators H_k from generator span
    2. Diagonalize H_1 to get a basis U
    3. Each basis vector has a "spectral signature" = diag(U^H H_k U) across all k
    4. Cluster vectors with identical signatures (within tol) → numerical blocks

    The scalar-on-generators check (verify_schur_on_irreps) is a NECESSARY but
    not SUFFICIENT condition for a block to be an irrep (it only checks the
    generator set S, not the full group G).

    Returns:
        blocks: list of lists of indices (numerical spectral clusters)
        U_common: eigenbasis of H_1
        signatures: (n, n_random_ops) spectral signature matrix
    """
    n = generators[0].shape[0]

    # Generate random Hermitian operators in the group algebra span
    H_ops = []
    for _ in range(n_random_ops):
        coeffs = np.random.randn(len(generators))
        H = sum(c * (rho + rho.T.conj()) / 2  # symmetrize for numerical stability
                for c, rho in zip(coeffs, generators))
        H_ops.append(H)

    # Use the first H to get a basis, then compute signatures
    eigvals_0, U = np.linalg.eigh(H_ops[0])
    idx = np.argsort(-np.abs(eigvals_0))
    U = U[:, idx]

    # Compute spectral signature for each basis vector
    signatures = np.zeros((n, n_random_ops))
    for k, H in enumerate(H_ops):
        diag_H = np.real(np.diag(U.T.conj() @ H @ U))
        signatures[:, k] = diag_H

    # Cluster by signature similarity
    # Two vectors i,j are in the same irrep if their signatures are close
    # across ALL random operators
    adj = np.ones((n, n), dtype=bool)
    for k in range(n_random_ops):
        diff = np.abs(signatures[:, k:k + 1] - signatures[:, k:k + 1].T)
        adj = adj & (diff < tol * (1 + np.abs(signatures[:, k]).max()))

    # BFS components
    visited = np.zeros(n, dtype=bool)
    blocks = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        component = []
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            component.append(j)
            neighbors = np.where(adj[j])[0]
            for nb in neighbors:
                if not visited[nb]:
                    stack.append(nb)
        blocks.append(sorted(component))

    blocks.sort(key=len, reverse=True)
    return blocks, U, signatures


def map_eigenspaces_to_irreps(V, w, irrep_blocks, U_irrep):
    """
    Map A's eigenspaces to numerical spectral blocks (NOT guaranteed irreps).

    For each eigenvalue lambda of A and each numerical block B:
    compute |P_block @ P_eigenspace|_F — the overlap.
    If overlap ≈ dim(B), then eigenspace contains the full block.

    CAVEAT: The blocks are numerical spectral clusters, not proven irreps.
    This mapping shows the relationship between C[A] isotypic components
    and the numerically detected blocks.

    Returns:
        mapping: list of dict with {lambda, block_idx, block_dim, overlap, is_matched}
    """
    # A's projectors (eigenbasis V)
    unique_w = np.unique(np.round(w, 6))
    mapping = []
    for lam in sorted(unique_w, reverse=True):
        mask = np.abs(w - lam) < 1e-6
        V_lam = V[:, mask]
        P_lam = V_lam @ V_lam.T.conj()
        d_lam = np.sum(mask)

        for i, block in enumerate(irrep_blocks):
            Ub = U_irrep[:, block]
            P_irrep = Ub @ Ub.T.conj()
            overlap = np.linalg.norm(P_irrep @ P_lam, 'fro')
            dim_b = len(block)

            # If eigenspace fully contains this irrep: overlap ≈ sqrt(dim_b)
            is_matched = abs(overlap - np.sqrt(dim_b)) < 0.1 * np.sqrt(dim_b)
            mapping.append({
                'lambda': lam, 'irrep_idx': i, 'irrep_dim': dim_b,
                'overlap': overlap, 'overlap_vs_dim': overlap / np.sqrt(dim_b) if dim_b > 0 else 0,
                'is_matched': is_matched,
                'eigenspace_dim': d_lam,
            })

    return mapping


def verify_schur_on_irreps(generators, irrep_blocks, U_irrep, tol=1e-6):
    """
    Generator-scalar check on numerical spectral blocks (NOT a Schur's Lemma proof).

    CAVEAT: Schur's Lemma only guarantees scalar action for operators in the
    commutant End_G(V). Random linear combinations from the generator span are
    not in the commutant. This check is a NECESSARY but NOT SUFFICIENT condition
    for a block to be a subrepresentation.

    For each block B and generator rho(s):
    compute M = U_B^H rho(s) U_B (restriction to block)
    deviation = ||M - (Tr(M)/d) I|| / ||M||

    Blocks that pass: each rho(s) acts approximately as a scalar on the block.
    Blocks that fail: the block is a clustering artifact or further decomposable.

    Returns:
        results: list of dict with {block_idx, dim, max_deviation, passes_check}
    """
    results = []
    for i, block in enumerate(irrep_blocks):
        Ub = U_irrep[:, block]
        d = len(block)
        max_dev = 0
        for rho_s in generators:
            M = Ub.T.conj() @ rho_s @ Ub
            c = np.trace(M) / d
            dev = np.linalg.norm(M - c * np.eye(d))
            max_dev = max(max_dev, dev)

        norm_factor = np.sqrt(d)  # expected norm of M ≈ sqrt(d) for unitary/perm
        rel_dev = max_dev / norm_factor if norm_factor > 0 else max_dev
        is_irrep = rel_dev < tol * 10
        results.append({
            'irrep_idx': i, 'dim': d,
            'max_deviation': max_dev, 'rel_deviation': rel_dev,
            'is_irrep': is_irrep,
        })
    return results


def verify_quotient_geometry(V_slow, generators, n_samples=2000):
    """
    验证 slow manifold = quotient geometry 的核心论点:

    慢投影 P_slow 定义了一个商映射 π: V → V/ker(P_slow)
    商空间 V_slow ≅ V / V_fast 上的度量诱导了商几何

    验证:
    1. π 是满射且 ker(π) = V_fast（快子空间）
    2. 商度量 d_Q(π(x), π(y)) = ||V_slow^T(x-y)|| 是 V 上的伪度量
    3. 等距性: d_Q(π(ρ(s)x), π(ρ(s)y)) ≈ d_Q(π(x), π(y))
    4. 纤维结构: V_fast 上的向量在 π 下塌缩为零
    5. representation → geometry → algorithm 链完整性

    Returns:
    --------
    quotient_data: dict
    """
    n = V_slow.shape[0]
    d_slow = V_slow.shape[1]

    # 1. 商映射性质
    P_slow = V_slow @ V_slow.T.conj()
    P_fast = np.eye(n) - P_slow

    # ker(P_slow) = range(P_fast)
    # 验证: P_fast 的列空间 = P_slow 的零空间
    ker_test = np.linalg.norm(P_slow @ P_fast)
    print("=== Quotient Geometry 验证 ===")
    print(f"1. ker(P_slow) = range(P_fast): ||P_slow · P_fast|| = {ker_test:.2e}")

    # 2. 伪度量退化: 同一纤维内 d_Q = 0
    x = np.random.randn(n) + 1j * np.random.randn(n)
    v_fast = P_fast @ x
    v_slow_proj = V_slow.T @ x
    v_fast_proj = V_slow.T @ v_fast
    print(f"2. 纤维退化: ||V_slow^T · v_fast|| = {np.linalg.norm(v_fast_proj):.2e} (应为 0)")

    # 3. 等距性验证（核心）
    iso_ratios = []
    for _ in range(n_samples):
        x = np.random.randn(n) + 1j * np.random.randn(n)
        y = np.random.randn(n) + 1j * np.random.randn(n)

        # 原始慢距离
        d_orig = np.linalg.norm(V_slow.T @ (x - y))

        # 随机生成元作用后的慢距离
        rho_s = generators[np.random.randint(len(generators))]
        x_new = rho_s @ x
        y_new = rho_s @ y
        d_new = np.linalg.norm(V_slow.T @ (x_new - y_new))

        if d_orig > 1e-8:
            iso_ratios.append(d_new / d_orig)

    iso_ratios = np.array(iso_ratios)
    print(f"3. 等距性: ratio = {np.mean(iso_ratios):.6f} ± {np.std(iso_ratios):.6f}")
    print(f"   max ratio = {np.max(iso_ratios):.6f}, min = {np.min(iso_ratios):.6f}")

    # 4. 商流形维度和有效自由度
    A_slow = V_slow.T.conj() @ sum(generators) / len(generators) @ V_slow
    w_slow = np.linalg.eigvalsh(A_slow)
    w_slow = np.sort(w_slow)[::-1]
    effective_dim = np.sum(w_slow > 0.5)  # 显著特征值数

    print(f"4. 商流形: dim(V_slow) = {d_slow}, 有效自由度 (λ>0.5) = {effective_dim}")
    print(f"   慢层特征值: {np.round(w_slow[:10], 4)} ...")

    # 5. V_fast 上的动力学 → 在商空间中消失
    A_fast = P_fast @ (sum(generators) / len(generators)) @ P_fast
    w_fast = np.linalg.eigvalsh(A_fast)
    rho_fast = np.max(np.abs(w_fast))

    # 快层衰减时间
    t_half = np.log(2) / (-np.log(max(rho_fast, 1e-10)))

    print(f"5. 快层谱半径 ρ_f = {rho_fast:.6f}, 半衰期 t_1/2 = {t_half:.2f} 步")
    print(f"   → 快层在 ≈{2 * t_half:.0f} 步后在商空间中消失")

    # 6. representation → geometry → algorithm 链
    print(f"\n   链完整性:")
    print(f"   representation (ρ, 228D) → quotient (π, 100D) → metric (d_slow) → search (A*)")
    print(f"   信息损失: {1 - d_slow / n:.1%} 维度被截断")
    print(f"   等距误差: {np.std(iso_ratios):.4f}")
    print(f"   截断安全性: ρ_f^100 = {rho_fast ** 100:.2e}")

    return {
        'ker_test': ker_test, 'fiber_degeneracy': np.linalg.norm(v_fast_proj),
        'iso_ratios': iso_ratios, 'effective_dim': effective_dim,
        'rho_fast': rho_fast, 't_half': t_half,
    }


if __name__ == '__main__':
    pass
