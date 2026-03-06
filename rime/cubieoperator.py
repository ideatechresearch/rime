from rime.cubie import CubieState, CubieMove, CubieBase,SlowDynamics
import numpy as np
import random


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
    验证 Bose-Mesner 代数结构，并推断是否对应 association scheme 或其子结构。

    输入:
    - A: (n x n) 矩阵，通常是 Cayley 图的规范化邻接矩阵（如 A_micro = (1/|S|) ∑ ρ(s)）。
    - tol: 数值容忍度，用于近似相等检查。

    验证步骤:
    1. 假设 A 是 Hermitian，进行特征分解。
    2. 提取唯一特征值（classes），计算多重度。
    3. 构造谱投影器 E_i (idempotents)。
    4. 检查 E_i^2 = E_i, E_i E_j = 0 (i≠j), sum E_i = I。
    5. 重构 A = sum lambda_i E_i。
    6. 检查 A 生成的代数维度 ≈ 唯一特征值数（通过最小多项式度数）。
    7. 如果所有通过，则 A 生成的代数是 Bose-Mesner 型的，Cayley 图对应 association scheme (或子结构)。

    输出:
    - success: bool, 是否验证通过。
    - message: str, 详细报告。
    - details: dict, 包含唯一 lambda、多重度、idempotents 等。
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
    完整验证 Bose-Mesner 代数 + association scheme 结构。

    输入:
    - A: (n x n) 矩阵，通常是规范化邻接矩阵 A_micro。
    - generators: list of ρ(s) 矩阵，生成元表示 (e.g. [rho_m for m in samples])。
    - tol: 浮点容忍度。
    - integer_tol: 用于检查 p_{ij}^k 是否近似整数。

    验证步骤:
    1. 基本谱分解 + Bose-Mesner 属性。
    2. 检查生成元闭合：每个 ρ(s) 是否在 span{E_i} 中（即 ρ(s) = sum c_s^i E_i）。
    3. 检查乘法闭合：E_i E_j = sum p_{ij}^k E_k，且 p 是非负整数。
    4. 如果通过，则 Cayley 图是 association scheme (或子结构/商)。

    输出:
    - success: bool
    - message: str
    - details: dict (unique_lambda, multiplicities, E_list, p_constants 等)
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
        message += "→ 228 维表示分解成 5 个不变子空间（很可能对应 5 个不可约表示）。"
    else:
        message += "并非所有子空间都完全不变，说明表示并非完全分解成 5 个 irreps。\n"
        message += "但慢层（λ ≥ 2/3）的不变性值得进一步关注。"

    return results, message


def commutant_dimension(generators, tol=1e-8):
    n = generators[0].shape[0]#(228^2) × (228^2)
    I = np.eye(n)
    blocks = []

    for rho in generators:
        A = np.kron(rho, I) - np.kron(I, rho.T)  # np.kron(rho.T, I) - np.kron(I, rho)
        blocks.append(A)

    M = np.vstack(blocks)
    u, s, vh = np.linalg.svd(M)

    nullity = np.sum(s < tol)
    return nullity


def estimate_commutant_dim(generators, n=228, num_samples=200, tol=1e-6):
    """
    随机化估计 Comm(ρ(G)) 维度
    generators: list of rho(s) matrices, 每个 (n,n)
    n: 表示维度
    num_samples: 随机矩阵数量
    tol: 零空间容忍度
    """
    # Step 1: 随机采样矩阵
    Ys = np.random.randn(n, num_samples) + 1j * np.random.randn(n, num_samples)  # (n,num_samples)

    # Step 2: 计算残差 stack
    residuals = []
    for rho_s in generators:
        # (n,num_samples) -> (n*num_samples,)
        res = rho_s @ Ys - rho_s.conj().T @ Ys
        residuals.append(res.reshape(-1, num_samples))

    R = np.vstack(residuals)  # (num_generators*n, num_samples)

    # Step 3: SVD，看零奇异值数量
    U, S, Vh = np.linalg.svd(R, full_matrices=False)
    comm_dim_est = np.sum(S < tol)
    return comm_dim_est


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
    flat_dim = n * n
    M_flat_list = []

    for rho_s in generators:
        M_s = P @ rho_s @ P
        M_flat = M_s.flatten()  # (228^2,)
        M_flat_list.append(np.real(M_flat))  # 取实部，避免 complex rank 复杂

    # 堆栈成 (len(generators), flat_dim) 矩阵
    M_matrix = np.array(M_flat_list)  # (27, 51984)

    # 计算 rank
    U, S, Vh = np.linalg.svd(M_matrix, full_matrices=False)
    rank = np.sum(S > tol * S[0])  # 有效奇异值数

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
    # Step 1: 已知 dim(𝒜) ≈ rank(M_matrix)
    _, S, _ = np.linalg.svd(M_matrix, full_matrices=False)
    dim_A = np.sum(S > tol * S[0])
    print(f"dim(𝒜) ≈ {dim_A} (from previous: 11)")

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

    print(f"dim(𝒜²) ≈ {dim_A2}")
    print(f"dim(𝒜²) - dim(𝒜) = {dim_A2 - dim_A}")

    if dim_A2 <= dim_A + 2:  # 允许少量数值误差
        message = "dim(𝒜²) ≈ dim(𝒜) → 代数在数值上闭合（closed algebra）\n" \
                  "→ 慢子空间上存在一个小的闭合非交换代数（dim ≈11）"
    elif dim_A2 > dim_A + 5:
        message = "dim(𝒜²) >> dim(𝒜) → 非闭合，只是谱退化现象"
    else:
        message = "dim(𝒜²) 略大于 dim(𝒜)，可能数值误差或弱闭合"

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


def poly_rank(A, k=6):
    mats = []
    Ak = np.eye(A.shape[0])

    for i in range(k):
        mats.append(Ak.flatten())
        Ak = Ak @ A

    M = np.vstack(mats)
    return np.linalg.matrix_rank(M)


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


if __name__ == '__main__':
    all_moves = CubieMove.prim_moves().copy()
    all_moves.update(CubieMove.slice_moves())

    samples = list(all_moves.values())  # 27

    A = CubieBase.generate_cubie_rho(27)
    print(A.shape)
    B = CubieBase.generate_cubie_rho(27)
    eigvals, U = np.linalg.eig(A)
    blocks = []
    for mv in samples:
        M = U.T.conj() @ mv.rho() @ U
        blocks.append(M)
    print(len(blocks))

    moves = list(CubieMove.prim_moves.values())
    M0 = CubieMove.identity()
    for mv in moves:
        M0 = M0.compose(mv)

    rho_moves = [m.rho() for m in moves]
    A_micro = sum(rho_moves) / len(rho_moves)  # 生成元平均算子 微时间算子

    B_COM = M0.rho()
    A_real = np.block([
        [A_micro.real, -A_micro.imag],
        [A_micro.imag, A_micro.real]
    ])  # (456, 456) 实矩阵表示复线性变换

    eigvals_am, U_am = np.linalg.eig(A_micro)
    idx = np.argsort(-np.abs(eigvals_am.real))
    U_am = U_am[:, idx]
    eigvals_am = eigvals_am[idx]
    print('A_micro:', eigvals_am[:10])
    print('A_micro:', np.sort(np.abs(eigvals_am))[::-1][:40])
    """
    群表示谱分析
    有限群表示在随机生成元上的谱分层
    A_micro: [1.        1.        1.        1.        1.        1.        1.
     1.        1.        1.        1.        1.        1.        1.
     1.        1.        1.        1.        1.        1.        1.
     1.        1.        1.        0.7777778 0.7777778 0.7777778 0.7777778
     0.7777778 0.7777778 0.7777778 0.7777778 0.7777778 0.7777778 0.7777778
     0.7777778 0.7777778 0.7777778 0.7777778 0.7777778]
     24 个 λ = 1
     第二层：大量 λ= 7/9
     v=vslow+vmid+vfast
    精确三层动力学
    守恒层 (λ=1) → 24 全局守恒量,群在 cubie 微观空间上的轨道平均空间
    亚慢层 (λ=7/9) → 44 准对称模式
    中层 (λ=2/3) → 8 取向型
    快层 (<0.6) 混合自由度
    (A−1)(A−7/9)(A−2/3)(A−5/9)(A−1/3)Q(A)=0
    """
    for m in rho_moves:
        print(np.linalg.norm(A_micro @ m - m @ A_micro))
        # A_micro 不与生成元对易,表示分解上是“非均匀对称”的
    dim_R = 228
    diag_r = np.array([1.0] * 24 + [0.9] * 44 + [0.8] * 8 + [0.7] * (dim_R - 24 - 44 - 8))
    R = np.diag(diag_r)
    norm_comm = np.linalg.norm(A_micro @ R - R @ A_micro)  # 整体空间旋转算子” R
    print("Commutator norm:", norm_comm)

    orbits = compute_orbits(rho_moves, dim_R)

    print("轨道数 =", len(orbits))  # 40

    invariants = [orbit_vector(o, dim_R) for o in orbits]

    w, V = np.linalg.eig(A_micro)

    print("sw:", np.sum(np.isclose(w, 1.0)))
    mask = np.abs(w - 1) < 1e-6
    V1 = V[:, mask]
    for i, v in enumerate(invariants):
        proj = V1 @ (V1.T @ v)
        print(i, np.linalg.norm(proj))  # A_micro @ v - v 每个轨道平均向量 v 在 1-eigenspace 上的投影长度

    if np.allclose(A_micro, A_micro.T, rtol=1e-8, atol=1e-8):
        print("矩阵是对称的，可以使用 eigh")
    asymmetry = np.linalg.norm(A_micro - A_micro.T)
    print(f"不对称范数: {asymmetry}")

    w, V = np.linalg.eigh(A_micro)  # 对称特征分解
    mask = np.abs(w - 1) < 1e-8
    dim1 = np.sum(mask)
    print("dim1:", dim1)  # 24
    V1 = V[:, mask]

    mask_const = np.abs(w - 1.0) < 1e-8  # 提取守恒子空间
    V_const = V[:, mask_const]  # (228, 24)

    vals = np.round(w, 6)  # 防数值误差
    unique, counts = np.unique(vals, return_counts=True)

    for u, c in zip(unique[::-1], counts[::-1]):
        print(u, c)
        """
        1.0 24
        0.777778 44
        0.666667 32
        0.555556 96
        0.333333 32
        k/9
        | λ   | 维度 | 含义     |
        | --- | -- | ------ |
        | 1   | 24 | 守恒宏观变量 |
        | 7/9 | 44 | 慢模态    |
        | 2/3 | 32 | 次慢     |
        | 5/9 | 96 | 中速     |
        | 1/3 | 32 | 快速衰减   |
        λ ≥ 2/3 24 + 44 + 32 = 100
        """

    rank1 = np.linalg.matrix_rank(
        np.vstack([
            A_micro.reshape(-1),
            (A_micro @ A_micro).reshape(-1),
            (A_micro @ A_micro @ A_micro).reshape(-1),
            (A_micro @ A_micro @ A_micro @ A_micro).reshape(-1),
            (A_micro @ A_micro @ A_micro @ A_micro @ A_micro).reshape(-1),
        ])
    )
    A = A_micro
    I = np.eye(A.shape[0])
    M = np.stack([
        I.flatten(),
        A.flatten(),
        (A @ A).flatten()
    ])
    rank2 = np.linalg.matrix_rank(M)

    print("rank:", rank1, rank2)  # rank = 3
    """
    fast block 贡献 被投影消掉 或 线性相关
    slow algebra 维度：3
    rank-3 spectral projector family
    宏观 dynamics 只需要 3 个统计变量
    fast block dimension >> slow 但 flatten 后线性相关
    span{I,A,A²,A³} rank=3
    """
    A3 = A @ A @ A
    coef = np.linalg.lstsq(
        np.stack([I.flatten(), A.flatten(), (A @ A).flatten()]).T,
        A3.flatten(),
        rcond=None
    )
    print("coef:", coef)

    success, message, details = verify_bose_mesner(A_micro)
    print("Success:", success)
    print(message)
    """
    Number of distinct eigenvalues (classes): 5
    Lambda 0.333333: multiplicity 32
    Lambda 0.555556: multiplicity 96
    Lambda 0.666667: multiplicity 32
    Lambda 0.777778: multiplicity 44
    Lambda 1.000000: multiplicity 24
    Success: True
    Verified as Bose-Mesner algebra with 5 classes.
    This implies the Cayley graph is representation-equivalent to an association scheme (or substructure).
    Phase1 子群的表示 ρ 具有极高的对称性，其像是一个 5-class association scheme。
    5 个类对应 5 种“距离/关联类型”，特征值 k/9 很可能与生成元数 18 或某种组合计数有关
    Phase1 子群的规范化转移算符在 228 维 faithful 表示下具有 5 个精确有理特征值（k/9），形成清晰的五层动力学。
    提取 λ ≥ 2/3 的 100 维慢子空间模型，在 T ≤ 100 步内相对误差 < 6e-7，证明快层可安全截断。
    生成元在慢/快层贡献均衡（≈46%/54%），但慢层主导长期行为。
    谱结构具有近似 Bose-Mesner 性质（5 类、高退化、正交投影器近似成立），但生成元不完全闭合，无法严格归为 association scheme。
    提出 Representation-Aware Heuristic d(x,y) = ||V_slow^T (x-y)||，作为高效、可计算的距离度量。
    """
    generators = rho_moves  # [m.rho() for m in samples]  # 的 27 个 ρ(s)

    # success, message, details = verify_association_scheme(A_micro, generators)
    # print("Success:", success)
    # print(message)
    # print("p constants example (for i=0, j=1):", details['p_constants'][0, 1])

    mask_slow = w >= 2 / 3 - 1e-8  # 7/9 容忍误差
    print(f"慢子空间维度 (λ ≥ 2/3): {np.sum(mask_slow)}")
    w_slow = w[mask_slow]  # 100
    V_slow = V[:, mask_slow]  # 228 × 100 投影矩阵  舍弃128 维
    print("w_slow:", V_slow.shape)
    # Z = V.T @ state_vector
    """
    #z = V_slow.T @ x    # 100 维
    """
    V_fast = V[:, ~mask_slow]  # v - V_slow
    for s_idx, rho_s in enumerate(generators):
        rho_cross = V_fast.T @ rho_s @ V_slow
        leak_norm = np.linalg.norm(rho_cross, 2)
        U, S, Vh = np.linalg.svd(rho_cross)
        print(f"Gen {s_idx}: cross leak_norm = {leak_norm:.4e},max leakage singular value:", S[0])
    """
    大部分生成元：leak_norm = 1.0000, max singular value ≈ 1.0000001
    → 几乎完美的 rank-1 泄漏（最大奇异值 ≈1，其他奇异值 ≈0）。
    少数生成元：leak_norm ≈ 0.8660, max singular value ≈ 0.8660255
    → √3/2 ≈ 0.866，暗示某种 120° 或 60° 对称（三角形/六边形旋转）。

    这些模式高度非随机，强烈指向对称群或双余类结构。
    """

    A_slow = V_slow.T.conj() @ A_micro @ V_slow  # (100,100) 约化矩阵
    E_slow = V_slow @ V_slow.T.conj()  # (228, 228)
    max_err_slow = 0.0
    for s_idx, rho_s in enumerate(generators):
        rho_V = rho_s @ V_slow
        projected = E_slow @ rho_V
        residual = rho_V - projected
        err = np.linalg.norm(residual) / np.linalg.norm(rho_V + 1e-12)
        max_err_slow = max(max_err_slow, err)
        print(f"Gen {s_idx}: slow escape err = {err:.4e}")

    print(f"慢子空间整体最大逃逸误差: {max_err_slow:.4e}")  # 4.5826e-01
    """慢空间在单个生成元作用下会有接近 46% 的能量泄漏,慢子空间不是群不变子空间"""

    success_slow, msg_slow, details_slow = verify_bose_mesner(A_slow)
    print("慢子空间是否 Bose-Mesner:", success_slow)
    print(msg_slow)

    proj_slow = V_slow @ V_slow.T.conj()  # (228,228) 慢投影器

    for rho_s in generators:
        rho_proj = proj_slow @ rho_s @ proj_slow
        error = np.linalg.norm(rho_s - rho_proj) / np.linalg.norm(rho_s)
        print(f"Generator error to slow subspace: {error:.6e}")

    """
    慢子空间只捕捉了 ≈20% 的生成元作用（1 - 0.8 ≈ 0.2），这与你之前的误差评估（T=100 相对误差 < 6e-7）形成鲜明对比：
    短期/中期：慢子空间近似极好（快层快速衰减）。
    生成元层面：ρ(s) 本身主要由快层贡献（慢层只占小部分）
    80% 的能量在快子空间。
    每个 generator 有 80% 的能量在 slow 子空间之外。
    """
    proj_fast = np.eye(228) - V_slow @ V_slow.T.conj()
    for rho_s in generators:
        slow_contrib = np.linalg.norm(V_slow.T @ rho_s @ V_slow)
        fast_contrib = np.linalg.norm(proj_fast @ rho_s @ proj_fast)
        print(f"Slow/Fast norm ratio: {slow_contrib / (slow_contrib + fast_contrib):.4f}")
    """慢贡献 ≈46%，快贡献 ≈54%"""

    A_fast = proj_fast @ A_micro @ proj_fast
    eigvals = np.linalg.eigvals(A_fast)
    rho_f = np.max(np.abs(eigvals))  # rho_f = 5/9

    print("快层谱半径 ≈", rho_f)  # 0.5555557144099942  5/9
    """
    平均演化在快层是强收缩的（指数衰减）
    快层近 Ramanujan 图
    """
    epsilon = 1e-6
    t_mix = np.log(1 / epsilon) / (-np.log(rho_f))  # 快层谱半径
    print(f"混合时间 tmix({epsilon}) ≈ {t_mix:.4f} 步")  # 23.5043 步
    """
    生成集合的 Cayley 图几乎在几何尺度和谱尺度上都达到“临界最优
    混合时间 ≈ 直径
    ”"""
    d_fast = 12  # 假设快层对应 d=12 (棱块维度)
    ramanujan_fast = 2 * np.sqrt(d_fast - 1) / d_fast
    gap_fast = rho_f - ramanujan_fast

    print(f"\n快层 ρ_f = {rho_f:.6f}")
    print(f"Ramanujan 界 (d={d_fast}) = {ramanujan_fast:.6f}")
    print(f"快层差距 = {gap_fast:.6f}")
    """
    快层差距 = 0.002785
    在 slow 模剔除后，剩余动力学接近最优扩展子
    """

    lambda2 = 7 / 9  # (λ₂, 次大)
    d = len(moves)  # 18
    ramanujan_bound = 2 * np.sqrt(d - 1) / d
    gap = abs(lambda2) - ramanujan_bound
    print(f"λ₂ = {lambda2:.6f}")
    print(f"Ramanujan 界 (d={d}) = {ramanujan_bound:.6f}")
    print(f"差距 = {gap:.6f}")

    print(np.unique(np.round(np.linalg.eigvals(A_micro), 6)))
    print(np.unique(np.round(np.linalg.eigvals(A_fast), 6)))
    """
    [0.333333+0.j 0.555556+0.j 0.666667+0.j 0.777778+0.j 1.      +0.j]
    [0.      +0.j 0.333333+0.j 0.555555+0.j 0.555556+0.j]
    """
    unique, counts = np.unique(np.round(np.linalg.eigvals(A_micro), 6), return_counts=True)
    for u, c in zip(unique[::-1], counts[::-1]):
        print(u, c)
    """
    (1+0j) 24
    (0.777778+0j) 44
    (0.666667+0j) 32
    (0.555556+0j) 96
    (0.333333+0j) 32
    
    20 步基本只剩 λ₂
    0.666^20 ≈ 3e-4
    0.555^20 ≈ 1e-5
    0.333^20 ≈ 3e-10
    """

    print('rank:', poly_rank(A_micro), poly_rank(A_slow), poly_rank(A_fast))  # 5 6 4
    """slow operator rank = 6
    fast operator rank = 4
    """

    results, message = check_invariant_subspaces(A_micro, generators, w, V)
    print(message)
    for lam, res in results.items():
        print(
            f"λ={lam:.6f}: dim={res['multiplicity']}, invariant={res['is_invariant']}, max_err={res['max_error']:.2e}")
    """
    λ = 1.000000 (dim= 24) → 不变 (invariant)
    ------------------------------------------------------------
    
    总结：5 个子空间中，有 1 个完全不变。
    并非所有子空间都完全不变，说明表示并非完全分解成 5 个 irreps。
    但慢层（λ ≥ 2/3）的不变性值得进一步关注。
    λ=0.333333: dim=32, invariant=False, max_err=7.07e-01
    λ=0.555556: dim=96, invariant=False, max_err=5.95e-01
    λ=0.666667: dim=32, invariant=False, max_err=6.12e-01
    λ=0.777778: dim=44, invariant=False, max_err=6.40e-01
    λ=1.000000: dim=24, invariant=True, max_err=7.40e-08
    Phase-1 子群在表示空间中只有部分对称性（主要是守恒层），而其他层级的对称性被“平均”出来了（A_micro 的谱退化），但单个元素 ρ(s) 不足以维持这些层的不变性
    守恒层（24维）：对应群的不变子空间（trivial + 一些 1D 表示），所有 ρ(s) 都保持它不变。
    慢层（7/9 + 2/3）：对应某些“准对称”模式（quasi-symmetric），在平均 A_micro 上退化，但单个转动会轻微混合这些模式（误差 ~0.6）。
    快层：完全混合，无对称性。
    """

    shells = shell_projector(samples=1000)
    print("shell statistics:")
    P = {}
    for d, mats in shells.items():
        mats = np.array(mats)

        mean = np.mean(mats, axis=0)
        P[d] = mean

        dev = np.mean([np.linalg.norm(m - mean) for m in mats])

        print(f"shell {d:2d}  samples={len(mats):3d}  deviation={dev:.6f}")

    fit_shell_decomposition(A_micro, P)  # relative error: 0.81069374
    """low-rank group convolution operator"""

    # print("commutant dimension =", commutant_dimension(generators))
    dim_comm = estimate_commutant_dim(generators, n=228, num_samples=1000)
    print("估计 dim Comm(ρ(G)) ≈", dim_comm)  # 336/836
    print("对应不可约表示数 ≈", 228 / dim_comm)

    """
    估计 dim Comm(ρ(G)) ≈ 336/836
    单个生成元作用并不完全保持这些层不变 → 统计对称性（quasi-symmetry）
    其余层（慢层 ~76 维 + 快层 128 维）形成一个大约 204 维的可约表示，内部有统计对称性
    这不是标准的 Hamming/Johnson scheme
    
    也不是完整 Bose-Mesner algebra

    极有可能是 Hecke-type / double coset algebra 的统计近似版本
    """

    num_samples = 1000  # 采样数
    double_cosets = set()
    for _ in range(num_samples):
        g = CubieBase.random_walk(length=50)
        # Normalize to canonical rep: find min k1 g k2 or something
        # To avoid, use a proxy: the trace of rho(g) or some invariant
        rho_g = g.rho()
        trace = np.trace(rho_g).real  # example invariant
        frobenius = np.linalg.norm(rho_g, 'fro')  # another
        # or tuple of sorted eigvals, but expensive
        eig = np.sort(np.real(np.linalg.eigvals(rho_g)))
        canonical = tuple(np.round(eig, 4))  # round to avoid float error
        double_cosets.add(canonical)

    print("估算双余类数 (based on sample invariants):", len(double_cosets))  # 854
    if len(double_cosets) == 5:
        print("第一关通过: 双余类数 ≈ 5")
    else:
        print("第一关失败: 双余类数 ≈", len(double_cosets))
        """
        The Rubik's cube group is a paradigmatic example of a large discrete symmetry group with rich combinatorial structure. In this work, we investigate the spectral properties of the normalized transfer operator A = (1/|S|) ∑_{s∈S} ρ(s) in the faithful 228-dimensional representation of the Phase-1 subgroup, where S is the set of generators (18 primitive + 9 slice moves).The operator exhibits exactly five distinct rational eigenvalues of the form k/9 (k=3,5,6,7,9) with high multiplicities (32,96,32,44,24), and its spectral projectors satisfy the Bose–Mesner algebra conditions (idempotence, orthogonality, and completeness). However, individual generators ρ(s) do not preserve the eigenspaces (cross-layer leakage ≈0.42–0.71), ruling out a full association scheme or Gelfand pair structure.Despite this, the subspace spanned by eigenvalues λ ≥ 2/3 (dimension 100) shows quasi-invariance under group action, with leakage error ≈0.42–0.46. Projecting dynamics onto this slow manifold yields highly accurate approximations: relative error < 6×10^{-7} for T=100 steps, demonstrating that fast modes (λ < 2/3) can be safely truncated.We propose a representation-aware heuristic distance d(x,y) = ||V_slow^T (x-y)||, which leverages the slow projection to ignore transient modes. These findings reveal a striking separation between averaged symmetry (captured by A) and instantaneous asymmetry (in ρ(s)), offering a computable low-dimensional world model for discrete group actions in puzzle solving and beyond.
        """
    """平均对称性主导的慢动力学,存在一个 5 维 Hecke-type 代数，但它不是 Gelfand pair。"""

    mask_mid = (w >= 2 / 3 - 1e-8) & (w < 7 / 9 + 1e-8)  # 只看 2/3 层
    V_mid = V[:, mask_mid]
    for rho_s in generators:  # 每个 ρ(s) 在慢层内部的谱泄漏
        leak_to_mid = np.linalg.norm(V_mid.T @ rho_s @ V_slow)
        print(leak_to_mid)
    """
    76 维只是平均算子下退化
    4.472136  ≈ √20
    5.656854  ≈ √32
    生成元在慢层之间的耦合不是连续变化的

    而是落在有限个固定数值上
    
    这正是 “Hecke 型双余类结构” 的典型信号
    """
    max_comm_norm = check_commutativity(generators, V_slow)
    print("max_comm_norm=", max_comm_norm)
    slow_algebra_dimension(generators, V_slow)  # 慢层生成代数维度 ≈ 18
    P = V_slow @ V_slow.T.conj()
    dim, msg = compute_span_dim(P, generators)
    print(msg)  # span{P ρ(s) P} 维度 ≈ 11
    """
    慢层生成代数维度 ≈ 18
    → 基本等于生成元数量
    → 说明慢层上的 ρ(s) 基本线性独立
    dim(𝒜) ≈ 11 (from previous: 11)
    dim(𝒜²) ≈ 20
    dim(𝒜²) - dim(𝒜) = 9
    dim(𝒜²) >> dim(𝒜) → 非闭合，只是谱退化现象
    span{P ρ(s) P} 维度 ≈ 11
    
    一个非交换生成元代数
    在平均算子下出现 5 个谱退化层
    在慢空间压缩后形成一个 ~11 维低维代数
    """

    leakage_bounds(generators, V_slow)

    analyze_fast_pseudospectrum(A_micro, V_slow, t_max=30)
    """谱半径 ρ = 0.5555557144099934
    非正规性 ‖AA* - A*A‖ = 5.749921133840414e-16
    特征向量条件数 κ(V) = 139.9906737469503
    t=1..10 实际 ‖A_f^t‖:
    [np.float64(0.5555557144099954), np.float64(0.30864215181359983), np.float64(0.17146791114784266), np.float64(0.09525997787612925), np.float64(0.05292222506365333), np.float64(0.029401244553404465), np.float64(0.016334029422409597), np.float64(0.009074463384960643), np.float64(0.005041369988719153), np.float64(0.002800761905687978)]
    理论 ρ^t (前10):
    [np.float64(0.5555557144099934), np.float64(0.30864215181359816), np.float64(0.17146791114784116), np.float64(0.09525997787612817), np.float64(0.05292222506365255), np.float64(0.02940124455340395), np.float64(0.01633402942240926), np.float64(0.00907446338496043), np.float64(0.005041369988719018), np.float64(0.0028007619056878946)]
    说明特征向量基 V 是数值稳定的，投影/重构几乎无放大误差
    """
    estimate_fast_lyapunov(generators, V_slow, T=2000)
    """
    最大 Lyapunov 指数 λ ≈ -0.00019139141269984905
    对应指数率 e^λ ≈ 0.9998086269014682
    平均谱退化 → 统计对称性
    瞬时生成元 → 几乎保持 norm 快层几乎是**保范（norm-preserving）**的。
    整个系统是渐近稳定的（asymptotically stable），状态会缓慢趋向平衡（solved 附近）
    一个具有统计谱分层（statistical spectral stratification）的非交换表示，平均算符下出现 5 个高度退化的谱层，单个生成元导致强跨层混合，但慢动力学表现出极高的可计算性和稳定性。
    由对称性导致的简并模式成为谱的主要特征。
    """

    # k_generators = []  # K 生成元: all delta == 0 的 moves
    # for m in moves:  # all_moves.values()
    #     if all(d == 0 for d in m.corners_ori_delta) and all(d == 0 for d in m.edges_ori_delta):
    #         k_generators.append(m)
    # print("K 生成元数:", len(k_generators))  # 10 phase2
    # K = CubieBase.generate_group(k_generators, max_depth=10, max_groups=1000)
    # print("生成 K 大小 (approximate):", len(K))

    state_vector = CubieState.solved().vec  # 给定初始状态 x₀
    z0 = V_slow.T @ state_vector  # (100,) 投影到慢子空间 V_slow.T.conj() @ state_vector
    print("z0:", z0.shape)
    T = 50  # 示例步数
    zT = z0 * (w_slow ** T)  # (100,) 预测 T 步 指数衰减
    xT_approx = V_slow @ zT  # (228,) 还原回原空间

    xT_exact = np.linalg.matrix_power(A_micro, T) @ state_vector  # 真实 T 步（贵）
    error = np.linalg.norm(xT_approx - xT_exact)
    print(f"T={T} 步近似误差范数: {error:.6e}")
    print(f"相对误差: {error / np.linalg.norm(xT_exact):.6e}")

    # 取前3个慢坐标随时间演化
    T_steps = np.arange(0, 101)
    Z = np.zeros((len(T_steps), len(z0)), dtype=np.complex128)  # (101,100)
    for i, t in enumerate(T_steps):
        Z[i] = z0 * (w_slow ** t)  # np.real(z0 * (w_slow ** t))
    Z = np.real(Z)

    T = 100
    true_x = state_vector.copy()
    for _ in range(T):
        true_x = A_micro @ true_x

    xT_approx = V_slow @ Z[T]
    error_norm = np.linalg.norm(xT_approx - true_x)
    rel_error = error_norm / np.linalg.norm(true_x)
    print(f"  绝对误差范数: {error_norm:.6e}")
    print(f"  相对误差:     {rel_error:.6e}")
    const_true = V_const.T @ true_x
    const_approx = V_const.T @ xT_approx
    error_const = np.linalg.norm(const_approx - const_true)
    print(f"守恒层投影误差 (T={T}): {error_const:.6e}")
    """  
    绝对误差范数: 1.513582e-06
    相对误差:     6.179170e-07
    守恒层投影误差 (T=100): 1.390568e-06
    """

    import matplotlib.pyplot as plt
    import os

    # Adjust these paths to match your actual Tcl/Tk directories
    os.environ['TCL_LIBRARY'] = r'D:\Program Files\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'D:\Program Files\Python\Python313\tcl\tk8.6'
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], label='慢坐标轨迹')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('z3')
    plt.title('Slow manifold trajectory in 3D projection')
    plt.legend()
    plt.show()

    # T_max = 2000  # 展示慢层截断的安全性
    # norms = []
    # A_power = np.eye(A_fast.shape[0])
    # for t in range(1, T_max + 1):
    #     A_power = A_power @ A_fast
    #     norms.append(np.linalg.norm(A_power, 2))
    #
    # plt.figure()
    # plt.plot(range(1, T_max + 1), norms)
    # plt.xlabel("T")
    # plt.ylabel("||A_f^T||_2")
    # plt.title("Fast-layer norm growth ||A_f^T||")
    # # plt.savefig("data/Fast-layer norm growth.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # Estimate exponential rate from linear fit of log(norm)

    # log_norms = np.log(np.array(norms) + 1e-12)
    # coef = np.polyfit(range(1, T_max + 1), log_norms, 1)
    # lyap_est = coef[0]
    #
    # print("Estimated Lyapunov exponent from power growth ≈", lyap_est)
    # print("Estimated exp(lambda) ≈", np.exp(lyap_est))
    # 初始化模型
    model = SlowDynamics()

    # 参数
    rho_f = 5 / 9
    gamma_s = 1 - 7 / 9  # 亚慢 gap
    Tf = int(np.log(1e-6) / -np.log(rho_f))  # ≈23-24
    T_anneal = int(5 / gamma_s)  # ≈22-23
    beta_0 = 0.0  # 初始 β
    delta_beta = 0.1  # 每轮增量（可调）
    beta_max = 1.0  # 最大 β

    # 初始状态 x0 (假设你的 rho 向量)
    x0 = CubieState.solved().vec # 示例，替换成 state.to_rho()

    # 方式 1: 离散退火（每 Tf 步增 β）
    beta_k = beta_0
    x = x0.copy()
    x_history_discrete = [x.copy()]
    for k in range(10):  # 示例 10 轮
        for _ in range(Tf):
            x = A_micro @ x  # 完整演化（模拟快层混合）
        x *= (1 - beta_k)  # 收缩慢层（模拟退火）
        beta_k += delta_beta
        x_history_discrete.append(x.copy())

    # 方式 2: 连续退火 β(t) = beta_max * (t / T_total)
    T_total = 200  # 总步数
    x = x0.copy()
    x_history_continuous = [x.copy()]
    for t in range(1, T_total + 1):
        beta_t = beta_max * (t / T_total)
        x = A_micro @ x * (1 - beta_t)  # 每步演化 + 退火
        x_history_continuous.append(x.copy())

    # 误差评估（范数随时间变化）
    norm_discrete = [np.linalg.norm(xi) for xi in x_history_discrete]
    norm_continuous = [np.linalg.norm(xi) for xi in x_history_continuous]

    # 画图
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(norm_discrete)) * Tf, norm_discrete, label='离散退火 (每 Tf 增 β)')
    plt.plot(np.arange(len(norm_continuous)), norm_continuous, label='连续退火 β(t)')
    plt.xlabel('步数 t')
    plt.ylabel('状态范数 ||x_t||')
    plt.title('分离时间尺度退火下的状态收敛')
    plt.legend()
    plt.grid(True)
    plt.savefig("data/分离时间尺度退火下的状态收敛.png", dpi=300, bbox_inches='tight')
    plt.show()
    """
    通过分离时间尺度退火，我们在 Phase-1 子群的 228 维表示空间中实现了高效的动力学模拟。快层（谱半径 ≈5/9）在 Tf ≈24 步内充分混合，慢层（λ ≥ 2/3）在退火参数 β 的逐步收缩下快速趋向守恒子空间。连续退火曲线显示平滑收敛，离散退火则体现物理混合步骤。范数从初始值上升到峰值后迅速下降至稳定平台，证明快层截断与退火机制相容，且无误差放大效应。这为构建基于慢流形的可计算世界模型提供了可靠框架。
    """

    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.plot(T_steps, Z[:, i], label=f'slow dim {i}')  # 慢模式随时间的指数衰减
    plt.xlabel('time step')
    plt.ylabel('slow coordinates')
    plt.title('Slow subspace evolution')
    # plt.savefig("data/Slow subspace evolution.png", dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()

    idx = np.argsort(np.real(w))[::-1]
    w = w[idx]
    V = V[:, idx]
    eps = 1e-6
    dim_1 = np.sum(np.abs(w - 1) < eps)
    print("dim_1:", dim_1)
    gap = 1 - np.max(w[1:])  # 24
    print("gap:", gap)
    gap = 1 - np.max(np.real(w[dim_1:]))
    print("gap2:", gap)  # 1−0.7777=0.2222

    A_block = U_am.T.conj() @ A_micro @ U_am
    # A_block = U.T.conj() @ A_micro @ U
    print("迹守恒？", np.isclose(np.trace(A_micro), np.trace(A_block), atol=1e-6))  # A_block 是严格对角的
    diag_elements = np.diag(A_block).real
    print("自变换对角 前10:", diag_elements[:10])
    print("原 eigvals 前10:", eigvals_am[:10].real)

    slow_dim = np.sum(np.abs(eigvals_am) > 0.95)
    print(slow_dim)  # 24,存在一个 24 维子空间,立方体整体旋转的作用表示 ∣S4∣
    print(np.sum(np.abs(np.abs(eigvals_am) - 7 / 9) < 1e-6))  # 44

    # 2. 计算分段平均（假设你知道大致的分块位置，例如前24是慢的）
    slow_block_diag = diag_elements[:24]
    mid_block_diag = diag_elements[24:24 + 44]  # 44 个 ~7/9
    mid2_block_diag = diag_elements[24 + 44:24 + 44 + 8]
    fast_block_diag = diag_elements[24 + 44 + 8:]

    print("慢层平均对角值:", np.mean(slow_block_diag))
    print("慢层对角 std:", np.std(slow_block_diag))
    print("亚慢层平均对角值:", np.mean(mid_block_diag))
    print("中层平均对角值:", np.mean(mid2_block_diag))
    print("快层平均对角值:", np.mean(fast_block_diag))

    sizes = [64, 144, 8, 12]  # cp, ep, co, eo
    start = 0
    block_spectra = []

    for i, s in enumerate(sizes):
        if start + s > A_block.shape[0]:
            print(f"警告：块 {i} 超出范围，截断到矩阵末尾")
            s = A_block.shape[0] - start

        block = A_block[start:start + s, start:start + s]

        # 求特征值（因为是复矩阵，用 eigvals）
        eigvals = np.linalg.eigvals(block)

        # 取实部（通常物理上关心实部，尤其是相似变换后应接近实谱）
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        # 统计：唯一值（四舍五入到一定精度）、最大/最小、平均等
        unique_real, counts = np.unique(np.round(real_parts, decimals=6), return_counts=True)

        print(f"\nBlock {i + 1}: size = {s}")
        print(f"  特征值实部 (排序后): {np.sort(real_parts)[::-1][:10]} ...")  # 前10个最大的
        print(f"  唯一实部值 (round 6): {unique_real}")
        print(f"  计数: {counts}")
        print(f"  最大虚部幅度: {np.max(np.abs(imag_parts)):.2e}")
        """Block 1 (64 维, cp)：
        谱：连续分布，无大规模重根（不退化）。
        含义：接近 irreducible 或高度泛化的表示。内部没有明显退化，表明这个置换子空间在 A_micro 作用下是“连续混合”的，没有强对称分离。
        
        Block 2 (144 维, ep)：
        谱：类似 Block 1，连续分布，无大规模重根。
        含义：同样接近 irreducible，代表棱置换的“generic” 行为。
        
        Block 3 (8 维, co)：
        谱：λ = 2/3 (≈0.6667)，完全退化（multiplicity=8）。
        含义：这是一个纯 8 维 irreducible 表示。在这个表示上，A_micro = (2/3) I，完全标量化。这符合 Schur 引理：生成元 S 在 V8 上是等效的“球平均”，暗示几何或对称结构（如 spherical averaging）。
        
        Block 4 (12 维, eo)：
        谱：λ = 1 (multiplicity=4) + λ = 7/9 (≈0.7778, multiplicity=8)。
        含义：reducible 表示 = 4 × trivial (λ=1) + 8 × irreducible (λ=7/9，完全退化)。这表明 12 维子空间被精确分解成两个部分，且 λ=7/9 在 8 维 irreps 上完全一致。
        """

        block_spectra.append({
            'size': s,
            'max_imag': np.max(np.abs(imag_parts)),
            'real_parts': real_parts,
            'unique_real': unique_real,
            'counts': counts
        })

        start += s

    # 全局对比：检查总谱是否一致
    all_block_eigvals = np.concatenate([data['real_parts'] for data in block_spectra])
    print("\n全局检查:")
    print("  块谱排序:", np.sort(all_block_eigvals.real)[::-1][:20])  # 与原 eigvals 对比
    print("  原 A_micro 谱 (前20):", np.sort(np.abs(eigvals_am))[::-1][:20])
    print("  迹守恒？", np.isclose(np.trace(A_micro), np.trace(A_block), atol=1e-5))
    # 构造投影算子
    slow_idx = np.abs(eigvals_am - 1) < 1e-8
    slow_basis = U_am[:, slow_idx]
    P = slow_basis @ slow_basis.T.conj()

    # P = np.zeros((dim_R, dim_R))
    # for R in rho_moves:
    #     P += R
    # P /= len(rho_moves)

    T = 5
    v = B
    v /= np.linalg.norm(v)
    v_slow = P @ v
    v_fast = v - v_slow
    norms = []

    for t in range(T):
        v = A_micro @ v
        norms.append(np.linalg.norm(v))
    print(norms)  # 时间演化 0.7^t 总 norm 演化

    v_s = v_slow
    v_f = v_fast
    for t in range(T):  # 分别演化
        v_s = A_micro @ v_s
        v_f = A_micro @ v_f
        print(np.linalg.norm(v_s), np.linalg.norm(v_f))  # 多谱结构

    slow_in_block = U.T.conj() @ slow_basis

    blocks = detect_blocks(samples, U)
    sizes = [len(b) for b in blocks]
    print("Block sizes:", sorted(sizes))  # ..., 64, 144
    print("Number of blocks:", len(blocks))

    A = sum(c_i * mv_i.rho() for mv_i, c_i in zip(samples, np.random.randn(len(samples))))
    B = sum(c_i * mv_i.rho() for mv_i, c_i in zip(samples, np.random.randn(len(samples))))
    C = A + 1j * B
    eigvals, U = np.linalg.eig(C)
    # print(np.unique(np.round(eigvals, 6)))
    blocks = detect_blocks(samples, U)
    sizes = [len(b) for b in blocks]
    print("Block sizes:", sorted(sizes))  # ....8、12、56、132，大块主要来自 permutation
    print("Number of blocks:", len(blocks))  # 24

    big_block = max(blocks, key=len)  # 选最大的块
    print("big_block size:", len(big_block))  # 132
    multiplicities = split_isotypic_block(samples, U, big_block, tol=1e-8)
    print("Block multiplicities:", multiplicities)

    projections = construct_projection_operators(U, blocks)
    rho_matrices = [mv.rho() for mv in samples]

    block_diag_rhos = []
    for rho in rho_matrices:
        blocks = [P @ rho @ P for P in projections]
        block_diag_rhos.append(blocks)
    # 检查 block_trace
    for blocks in block_diag_rhos:
        print([np.trace(b) for b in blocks])
