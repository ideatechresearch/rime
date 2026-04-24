"""
cubieoperator 谱分析 & 代数性质实验

实验模块分组：
  1. 基础设置 & 块检测
  2. 谱结构 (5 层有理谱 k/9)
  3. Bose-Mesner & 代数性质
  4. 慢子空间近似 & 群谐函数
  5. 退火 & 块谱分解

运行: python test/test_cubieoperator.py
"""

import matplotlib

matplotlib.use('Agg')  # 无头渲染
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from rime.cubieoperator import *
from rime.cubeplot import (
    draw_error_histogram, draw_slow_coordinates,
    draw_gram_matrix, draw_annealing,
)

N_MODES = 10
N_SAMPLES = 2000


def test_move_composition():
    """move 组合计数实验"""
    prim_list18 = list(CubieMove.prim_moves.values())

    # 18 个 prim 两两 compose
    products = set()
    for g1 in prim_list18:
        for g2 in prim_list18:
            prod = g1.compose(g2)
            if prod != CubieMove.identity():
                products.add(prod)
    print(f"18 两两 compose 去重+去 identity: {len(products)}")  # 269

    # 12 outer + identity 两两
    prim_list12 = [v for k, v in CubieMove.prim_moves.items() if k[2] != 2]
    ME = CubieMove.identity()
    prim_list13 = prim_list12 + [ME]
    products = set()
    for g1 in prim_list13:
        for g2 in prim_list13:
            prod = g1.compose(g2)
            if prod != ME:
                products.add(prod)
    print(f"12 两两 compose 去重+去 identity: {len(products)}")  # 134

    # + inverse
    products2 = products.copy()
    for g in products:
        g2 = g.inverse()
        if g2 not in products2:
            products2.add(g2)
    print(f"+ inverse: {len(products2)}")  # 268

    # + commutator
    products2 = CubieBase.generate_compose_moves(CubieMove.prim_moves(), commutator=True)
    print(f"+ commutator: {len(products2)}")  # 224
    """
    结果:
    18 两两 compose 去重+去 identity: 269
    12 两两 compose 去重+去 identity: 134
    + inverse: 268
    + commutator: 224
    """


def test_block_detection(A_micro, U_am):
    """块检测 & corner/edge 分离"""
    blocks = detect_blocks(list(CubieMove.prim_moves().values()), U_am)
    sizes = [len(b) for b in blocks]
    print("Block sizes:", sorted(sizes))
    print("Number of blocks:", len(blocks))
    """
    Block sizes: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 144]
    Number of blocks: 22
    20 个 1D block: 11 + 7 + 1 + 1 = 20 合法状态守恒量
    208: 真正参与 random walk dynamics 的空间
    144: fast mixing bulk
    """


# ── 2. 谱结构 ─────────────────────────────────────────────────────────

def test_spectral_layers(A_micro, generators):
    """5 层有理谱 k/9 分析"""
    w, V = np.linalg.eigh(A_micro)
    mask = np.abs(w - 1) < 1e-8
    dim1 = np.sum(mask)
    print("dim1:", dim1)  # 24

    print("dim span {ρ(g)}=", group_algebra_dim(generators))
    print('poly_rank(A_micro):', poly_rank(A_micro))  # 6

    vals = np.round(w, 6)
    unique, counts = np.unique(vals, return_counts=True)
    for u, c in zip(unique[::-1], counts[::-1]):
        print(u, c)
    """
    1.0 24
    0.777778 44
    0.666667 32
    0.555556 96
    0.333333 32

    | λ   | 维度 | 含义       |
    | --- | -- | ---------- |
    | 1   | 24 | 守恒宏观变量 |
    | 7/9 | 44 | 慢模态      |
    | 2/3 | 32 | 次慢        |
    | 5/9 | 96 | 中速        |
    | 1/3 | 32 | 快速衰减    |
    λ ≥ 2/3 → 24 + 44 + 32 = 100
    """

    # poly rank
    A = A_micro
    I = np.eye(A.shape[0])
    rank1 = np.linalg.matrix_rank(np.vstack([
        A.reshape(-1), (A @ A).reshape(-1), (A @ A @ A).reshape(-1),
        (A @ A @ A @ A).reshape(-1), (A @ A @ A @ A @ A).reshape(-1),
    ]))
    M = np.stack([I.flatten(), A.flatten(), (A @ A).flatten()])
    rank2 = np.linalg.matrix_rank(M)
    print("rank(I,A,A^2,...):", rank1, rank2)  # 5, 3
    """
    span{I,A,A²,A³} rank=3
    fast block 贡献被投影消掉或线性相关 → slow algebra 维度 3
    宏观 dynamics 只需要 3 个统计变量
    """


def test_block_spectrum(A_micro, V, blocks, corner_idx, edge_idx):
    """角块/棱块谱贡献分析 + 扩散距离实验"""
    A_block = V.T.conj() @ A_micro @ V
    analyze_block_spectrum(A_block, blocks)
    """
    64 dim 角块: 谱缓慢下降，覆盖 7/9 和 2/3 → 主导慢层
    144 dim 棱块: 谱快速下降，覆盖 5/9 和 1/3 → 主导快层
    21 个 1D 小块: λ=1.0 (4个), λ=7/9 (8个), λ=2/3 (8个)

    Slow energy from 64 block: 15.22%
    
    The 228-dimensional transfer operator A decomposes into block-diagonal form under the cubie basis, with principal blocks of size 64 (corners) and 144 (edges), plus 19 trivial 1-dimensional blocks. The eigenvalue spectrum of each block reveals clear separation: the 64-dimensional corner block dominates the slow layers (λ ≈ 7/9 and 2/3), exhibiting slower decay and contributing to quasi-invariant dynamics. In contrast, the 144-dimensional edge block dominates the fast layers (λ ≈ 5/9 and 1/3), with rapid spectral fall-off consistent with chaotic mixing. The trivial blocks concentrate near λ = 1, supporting the exact invariant subspace (dim 24). This block-level spectral stratification confirms that slow dynamics are primarily driven by corner orientations and permutations, while fast randomization arises from edge permutations, providing a structural explanation for the observed 5-layer rational spectrum and slow-fast separation.
    The 228-dimensional transfer operator decomposes into a block-diagonal form under the cubie basis, with principal blocks of size 64 (corners), 144 (edges), and 21 trivial 1-dimensional blocks. Spectral analysis of each block reveals clear separation of contributions:
    The 64-dimensional corner block dominates the slow layers (λ ≈ 7/9 and 2/3), exhibiting gradual spectral decay and contributing to quasi-invariant dynamics.
    The 144-dimensional edge block dominates the fast layers (λ ≈ 5/9 and 1/3), with rapid fall-off consistent with chaotic mixing.
    The 21 1-dimensional blocks concentrate at discrete values: λ = 1.0 (invariant/trivial), λ = 7/9 (slow scaling), and λ = 2/3 (intermediate diffusion), each contributing exactly one eigenvalue per block.
    This block-level stratification confirms that slow dynamics are primarily driven by corner orientations and permutations, fast randomization by edge permutations, and conserved quantities by trivial 1D representations. The clean separation explains the observed 5-layer rational spectrum and the slow manifold's robustness under group action.
    Block-level energy decomposition of the slow manifold (λ ≥ 2/3, 100 dimensions) reveals that the 144-dimensional edge block contributes approximately 84.78% of the slow-layer energy, while the 64-dimensional corner block accounts for 15.22%. The 21 trivial 1-dimensional blocks contribute negligibly (<1%). This distribution indicates that collective edge permutations and orientations dominate the slow dynamics, providing long-range correlations and quasi-invariant modes, whereas corner configurations contribute more localized slow scaling. Fast-layer energy (λ < 2/3, 128 dimensions) is overwhelmingly from the edge block, confirming its role in rapid randomization. The spectral separation by cubie type underscores a structural origin for the observed 5-layer rational spectrum: slow layers emerge from edge-driven collective behavior, while corner blocks support intermediate quasi-conserved modes.   
    """

    depths, mean_corner, mean_edge = compute_block_distance_expectation(
        corner_idx, edge_idx, num_samples_per_depth=300)
    """
    角块(64d): 扩散几何 → 早期 √k，中后期饱和 ≈3.3
    棱块(144d): 混沌体 → 快速饱和 ≈4.5
    角块（64 dim） ≈ 扩散几何（diffusion geometry，距离随 √k 增长）
    棱块（144 dim） ≈ 混沌体（chaotic bulk，距离快速饱和）
    角块：早期 √k + 中后期饱和 → 有限扩散几何（diffusion in bounded space）
    棱块：极快饱和 → 混沌体（strong mixing, ergodic-like）
    两条曲线在 depth ≈5 后完全分离，棱块饱和值（≈4.5）明显高于角块（≈3.3），说明棱块的“混沌容量”更大，状态空间更“宽广”。
    Expected spectral distance in block subspaces as a function of scramble depth k reveals stark geometric differences. The 64-dimensional corner block exhibits diffusion-like scaling, with distance growing approximately as √k in the early regime (k ≤ 10) before saturating around 3.3–3.5 at higher depths. In contrast, the 144-dimensional edge block displays rapid saturation, reaching ≈4.4–4.5 by depth ≈5 and remaining flat thereafter. This confirms the prediction: corners behave as a diffusion geometry (slow, √k-like spreading in bounded space), while edges constitute a chaotic bulk (fast mixing to equilibrium). The higher saturation level of the edge block (≈4.5 vs ≈3.3) further indicates greater mixing capacity and ergodicity in edge permutations. These scaling laws provide a geometric origin for the spectral stratification: slow layers arise from diffusion-like corner dynamics, fast layers from chaotic edge randomization.
    """

    depths, mean_corner, mean_edge, std_corner, std_edge = compute_inter_state_block_distance(
        corner_idx, edge_idx, num_pairs_per_depth=300)


# ── 3. Bose-Mesner & 代数性质 ─────────────────────────────────────────

def test_bose_mesner(A_micro, generators):
    """Bose-Mesner 代数验证 & 不变子空间检查"""
    success, message, details = verify_bose_mesner(A_micro)
    print("Bose-Mesner:", success)
    print(message)
    """
    Number of distinct eigenvalues (classes): 5
    Success: True — 近似 Bose-Mesner algebra with 5 classes
    但生成元不完全闭合，无法严格归为 association scheme
    """

    w, V = np.linalg.eigh(A_micro)
    results, message = check_invariant_subspaces(A_micro, generators, w, V)
    print(message)
    for lam, res in results.items():
        print(f"λ={lam:.6f}: dim={res['multiplicity']}, invariant={res['is_invariant']}, "
              f"max_err={res['max_error']:.2e}")
    """
    λ=1.0: invariant=True, max_err≈7.4e-08
    其余: invariant=False, max_err≈0.6-0.7
    守恒层(24d)完全不变，其他层只有统计对称性
    ...
    λ = 0.555556 (dim= 96) → 不完全不变 (max err=5.95e-01)
    ------------------------------------------------------------
      λ=0.666667 (dim=32) | Generator  0 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  1 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  3 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  4 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  6 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  7 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator  9 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator 10 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator 12 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator 13 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator 15 → 逃逸误差 = 6.12e-01 > tol
      λ=0.666667 (dim=32) | Generator 16 → 逃逸误差 = 6.12e-01 > tol
    λ = 0.666667 (dim= 32) → 不完全不变 (max err=6.12e-01)
    ------------------------------------------------------------
      λ=0.777778 (dim=44) | Generator  0 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  1 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  2 → 逃逸误差 = 6.40e-01 > tol
      λ=0.777778 (dim=44) | Generator  3 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  4 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  5 → 逃逸误差 = 6.40e-01 > tol
      λ=0.777778 (dim=44) | Generator  6 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  7 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator  8 → 逃逸误差 = 6.40e-01 > tol
      λ=0.777778 (dim=44) | Generator  9 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 10 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 11 → 逃逸误差 = 6.40e-01 > tol
      λ=0.777778 (dim=44) | Generator 12 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 13 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 14 → 逃逸误差 = 6.40e-01 > tol
      λ=0.777778 (dim=44) | Generator 15 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 16 → 逃逸误差 = 4.52e-01 > tol
      λ=0.777778 (dim=44) | Generator 17 → 逃逸误差 = 6.40e-01 > tol
    λ = 0.777778 (dim= 44) → 不完全不变 (max err=6.40e-01)
    ------------------------------------------------------------
    λ = 1.000000 (dim= 24) → 不变 (invariant)
    ------------------------------------------------------------
    """


def test_commutant_and_algebra(A_micro, generators, V_slow):
    """交换子维度 / 慢层代数 / 泄漏"""
    dim_comm = estimate_commutant_dim(generators, n=228, num_samples=1000)
    print("估计 dim Comm(ρ(G)) ≈", dim_comm)
    print("对应不可约表示数 ≈", 228 / dim_comm)
    """
    估计 dim Comm ≈ 336~836
    不是标准 Hamming/Johnson scheme，极可能是 Hecke-type / double coset 统计近似
    """

    max_comm_norm = check_commutativity(generators, V_slow)
    print("max_comm_norm=", max_comm_norm)
    gspan = group_algebra_dim(generators)
    print("dim span {ρ(g)}=", gspan)
    slow_algebra_dimension(generators, V_slow)

    P = V_slow @ V_slow.T.conj()
    dim, msg = compute_span_dim(P, generators)
    print(msg)
    """
    span{P ρ(s) P} 维度 ≈ 11
    dim(𝒜) ≈ 11, dim(𝒜²) ≈ 20 → 非闭合
    """

    leakage_bounds(generators, V_slow)
    """
    最大 slow→fast 泄漏 ‖B‖ = 1.000000154508232
    最大 fast 内部谱 ‖D‖ = 1.0000002249086348
    """


def test_fast_layer_properties(A_micro, V_slow, generators):
    """快层谱半径 / Ramanujan 界 / 伪谱 / Lyapunov"""
    proj_fast = np.eye(228) - V_slow @ V_slow.T.conj()
    A_fast = proj_fast @ A_micro @ proj_fast
    eigvals = np.linalg.eigvals(A_fast)
    rho_f = np.max(np.abs(eigvals))
    print("快层谱半径 ≈", rho_f)  # ≈ 5/9

    d_fast = 12
    ramanujan_fast = 2 * np.sqrt(d_fast - 1) / d_fast
    gap_fast = rho_f - ramanujan_fast
    print(f"Ramanujan 界 (d={d_fast}) = {ramanujan_fast:.6f}")  # 0.552771
    print(f"快层差距 = {gap_fast:.6f}")  # ≈ 0.002785
    """
    快层差距 ≈ 0.003 → 接近最优扩展子 (near-Ramanujan)
    """

    t_mix = np.log(1 / 1e-6) / (-np.log(rho_f))
    print(f"混合时间 tmix(1e-6) ≈ {t_mix:.4f} 步")  # ≈ 23.5

    # 慢/快贡献比
    for rho_s in generators:
        slow_c = np.linalg.norm(V_slow.T @ rho_s @ V_slow)
        fast_c = np.linalg.norm(proj_fast @ rho_s @ proj_fast)
        print(f"Slow/Fast norm ratio: {slow_c / (slow_c + fast_c):.4f}")
    """
    慢贡献 ≈ 46%, 快贡献 ≈ 54%
    """

    analyze_fast_pseudospectrum(A_micro, V_slow, t_max=30)
    estimate_fast_lyapunov(generators, V_slow, T=2000)
    """
    最大 Lyapunov λ ≈ -0.00019 → 渐近稳定，快层近乎保范
    """


def test_shell_decomposition(A_micro):
    """Shell 投影器 & 拟合"""
    shells = shell_projector(samples=1000)
    print("shell statistics:")
    P = {}
    for d, mats in shells.items():
        mats = np.array(mats)
        mean = np.mean(mats, axis=0)
        P[d] = mean
        dev = np.mean([np.linalg.norm(m - mean) for m in mats])
        print(f"shell {d:2d}  samples={len(mats):3d}  deviation={dev:.6f}")
    fit_shell_decomposition(A_micro, P)


def test_double_cosets():
    """采样估算双余类数"""
    num_samples = 1000
    double_cosets = set()
    for _ in range(num_samples):
        g = CubieBase.random_walk(length=50)
        rho_g = g.rho()
        eig = np.sort(np.real(np.linalg.eigvals(rho_g)))
        canonical = tuple(np.round(eig, 4))
        double_cosets.add(canonical)
    print("估算双余类数 (based on sample invariants):", len(double_cosets))
    """
    ≈ 854，远大于 5 → 不是 Gelfand pair
    """


# ── 4. 慢子空间近似 & 群谐函数 ────────────────────────────────────────

def test_slow_approximation(A_micro, w, V):
    """慢子空间截断精度 & 慢坐标演化"""
    mask_slow = w >= 2 / 3 - 1e-8
    V_slow = V[:, mask_slow]
    w_slow = w[mask_slow]
    mask_const = np.abs(w - 1.0) < 1e-8
    V_const = V[:, mask_const]

    state_vector = CubieState.solved().vector
    result = verify_slow_approximation(A_micro, w, V, state_vector, T=100)
    print(f"绝对误差范数: {result['abs_error']:.6e}")
    print(f"相对误差:     {result['rel_error']:.6e}")
    print(f"守恒层投影误差 (T=100): {result['const_error']:.6e}")
    """
    绝对误差范数: 1.51e-06
    相对误差:     6.18e-07
    守恒层投影误差: 1.39e-06
    """

    # 慢坐标轨迹
    z0 = V_slow.T @ state_vector
    T_steps = np.arange(0, 101)
    Z = np.zeros((len(T_steps), len(z0)), dtype=np.complex128)
    for i, t in enumerate(T_steps):
        Z[i] = z0 * (w_slow ** t)
    Z = np.real(Z)
    draw_slow_coordinates(T_steps, Z, n_dims=3)
    """慢坐标轨迹 3D 可视化，Z: (T, dim)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], label='slow coords')
    ax.set_xlabel('z1');
    ax.set_ylabel('z2');
    ax.set_zlabel('z3')
    plt.title('Slow manifold trajectory in 3D projection')
    plt.legend()
    plt.show()


def test_harmonic_slowest(A_micro, w, V):
    """最慢模式 φ_1 的群谐函数误差"""
    phi = V[:, 0]
    lam = w[0]
    values, vals, depths = compute_harmonic_error(phi, lam, n_samples=N_SAMPLES)
    mean_error = np.mean(values)
    std_error = np.std(values)
    max_error = np.max(np.abs(values))
    print(f"\n群谐函数误差统计 (φ_1):")
    print(f"均值: {mean_error:.6f}")
    print(f"标准差: {std_error:.6f}")
    print(f"最大绝对误差: {max_error:.6f}")
    """
    前 24 个 trivial 层: 误差全为 0
    均值/标准差/最大: ≈0 (数值精度内)
    """
    draw_error_histogram(values, title=f'最慢模式 φ_1 的群谐函数误差分布 (n={N_SAMPLES})',
                         xlabel='误差: φ(gx) - λ φ(x)',
                         save_name="最慢模式 φ_1 的群谐函数误差分布.png")


def test_harmonic_79_block(V, w):
    """λ=7/9 块前 10 个模式的群谐函数误差"""
    start = 24
    error_stats = compute_harmonic_error_by_block(V, w, start, n_modes=N_MODES,
                                                  n_samples_per_mode=N_SAMPLES)
    print("\nλ=7/9 块群谐误差总结:")
    for stat in error_stats:
        print(f"模式 {stat['mode']}: λ={stat['lambda']:.6f}, "
              f"均值={stat['mean']:.6f}, std={stat['std']:.6f}, max_abs={stat['max_abs']:.6f}")
        if stat['mode'] - 1 in (1, 7, 8):
            draw_error_histogram(
                stat['values'],
                title=f'模式 {stat["mode"]} (λ={stat["lambda"]:.6f}) 的群谐函数误差分布',
                save_name=f"模式 {stat['mode']} (lam={stat['lambda']:.6f}) 的群谐函数误差分布.png")
    """
    模式 1-8: λ≈7/9, 均值≈0.01, std≈0.17, max_abs=4/9≈0.444 → 准谐函数
    模式 9-10: 误差=0 → 完全对称基
    0.444444 = 4/9 → 误差幅度来源于谱层间距
    """

    # Gram 矩阵验证正交性
    phis = V[:, start:start + 44]
    G = phis.T @ phis
    draw_gram_matrix(G, title='Gram Matrix of 44 Slow Modes (λ = 7/9)',
                     save_name="Gram Matrix of 44 Slow Modes.png")
    """
    严格正交 → 泄漏来自 λ 间的谱间隙，不是向量间非正交性
    """

    # 慢模式嵌入散点
    phi1 = V[:, start]
    phi2 = V[:, start + 1]
    x, y = [], []
    for _ in range(10000):
        s = CubieBase.generate_cubie(length=40)
        v = s.vector
        x.append(phi1 @ v)
        y.append(phi2 @ v)
    """两个模式投影的散点图"""
    plt.scatter(x, y, s=2)
    plt.title("Slow mode embedding")
    plt.show()
    """
    四个离散点 = Z₂ × Z₂ 约束 → λ=7/9 子空间表示高度退化
    """


def test_harmonic_23_block(V, w):
    """λ=2/3 块前 10 个模式的群谐函数误差"""
    start = 24 + 44
    error_stats = compute_harmonic_error_by_block(V, w, start, n_modes=N_MODES,
                                                  n_samples_per_mode=N_SAMPLES)
    print("\nλ=2/3 块群谐误差总结:")
    for stat in error_stats:
        print(f"模式 {stat['mode']}: λ={stat['lambda']:.6f}, "
              f"均值={stat['mean']:.6f}, std={stat['std']:.6f}, max_abs={stat['max_abs']:.6f}")
        if stat['mode'] - 1 in (1, 9):
            draw_error_histogram(
                stat['values'],
                title=f'模式 {stat["mode"]} (λ={stat["lambda"]:.6f}) 的群谐函数误差分布',
                save_name=f"模式 {stat['mode']} (lam={stat['lambda']:.6f}) 的群谐函数误差分布.png")
    """
    λ=2/3 模式: 对每个生成元严格线性本征方向，但不是单个 ρ(s) 的本征空间
    群谐误差 ≈ 0 (采样精度内)
    → ρ(G)-invariant subspace, 不是 ρ(s)-eigenspace
    """


def test_attention_reconstruction(A_micro, w, V):
    """5 层谱投影器重建 A + attention 演化精度"""
    E1 = V[:, np.abs(w - 1.0) < 1e-8] @ V[:, np.abs(w - 1.0) < 1e-8].T.conj()
    E7_9 = V[:, np.abs(w - 7 / 9) < 1e-6] @ V[:, np.abs(w - 7 / 9) < 1e-6].T.conj()
    E5_9 = V[:, np.abs(w - 5 / 9) < 1e-6] @ V[:, np.abs(w - 5 / 9) < 1e-6].T.conj()
    E1_3 = V[:, np.abs(w - 1 / 3) < 1e-6] @ V[:, np.abs(w - 1 / 3) < 1e-6].T.conj()
    E2_3 = V[:, np.abs(w - 2 / 3) < 1e-6] @ V[:, np.abs(w - 2 / 3) < 1e-6].T.conj()

    A_reconstructed = 1.0 * E1 + (7 / 9) * E7_9 + (2 / 3) * E2_3 + (5 / 9) * E5_9 + (1 / 3) * E1_3
    recon_error = np.linalg.norm(A_micro - A_reconstructed)
    print(f"重建误差: {recon_error:.2e}")  # ≈ 1.13e-06

    # attention 演化精度
    M_layers = [E1, E7_9, E2_3, E5_9, E1_3]
    lambda_list = [1.0, 7 / 9, 2 / 3, 5 / 9, 1 / 3]
    from rime.cubieoperator import attention_evolve_exact
    initial_rho = CubieMove.identity().rho()
    x = initial_rho.copy().astype(complex)
    for t in range(5):
        x_exact = A_micro @ x
        x_attn = attention_evolve_exact(x, lambda_list, M_layers)
        error = np.linalg.norm(x_attn - x_exact)
        print(f"T={t}: attention 误差 = {error:.2e}")
        x = x_exact
    """
    重建误差 ≈ 1.1e-06
    T=0..4: 误差从 1.1e-06 降到 5.2e-07 → 精确分解，eigenvalues 是有理数
    """


# ── 5. 退火 & 块谱分解 ────────────────────────────────────────────────

def test_annealing(A_micro):
    """分离时间尺度退火实验"""
    x0 = CubieState.solved().vector
    norm_discrete, norm_continuous, Tf = run_annealing(A_micro, x0)
    draw_annealing(norm_discrete, norm_continuous, Tf)
    """
    连续退火平滑收敛，离散退火体现物理混合步骤
    范数从峰值后迅速下降 → 快层截断与退火相容
    """


def test_cubie_block_spectra(A_micro, eigvals_am, U_am):
    """cp/ep/co/eo 分块谱分析"""
    block_spectra = analyze_cubie_block_spectra(A_micro, eigvals_am, U_am)
    """
    Block 1 (64d, cp): 连续分布，不退化 → irreducible / generic
    Block 2 (144d, ep): 类似 Block 1 → irreducible, generic
    Block 3 (8d, co): λ=2/3 完全退化 → 纯 8D irreducible (Schur 引理)
    Block 4 (12d, eo): λ=1(4) + λ=7/9(8) → reducible = 4×trivial + 8×irreducible
    """


def test_isotypic_decomposition(A_micro, eigvals_am, U_am, rho_moves):
    """随机线性组合 + isotypic 分解"""
    samples = list(CubieMove.prim_moves.values()) + list(CubieMove.slice_moves().values())
    A = sum(c_i * mv_i.rho() for mv_i, c_i in zip(samples, np.random.randn(len(samples))))
    B = sum(c_i * mv_i.rho() for mv_i, c_i in zip(samples, np.random.randn(len(samples))))
    C = A + 1j * B
    eigvals, U = np.linalg.eig(C)
    blocks = detect_blocks(samples, U)
    sizes = [len(b) for b in blocks]
    print("Block sizes:", sorted(sizes))
    print("Number of blocks:", len(blocks))

    big_block = max(blocks, key=len)
    print("big_block size:", len(big_block))
    multiplicities = split_isotypic_block(samples, U, big_block, tol=1e-8)
    print("Block multiplicities:", multiplicities)

    projections = construct_projection_operators(U, blocks)
    rho_matrices = [mv.rho() for mv in samples]
    for rho in rho_matrices:
        block_traces = [np.trace(P @ rho @ P) for P in projections]
        print(block_traces)


# ── main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("═══ 1. 基础设置 & 块检测 ═══")
    """计算 A_micro、块检测、corner/edge 索引"""
    prim_list18 = list(CubieMove.prim_moves.values())
    rho_moves = [m.rho() for m in prim_list18]
    A_micro = sum(rho_moves) / len(rho_moves)

    eigvals, U = np.linalg.eig(A_micro)
    blocks = detect_blocks(list(CubieMove.prim_moves().values()), U)  # 不依赖顺序

    idx = np.argsort(-np.abs(eigvals.real))
    U_am = U[:, idx]
    eigvals_am = eigvals[idx]

    corner_idx = blocks[0]  # size 64
    edge_idx = blocks[1]  # size 144

    # for b in blocks:
    #     if len(b) == 64:
    #         corner_idx = b
    #     elif len(b) == 144:
    #         edge_idx = b

    generators = rho_moves

    test_move_composition()
    test_block_detection(A_micro, U_am)

    print("\n╔══ 2. 谱结构 (5 层有理谱 k/9) ══╗")
    w, V = np.linalg.eigh(A_micro)
    test_spectral_layers(A_micro, generators)
    test_block_spectrum(A_micro, V, blocks, corner_idx, edge_idx)

    print("\n╔══ 3. Bose-Mesner & 代数性质 ══╗")
    test_bose_mesner(A_micro, generators)
    mask_slow = w >= 2 / 3 - 1e-8
    V_slow = V[:, mask_slow]
    test_commutant_and_algebra(A_micro, generators, V_slow)
    test_fast_layer_properties(A_micro, V_slow, generators)
    test_shell_decomposition(A_micro)
    test_double_cosets()

    print("\n╔══ 4. 慢子空间近似 & 群谐函数 ══╗")
    test_slow_approximation(A_micro, w, V)
    test_harmonic_slowest(A_micro, w, V)
    test_harmonic_79_block(V, w)
    test_harmonic_23_block(V, w)
    test_attention_reconstruction(A_micro, w, V)

    print("\n╔══ 5. 退火 & 块谱分解 ══╗")
    test_annealing(A_micro)
    test_cubie_block_spectra(A_micro, eigvals_am, U_am)
    test_isotypic_decomposition(A_micro, eigvals_am, U_am, generators)
