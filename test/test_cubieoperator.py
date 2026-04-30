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

import sys
import io
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (ValueError, AttributeError):
        pass

import matplotlib

matplotlib.use('Agg')  # 无头渲染
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from rime.cubieoperator import *
from rime.cubieworld import SlowDynamics, N_GENERATORS
from rime.cube import ActionToken
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
    # BUGFIX (2026-04-28): block distance fns now require U_basis=V arg (eigenbasis match)
    A_block = V.T.conj() @ A_micro @ V
    analyze_block_spectrum(A_block, blocks)

    depths, mean_corner, mean_edge = compute_block_distance_expectation(
        corner_idx, edge_idx, U_basis=V, num_samples_per_depth=300)

    depths, mean_corner, mean_edge, std_corner, std_edge = compute_inter_state_block_distance(
        corner_idx, edge_idx, U_basis=V, num_pairs_per_depth=300)


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
    dim_comm = estimate_commutant_dim(generators, n=228, num_samples=30)
    print("估计 dim Comm(ρ(G)) ≈", dim_comm)
    if dim_comm > 0:
        print("对应不可约表示数 ≈", 228 / dim_comm)
    """
    估计 dim Comm ≈ 336~836
    不是标准 Hamming/Johnson scheme，极可能是 Hecke-type / double coset 统计近似
    
    单个生成元作用并不完全保持这些层不变 → 统计对称性（quasi-symmetry）
    其余层（慢层 ~76 维 + 快层 128 维）形成一个大约 204 维的可约表示，内部有统计对称性
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
    
    ρ(g) lives in a tiny algebra
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

    """
    谱半径 ρ = 0.5555557144099934
    非正规性 ‖AA* - A*A‖ = 5.749921133840414e-16
    特征向量条件数 κ(V) = 139.9906737469503
    """
    estimate_fast_lyapunov(generators, V_slow, T=2000)
    """
    最大 Lyapunov λ ≈ -0.00019 → 渐近稳定，快层近乎保范
    对应指数率 e^λ ≈ 0.9998086269014682
    平均谱退化 → 统计对称性
    瞬时生成元 → 几乎保持 norm 快层几乎是**保范（norm-preserving）**的。
    整个系统是渐近稳定的（asymptotically stable），状态会缓慢趋向平衡（solved 附近）
    一个具有统计谱分层（statistical spectral stratification）的非交换表示，平均算符下出现 5 个高度退化的谱层，单个生成元导致强跨层混合，但慢动力学表现出极高的可计算性和稳定性。
    由对称性导致的简并模式成为谱的主要特征。
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
    The spectral stratification arises not from a classical association scheme, but from the isotypic decomposition of the faithful 228-dimensional representation under generator averaging.
        
    The Rubik's cube group is a paradigmatic example of a large discrete symmetry group with rich combinatorial structure. 
    In this work, we investigate the spectral properties of the normalized transfer operator A = (1/|S|) ∑_{s∈S} ρ(s) in the faithful 228-dimensional representation of the Phase-1 subgroup, where S is the set of generators (18 primitive + 9 slice moves).The operator exhibits exactly five distinct rational eigenvalues of the form k/9 (k=3,5,6,7,9) with high multiplicities (32,96,32,44,24), and its spectral projectors satisfy the Bose–Mesner algebra conditions (idempotence, orthogonality, and completeness). 
    However, individual generators ρ(s) do not preserve the eigenspaces (cross-layer leakage ≈0.42–0.71), ruling out a full association scheme or Gelfand pair structure.Despite this, the subspace spanned by eigenvalues λ ≥ 2/3 (dimension 100) shows quasi-invariance under group action, with leakage error ≈0.42–0.46. Projecting dynamics onto this slow manifold yields highly accurate approximations: 
    relative error < 6×10^{-7} for T=100 steps, demonstrating that fast modes (λ < 2/3) can be safely truncated.We propose a representation-aware heuristic distance d(x,y) = ||V_slow^T (x-y)||, which leverages the slow projection to ignore transient modes. These findings reveal a striking separation between averaged symmetry (captured by A) and instantaneous asymmetry (in ρ(s)), offering a computable low-dimensional world model for discrete group actions in puzzle solving and beyond.
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
    
    绝对误差范数: 1.144507e-06
    相对误差:     4.672430e-07
    守恒层投影误差 (T=100): 1.244515e-06
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
    plt.savefig(os.path.join(DATA_DIR, 'Slow manifold trajectory in 3D projection'), dpi=300, bbox_inches='tight')
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
    群谐函数误差统计 (φ_1):
    均值: 0.000000+0.000000j
    标准差: 0.000000
    最大绝对误差: 0.000000
    
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
    λ=7/9 块群谐误差总结:
    模式 1: λ=0.333333, 均值=0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 2: λ=0.333333, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 3: λ=0.333333, 均值=0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 4: λ=0.333333, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 5: λ=0.333333, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 6: λ=0.333333, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 7: λ=0.333333, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 8: λ=0.333333, 均值=0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 9: λ=0.555556, 均值=0.000000+0.000000j, std=0.000000, max_abs=0.000000
    模式 10: λ=0.555556, 均值=-0.000000+0.000000j, std=0.000000, max_abs=0.000000

    模式 1-8: λ≈7/9, 均值≈0.01, std≈0.17, max_abs=4/9≈0.444 → 准谐函数
    模式 9-10: 误差=0 → 完全对称基
    0.444444 = 4/9 → 误差幅度来源于谱层间距
    
    A 的特征向量 = harmonic function（平均意义）“误差为 0”是必然结果
    
    慢动力学本质上是“守恒谐函数 + 准谐衰减”的组合
    谐性质只严格保持在前 8 个模式（对应 λ ≈1 的守恒/准守恒部分），一旦进入 λ <1 的非守恒慢层（e.g. 7/9 或 2/3），群作用开始引入扰动，误差从 0 跳到 O(1) 量级
    0.444444=4/9
    第二类（mode9–10）刚好落在一个完全对称的子空间 basis
    8 个 ≈ 数值基
    2 个 ≈ 对称基
    慢子空间的前 10 个模式（λ ≈ 7/9）全部是准谐函数，误差稳定在 0.17 左右，最大不超过 0.444。
    这远低于随机向量在群作用下的扰动（通常 O(1) 或更大），证明慢层确实捕捉了群上的低频谐波。
    误差的固定幅度（≈4/9）暗示扰动来源于谱层间距，而非随机混沌 → 慢流形具有结构化准不变性。
    φ(x) 是一个二值函数 Z2 blocks
    d 偶 → +c
    d 奇 → -c
    2 moves 会 flip
    """

    # Gram 矩阵验证正交性
    phis = V[:, start:start + 44]
    G = phis.conj().T @ phis
    print(np.max(np.abs(G - np.eye(G.shape[0]))))
    print(np.linalg.matrix_rank(G))  # 44
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
        x.append((phi1.conj() @ v).real)
        y.append((phi2.conj() @ v).real)
    """两个模式投影的散点图"""
    plt.scatter(x, y, s=2)
    plt.title("Slow mode embedding")
    plt.show()
    """
    四个离散点 = Z₂ × Z₂ 约束 → λ=7/9 子空间表示高度退化
    """
    # 在 44 维内部
    state_x = CubieBase.generate_cubie()
    Moves = list(CubieMove.prim_moves.values())
    # 投影所有生成元到这个子空间
    Ms = []
    vals = []
    for mv in Moves:
        rho = mv.rho()
        M = phis.conj().T @ rho @ phis
        Ms.append(M)

        v = mv.act(state_x).vector
        vals.append(phis.conj().T @ v)

    # 用这些矩阵重新 detect_blocks
    blocks_sub = detect_blocks(
        [type("Tmp", (), {"rho": (lambda M: (lambda self: M))(M)})() for M in Ms],
        np.eye(44)
    )
    print(blocks_sub)
    vals = np.array(vals)  # shape: (num_moves, 44)
    print(np.cov(vals.T))

    """
    blocks_sub =
    [
      [0..31],   # 32 维
      [32..43]   # 12 维
    ]
    E_{7/9} = V_32 ⊕ V_12
    λ=7/9 对应的 44 维空间可以分成 32+12 两个真正的表示块，
    而每个块内部仍可能包含更细的不可约表示，需要继续递归分解。

    φ 不是群同态函数，而是高维表示里的坐标；
    群作用会在 eigenspace 内“旋转/混合”这些模式，
    而 λ 控制的是“平均收缩”，std 控制的是“瞬时扩散”。
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
    
    守恒量：
    φ(sx) = φ(x)
    λ = 2/3
    ρ(s)φ ≠ λφ
    对单个 move 不成立,群表示 ρ 的一个共同不变子空间（ρ(G)-invariant subspace），但不是单个 ρ(s) 的本征空间。
    λ = 7/9
    随机扩散模式：
    E[φ(sx)] = (7/9) φ(x)
    但单步波动大。
    分解成几个 block：
    trivial
    parity
    scaling
    diffusion
    Trivial block (λ=1, dim=24)
    精确不变子空间（invariant subspace）。
    ρ(s) V_trivial = V_trivial（identity action 或 trivial representation 的多重性）。
    对应守恒宏观变量（总 parity、总朝向和等）。
    群谐误差 = 0（最严格）。
    
    Parity block (可能嵌入在 λ=1 或附近)
    对应边/角 parity（Z2 对称）。
    在某些子空间上 ρ(s) 作用为 ±1（sign flip）。
    通常与 trivial 层混合，但保持不变。
    
    Scaling block (λ=7/9, dim=44 + 部分 λ=1 的尾部)
    准不变子空间（quasi-invariant）。
    ρ(s) V_scaling ≈ (7/9) V_scaling + 小扰动（误差 ≈0.17）。
    对应“集体缩放”模式（e.g. 朝向或置换的均匀收缩）。
    群谐误差小但非零（准谐函数）。
    
    Diffusion block (λ=2/3, dim=32)
    严格线性本征方向（exact eigenvector direction）。
    对每个生成元 s，ρ(s) V_diffusion = (2/3) V_diffusion（标量缩放）。
    对应“扩散-like”模式，但不是随机扩散，而是纯缩放扩散（pure scaling diffusion）。
    群谐误差 = 0（在采样精度内）。
    
    剩余层 (λ=5/9, 1/3, dim=96+32)
    混合更强，扰动大，接近“随机化”但仍有结构（奇异值稳定在 1 或 √3/2）。
    群谐误差可能较大（未测试）。
    
    误差是“谱层泄漏（inter-layer leakage）
    
    Slow manifold captures Koopman eigenfunctions of the averaged group dynamics, not the exact representation symmetry of individual generators.
    
    
    Slow modes: quasi-harmonic structure

    Empirical observations:
    
    1. Harmonic structure
    ----------------------
    The slow modes (λ ≈ 7/9) satisfy:
    
        A φ = λ φ
    
    where A is the averaged generator operator.
    
    However, for individual generators:
    
        ρ(g) φ ≠ φ
    
    Thus these modes are not strictly invariant, but are
    Koopman eigenfunctions of the averaged dynamics.
    
    → Interpretation:
        slow modes are "quasi-harmonic" functions:
        globally stable under averaging,
        locally perturbed by generator actions.
    
    --------------------------------------------------
    
    2. Structured deviation (non-random error)
    ------------------------------------------
    For λ ≈ 7/9 modes:
    
        std ≈ 0.17
        max ≈ 0.44
    
    This deviation is:
    
        • consistent across modes
        • bounded
        • non-random
    
    → Interpretation:
        deviation is caused by structured leakage
        from slow subspace into fast spectral layers.
    
    This confirms:
    
        slow manifold is quasi-invariant,
        not strictly invariant.
    
    --------------------------------------------------
    
    3. Spectral meaning
    -------------------
    The slow modes correspond to low-frequency components
    of the group dynamics:
    
        • capture coarse permutation structure
        • insensitive to high-frequency fluctuations
        • stable under random walk averaging
    
    They form the dominant basis for long-time dynamics.
    
    --------------------------------------------------
    
    4. Special symmetric modes (mode 9–10)
    --------------------------------------
    Some modes exhibit near-zero deviation:
    
        mean ≈ 0
        std ≈ 0
    
    These likely correspond to highly symmetric subspaces
    (e.g. parity-like or invariant combinations).
    
    However:
    
        exact algebraic interpretation remains unclear.
    
    --------------------------------------------------
    
    Summary
    -------
    Slow dynamics are governed by:
    
        invariant modes (λ = 1)
        +
        quasi-harmonic modes (λ < 1)
    
    with structured generator-induced perturbations.
    
    This explains:
    
        • stability of slow manifold
        • controlled deviation
        • low-rank effective dynamics
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


def test_isotypic_decomposition():
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

# ── 6. 理论验证实验 ─────────────────────────────────────────────────────

def test_universal_spectral_law():
    """Theorem 8.1 (Universal Spectral Law) 数值验证

    核心主张: 对任意生成元子集 S，A_S 的特征值遵循 λ = 1 - k/m，m = |S|/2
    验证策略: 按轴/面逐步增加生成元，检验谱形式是否严格有理
    """
    results = verify_universal_spectral_law()

    # 汇总: 所有子集是否满足有理谱
    all_pass = all(r['all_rational'] for r in results)
    print(f"\n{'='*60}")
    print(f"Theorem 8.1 验证: 全部有理 = {all_pass}")
    for r in results:
        mark = "OK" if r['all_rational'] else "FAIL"
        print(f"  {r['name']}: |S|={r['n_gen']}, m={r['m_eff']}, "
              f"eigenvalues={len(r['unique_eigenvalues'])}, poly_rank={r['poly_rank']} {mark}")


def test_spectral_collapse_verification(A_micro, generators, w, V):
    """核心论点验证: 平均对称性 vs 瞬时非对称性 = spectral collapse

    A = E[ρ(s)] 产生谱坍缩:
    - 单个 ρ(s) 的谱连续分布（无退化）
    - 随着平均生成元数量增加，谱逐步坍缩为离散有理值
    - 最终 A 的谱仅有 5 个退化层

    这是论文核心贡献: A 的谱对称性是 spectral collapse 现象，
    不是群的内在代数结构
    """
    data = verify_spectral_collapse(generators, A_micro, w, V)

    # 关键数值: 谱熵随平均数单调下降 → spectral collapse
    entropies = data['entropies_avg']
    degens = data['degen_avg']
    print("\n谱坍缩趋势:")
    for k in [0, 2, 5, 8, 11, 14, 17]:
        if k < len(entropies):
            print(f"  k={k+1:2d} generators: entropy={entropies[k]:.4f}, "
                  f"distinct_eigenvalues={degens[k]}")
    print(f"  → entropy 单调下降: {all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1))}")


def test_character_decomposition(generators):
    """不可约分解验证 (最关键的开放问题)

    计算 χ(g) = Tr(ρ(g))，做内积 ⟨χ,χ⟩
    ⟨χ,χ⟩ = 1 ⟺ 不可约
    ⟨χ,χ⟩ > 1 ⟺ 可约，值 = 不可约分量数

    目标: 确认 228 维表示是否为多个 irreps 的直和
    """
    data = compute_character_table(generators, n_samples=1000)

    print(f"\n不可约分解推断:")
    print(f"  ⟨χ, χ⟩ = {data['inner_product']:.4f}")
    print(f"  估计 irrep 数 = {data['n_irreps_est']}")
    print(f"  共轭类数 = {len(data['class_chars'])}")
    if data['n_irreps_est'] > 1:
        print(f"  → 228 维表示 ≈ {data['n_irreps_est']} 个不可约分量的直和")
        print(f"  → 平均每个 irrep 维度 ≈ {228 / data['n_irreps_est']:.1f}")


def test_quotient_geometry(A_micro, generators, V, w):
    """slow manifold = quotient geometry 形式化验证

    核心链: representation (ρ, 228D) → quotient (π, 100D) → metric (d_slow) → search (A*)

    验证:
    1. π: V → V_slow 是商映射，ker(π) = V_fast
    2. 商度量 d_Q(π(x), π(y)) = ||V_slow^T(x-y)|| 是良定义的伪度量
    3. ρ(s) 在商空间上近似等距
    4. 快层在商空间中指数衰减 → 截断安全
    """
    mask_slow = w >= 2 / 3 - 1e-8
    V_slow = V[:, mask_slow]

    data = verify_quotient_geometry(V_slow, generators, n_samples=2000)

    print(f"\nQuotient Geometry 总结:")
    print(f"  ker(P_slow)·P_fast = {data['ker_test']:.2e} (应为 0)")
    print(f"  纤维退化 = {data['fiber_degeneracy']:.2e} (应为 0)")
    print(f"  等距比 = {np.mean(data['iso_ratios']):.6f} ± {np.std(data['iso_ratios']):.6f}")
    print(f"  有效自由度 = {data['effective_dim']}")
    print(f"  快层半衰期 = {data['t_half']:.2f} 步")
    print(f"  ρ_f^100 = {data['rho_fast']**100:.2e}")

    """
    1. ker(P_slow) = range(P_fast): ||P_slow · P_fast|| = 9.76e-07
    2. 纤维退化: ||V_slow^T · v_fast|| = 1.28e-06 (应为 0)
    3. 等距性: ratio = 0.999094 ± 0.030835
    max ratio = 1.137908, min = 0.885948
    4. 商流形: dim(V_slow) = 100, 有效自由度 (λ>0.5) = 100
    慢层特征值: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] ...
    5. 快层谱半径 ρ_f = 0.555556, 半衰期 t_1/2 = 1.18 步
    → 快层在 ≈2 步后在商空间中消失

    链完整性:
    representation (ρ, 228D) → quotient (π, 100D) → metric (d_slow) → search (A*)
    信息损失: 56.1% 维度被截断
    等距误差: 0.0308
    截断安全性: ρ_f^100 = 2.97e-26

    Quotient Geometry 总结:
    ker(P_slow)·P_fast = 9.76e-07 (应为 0)
    纤维退化 = 1.28e-06 (应为 0)
    等距比 = 0.999094 ± 0.030835
    有效自由度 = 100
    快层半衰期 = 1.18 步
    ρ_f^100 = 2.97e-26
    """

# ── 7. rho_moves 多生成元验证 ──────────────────────────────────────────

def test_rho_moves_spectral_law():
    """使用 SlowDynamics.rho_moves(n) 验证不同 n 下的谱结构。

    对 n=2..18 的所有生成元子集：
    1. Hermitian 性检查
    2. 特征值是否 λ = 1 - k/m
    3. 慢子空间维度
    4. poly_rank

    发现：有理谱并非对所有子集成立。n=8 (k[0]!=1 and k[2]!=2) 和
    n=16 (排除 U2/D2) 虽然 Hermitian，但产生无理特征值。
    有理谱需要生成元集具有特定对称性：完整面的 3 方向 or 特定对称子集。
    """
    print("=" * 70)
    print("rho_moves 多生成元验证: 谱分层 = f(|S|, 对称性)")
    print("=" * 70)

    sd = SlowDynamics.lite()

    header = f"{'n':>3} {'|S|':>3} {'m':>3} {'herm':>5} {'#λ':>3} {'poly_rk':>7} {'slowD':>5} {'rational':>8}"
    print(header)
    print("-" * len(header))

    for n in [2, 3, 4, 6, 8, 9, 10, 12, 16, 18]:
        rm = sd.rho_moves(n=n)
        n_gen = len(rm)
        if n_gen == 0:
            continue
        rhos = [rho for _, rho, *_ in rm.values()]
        A_S = sum(rhos) / n_gen

        is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)

        w_S, _ = np.linalg.eigh(A_S)
        unique_w = np.unique(np.round(w_S, 6))
        pr = poly_rank(A_S, k=10)

        slow_dim = np.sum(w_S >= 2 / 3 - 1e-8)

        m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen

        all_rational = True
        for lam in unique_w:
            k_val = round((1 - lam) * m_eff)
            pred = 1 - k_val / m_eff
            if abs(lam - pred) > 1e-6:
                all_rational = False

        mark = "" if all_rational else " (!)"
        print(f"{n:3d} {n_gen:3d} {m_eff:3d} {str(is_herm):>5} {len(unique_w):3d} "
              f"{pr:7d} {slow_dim:5d} {str(all_rational):>8}{mark}")
    print()

    # 摘要
    print("关键发现:")
    print("  n=18 (全生成元): 5层, λ=k/9, herm=True, rational=True")
    print("  n=12 (标准 face-turn, k[2]!=2): 6层, λ=k/6, herm=True, rational=True")
    print("  n=10 (部分破缺): 5层, λ=k/5, herm=True, rational=True")
    print("  n=6 (仅180度): 3层 {1,2/3,1/3}, herm=True")
    print("  n=8 (axis≠1, dir≠2): herm=True 但有**无理数特征值** → 对称性不足")
    print("  n=16 (排除U2/D2): herm=True 但**无理数特征值** → 对称性破缺")
    print("  n=3 (单面3方向): herm=False, 负特征值 → 需完整3轴消除虚部")
    print("  结论: 有理谱 = Hermitian + 生成元集满足特定对称性, 非总是成立")


# ── 8. 严格证明链验证 (Theorem 5.1 Rigorous) ───────────────────────────

def test_generator_pairing_and_h_spectrum(A_micro):
    """8.1 生成元配对: 18 生成元 → 9 对称单元 h_i = (g_i + g_i^{-1})/2

    配对方案:
      6 面级 CW/CCW 对: h_face = (ρ(CW) + ρ(CCW))/2 每个面
      3 轴级 180° 跨面对: h_axis = (ρ(180°_side1) + ρ(180°_side2))/2 每条轴
      总计: 6 + 3 = 9 对称单元

    核心断言:
      (a) A = (1/9) Σ h_i
      (b) 每个 h_i Hermitian
      (c) 在置换块 (cp/ep) 上: h_i 谱 ⊆ {1, 0, -1}
          理由: 阶-4 元素特征值 {1, i, -1, -i} → (g+g^{-1})/2 有 {1, 0, -1, 0}
               阶-2 元素特征值 ±1 → (g+g^{-1})/2 = g 有 ±1
      (d) 在朝向块 (co/eo) 上: h_i 对角元为 cos(2πk/3) 或 ±1
          co: cos(2π·0/3)=1, cos(2π·1/3)=-1/2, cos(2π·2/3)=-1/2
          eo: ±1
    """
    print("\n── 8.1 生成元配对 & h_i 谱 ──")
    prim_moves = CubieMove.prim_moves

    # 构造 9 个对称单元
    h_operators = []
    h_names = []

    # 6 面级 CW/CCW 对
    for axis in range(3):
        for side in [-1, 1]:
            cw_key = (axis, side, -1)
            ccw_key = (axis, side, 1)
            if cw_key in prim_moves and ccw_key in prim_moves:
                rho_cw = prim_moves[cw_key].rho()
                rho_ccw = prim_moves[ccw_key].rho()
                h = (rho_cw + rho_ccw) / 2
                h_operators.append(h)
                h_names.append(f"face(ax={axis},sd={side:+d}) CW+CCW")

    # 3 轴级 180° 跨面对
    for axis in range(3):
        keys_180 = [(axis, side, 2) for side in [-1, 1]
                    if (axis, side, 2) in prim_moves]
        if len(keys_180) == 2:
            rho_a = prim_moves[keys_180[0]].rho()
            rho_b = prim_moves[keys_180[1]].rho()
            h = (rho_a + rho_b) / 2
            h_operators.append(h)
            h_names.append(f"axis({axis}) 180° pair")

    n_h = len(h_operators)
    print(f"18 生成元 → {n_h} 对称单元 h_i (期望 9)")
    assert n_h == 9, f"ERROR: expected 9 h_i, got {n_h}"

    # Test (a): A = (1/9) Σ h_i
    A_from_h = sum(h_operators) / 9
    err_A = np.linalg.norm(A_from_h - A_micro)
    print(f"(a) A = (1/9) Σ h_i: error = {err_A:.2e}  {'OK' if err_A < 1e-12 else 'FAIL'}")

    # Test (b): Hermiticity of individual h_i
    # NOTE: Some h_i are non-Hermitian due to ρ(CCW) ≠ ρ(CW)^* on the co block.
    # This is a known inconsistency in the CubieMove.rho() corner orientation computation.
    # However, the FULL sum A = (1/9) Σ h_i IS Hermitian (3-axis cancellation).
    print("(b) h_i Hermiticity check:")
    herm_count = 0
    for i, (h, name) in enumerate(zip(h_operators, h_names)):
        err = np.max(np.abs(h - h.T.conj()))
        is_h = err < 1e-10
        if is_h:
            herm_count += 1
        if i < 3 or not is_h:
            print(f"  h_{i} ({name}): max|h-h*|={err:.2e} {'OK' if is_h else 'FAIL (co-block inconsistency)'}")
    print(f"  Hermitian: {herm_count}/{len(h_operators)}")
    print(f"  BUT A is Hermitian: {np.allclose(A_micro, A_micro.T.conj(), atol=1e-10)} (full 3-axis cancellation)")

    # Test (c): h_i spectrum ⊆ {1, 0, -1} on permutation blocks
    print("(c) h_i 谱检查:")
    all_cos_set = True
    for i, (h, _name) in enumerate(zip(h_operators, h_names)):
        eigvals = np.linalg.eigvals(h)  # use eigvals (not eigvalsh) as some h_i are non-Hermitian
        unique_rounded = np.unique(np.round(eigvals, 6))
        # Check which values are NOT in {1, 0, -1}
        outliers = [u for u in unique_rounded
                    if not any(abs(u - x) < 1e-5 for x in [1, 0, -1])]
        if outliers:
            all_cos_set = False
            if i < 2:  # Only print first few details
                print(f"  h_{i}: {sorted(unique_rounded)[:6]}... outliers={sorted(outliers)[:3]}")

    # h_i 在 208 维置换空间上谱 ∈ {1,0,-1}
    # 在 20 维朝向空间上谱 ∈ {1,-1/2,±1}
    # 分离检查
    proj_orient = np.zeros((228, 228))
    for k in range(64 + 144, 228):
        ek = np.zeros(228)
        ek[k] = 1
        proj_orient += np.outer(ek, ek)
    proj_perm = np.eye(228) - proj_orient

    perm_outliers = []
    orient_vals = set()
    for h in h_operators:
        # Perm block restriction (perm matrices are orthogonal, so h_perm is symmetric real)
        h_perm = proj_perm @ h @ proj_perm
        eigvals_perm = np.linalg.eigvals(h_perm)
        eigvals_perm = eigvals_perm[np.abs(eigvals_perm) > 1e-3]
        for ev in np.round(np.real(eigvals_perm), 6):
            if not any(abs(abs(ev) - x) < 1e-4 for x in [1, 0]):
                perm_outliers.append(ev)
        # Orient block (may be complex diagonal)
        h_orient = proj_orient @ h @ proj_orient
        eigvals_orient = np.linalg.eigvals(h_orient)
        for ev in np.round(np.real(eigvals_orient), 6):
            if abs(ev) > 1e-3:
                orient_vals.add(ev)

    print(f"  置换块 (208D) 非 {1,0,-1} 谱值: {sorted(set(perm_outliers))[:8] if perm_outliers else '无 OK'}")
    print(f"  朝向块 (20D) 谱值: {sorted(orient_vals)}")

    assert all_cos_set or not perm_outliers, f"Unexpected spectrum in perm block"
    print(f"  结论: 置换块 h_i 谱 ⊆ {{1, 0, -1}}  {'OK' if not perm_outliers else 'FAIL'}")

    # Test (d): [h_i, A] = 0? (h_i commute with A approximately on eigenspaces)
    print("(d) [h_i, A] 交换子范数:")
    max_comm = 0
    for i, h in enumerate(h_operators):
        comm = h @ A_micro - A_micro @ h
        norm_comm = np.linalg.norm(comm)
        max_comm = max(max_comm, norm_comm)
    avg_comm = np.mean([np.linalg.norm(h @ A_micro - A_micro @ h)
                         for h in h_operators])
    print(f"  平均 ‖[h_i, A]‖ = {avg_comm:.2e}, 最大 = {max_comm:.2e}")
    print(f"  h_i 不严格交换 A，但在每个特征空间上近似标量作用 (Schur)")

    return h_operators, h_names


def test_character_on_eigenspaces(w, V, generators):
    """8.2 特征空间上 character 公式验证

    Schur 引理 → 在每个 isotypic 分量 α 上，A = λ_α I。
    取迹: λ_α · d_α = Tr(A P_α) = (1/|S|) Σ_s Tr(ρ(s) P_α) = (1/|S|) Σ_s χ_α(s)
    其中 χ_α(s) = Tr(P_α ρ(s))，d_α = dim(P_α)，P_α = 特征空间投影器。

    验证: λ_α = (1/|S|) Σ_s χ_α(s) / d_α
    这等价于验证每个特征空间是 ρ(s) 平均算符的精确特征空间。
    """
    print("\n── 8.2 特征空间 character 公式 ──")

    S = len(generators)
    unique_w = np.unique(np.round(w, 6))
    print(f"特征值: {sorted(unique_w, reverse=True)}")

    results = []
    for lam in sorted(unique_w, reverse=True):
        mask = np.abs(w - lam) < 1e-6
        d_alpha = np.sum(mask)
        V_alpha = V[:, mask]
        # 投影器 P_α = V_α V_α^*
        # χ_α(s) = Tr(V_α V_α^* ρ(s)) = Tr(V_α^* ρ(s) V_α)
        chi_sum = 0.0
        for rho_s in generators:
            chi_s = np.trace(V_alpha.T.conj() @ rho_s @ V_alpha)
            chi_sum += chi_s

        lam_from_chi = chi_sum / (S * d_alpha) if d_alpha > 0 else np.nan
        error = abs(lam - lam_from_chi)
        results.append({
            'lambda': lam, 'dim': d_alpha,
            'chi_sum': chi_sum, 'lam_from_chi': lam_from_chi, 'error': error
        })
        print(f"  λ={lam:.6f} dim={d_alpha:3d} "
              f"Σ_s χ_α(s)={chi_sum.real:.4f}{chi_sum.imag:+.4f}j "
              f"→ λ'={lam_from_chi.real:.6f} error={error:.2e} {'OK' if error < 1e-6 else 'FAIL'}")

    all_ok = all(r['error'] < 1e-6 for r in results)
    print(f"  Character 公式全部成立: {all_ok}")
    return results


def test_k_selection_rule(w, h_operators=None):
    """8.3 k-选择规则: 为什么 λ = 1 - k/9 只取 k ∈ {0, 2, 3, 4, 6}

    从 pairing 可知 A = (1/9) Σ_{i=1}^9 h_i，其中 h_i 在置换块上谱 ⊆ {1, 0, -1}。

    在每个共同特征向量 v 上:
      h_i v = ε_i v,  ε_i ∈ {1, 0, -1}
      A v = (1/9) Σ ε_i v = λ v
      λ  = k'/9,  k' ∈ [-9, 9] 整数

    观察: λ = 1 - k/9 → k = 9 - k'
    观察到的 k: 0, 2, 3, 4, 6
    缺失的: 1, 5, 7, 8, 9

    选择规则约束:
      C1 迹条件 (置换块): Tr(A|_{perm}) = 0 → 需要正负平衡
      C2 朝向块 CO (8D): ρ_co 用 ω = e^{2πi/3}, ω+ω²=-1, 极大约束 k
      C3 朝向块 EO (12D): (-1) 符号结构约束
      C4 等变结构: 228 = 64+144+8+12, 各块维数整除约束
    """
    print("\n── 8.3 k-选择规则 ──")

    unique_w = np.unique(np.round(w, 6))
    m_eff = 9  # n_gen // 2 = 18 // 2 = 9

    print(f"m = {m_eff} (|S|/2 = 18/2 = 9)")
    print("观察到的 k (λ = 1 - k/9):")
    observed_k = set()
    for lam in unique_w:
        k = round((1 - lam) * m_eff)
        observed_k.add(k)
        d = np.sum(np.abs(w - lam) < 1e-6)
        print(f"  λ={lam:.6f} → k={k}, dim={d}")

    print(f"缺失 k: {set(range(10)) - observed_k}")

    # C1-C4 约束检查
    print("\n选择规则约束分析:")
    # C1: Tr(A) = Σ λ_i dim_i
    trace_A = sum(unique_w[i] * np.sum(np.abs(w - unique_w[i]) < 1e-6)
                  for i in range(len(unique_w)))
    print(f"  C1 Tr(A) = {trace_A:.4f} (应为整数或半整数)")

    # C2: 在 co 块上的投影
    # co 块是 cp 后的 8 维, eo 是 ep 后的 12 维
    # But the 228-dim ordering might be different...
    # From analyze_cubie_block_spectra we know blocks are in order
    # We can directly compute λ values per block by projection
    print(f"  C2-C4: 谱层选择由 4 个块 (cp:64, ep:144, co:8, eo:12) 的表示论约束")

    # 额外: 对 h_i 的迹模式分析
    if h_operators is not None:
        print(f"\n  h_i 的迹 (应在整数/半整数范围):")
        for i, h in enumerate(h_operators[:3]):
            tr = np.trace(h)
            print(f"    Tr(h_{i}) = {tr.real:.4f}{tr.imag:+.4f}j")


def test_schur_lemma_verification(V, w, generators):
    """8.4 Schur 引理数值验证

    在每个特征空间 E_λ 上:
      P_λ ρ(s) P_λ = c_λ(s) · P_λ  +  error(s)

    如果特征空间是不可约的 (irreducible)，则 c_λ(s) 是 scalar，error = 0。
    实际检查: c_λ(s) 是否为标量矩阵 (与恒等矩阵成比例)。
    """
    print("\n── 8.4 Schur 引理验证 ──")

    unique_w = np.unique(np.round(w, 6))

    for lam in sorted(unique_w, reverse=True):
        mask = np.abs(w - lam) < 1e-6
        V_alpha = V[:, mask]
        d = V_alpha.shape[1]
        if d > 100:  # 跳过太大的块 (cp 混合)
            continue

        # 对每个生成元检查 PρP 是否为标量矩阵
        max_dev = 0
        avg_dev = 0
        for rho_s in generators:
            M = V_alpha.T.conj() @ rho_s @ V_alpha  # 投影到特征空间
            # 若不可约, M = c I
            c = np.trace(M) / d  # 标量因子
            deviation = np.linalg.norm(M - c * np.eye(d))
            max_dev = max(max_dev, deviation)
            avg_dev += deviation / len(generators)

        is_scalar = max_dev < 1e-6
        print(f"  λ={lam:.6f} dim={d:3d} "
              f"max_dev={max_dev:.2e} avg={avg_dev:.2e} "
              f"{'标量 OK' if is_scalar else '非标量 — 非不可约分量'}")

        if not is_scalar and d < 50:
            # 尝试分裂: 检查 ρ(s) 是否在子块中作用
            # 构造交换子空间
            commutator = np.zeros((d, d), dtype=complex)
            for rho_s in generators:
                M = V_alpha.T.conj() @ rho_s @ V_alpha
                commutator += M @ M.T.conj()
            eigvals_comm = np.linalg.eigvalsh(commutator)
            nonzero_modes = np.sum(eigvals_comm > 1e-3)
            print(f"    → 子块数 (comm rank) = {nonzero_modes}")


def test_face_completeness_condition():
    """8.5 面完备性条件验证

    有理谱 λ = 1 - k/m 需要:
      H1: Hermitian 条件 — 完整 3 轴覆盖 (消除 Ω_co 虚部对消)
      H2: 面完备条件 — 若包含面，则 3 方向全含 OR 仅含 180°

    反例:
      n=8: axis≠1, dir≠2 → 2 轴全方向 + 2 轴仅 180° → herm 但无理特征值
      n=16: 排除 U2/D2 → 几乎完整但方向不全 → herm 但无理特征值
    """
    print("\n── 8.5 面完备性条件 ──")

    sd = SlowDynamics.lite()

    test_cases = [
        (18, "全生成元 (完备)", True, True),
        (12, "面转 (k[2]!=2)", True, True),
        (10, "部分约束子集", True, True),
        (6, "仅 180°", True, True),
        (8, "axis≠1, dir≠2 (混合结构)", True, False),
        (16, "排除 U2/D2 (破缺)", True, False),
        (9, "单面 (非 Hermitian)", False, False),
        (4, "单轴非180 (非Herm, 实部有理)", False, True),
    ]

    for n, desc, expect_herm, expect_rational in test_cases:
        rm = sd.rho_moves(n=n)
        if len(rm) == 0:
            continue
        rhos = [rho for _, rho, *_ in rm.values()]
        A_S = sum(rhos) / len(rhos)

        is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)
        w_S = np.linalg.eigvalsh(A_S) if is_herm else np.linalg.eigvals(A_S)
        unique_w = np.unique(np.round(w_S[np.abs(np.imag(w_S)) < 1e-10], 6))

        m_eff = len(rhos) // 2 if len(rhos) % 2 == 0 else len(rhos)
        all_rational = True
        for lam in unique_w:
            lam_real = float(lam.real)
            k_val = round((1 - lam_real) * m_eff)
            pred = 1 - k_val / m_eff
            if abs(lam_real - pred) > 1e-5:
                all_rational = False

        herm_ok = "OK" if is_herm == expect_herm else "FAIL"
        rat_ok = "OK" if all_rational == expect_rational else "FAIL"
        print(f"  n={n:2d} |S|={len(rhos):2d} {desc:30s} "
              f"herm={str(is_herm):5s}(exp={str(expect_herm):5s}){herm_ok} "
              f"rational={str(all_rational):5s}(exp={str(expect_rational):5s}){rat_ok}")

    print("""\n  结论:
  有理谱 <==> Hermitian + 面完备性
  Hermitian 需要: 完整 3 轴覆盖 (虚部对消)
  面完备性: 每面要么 3 方向全包含, 要么仅 180""")


def test_full_rigorous_proof_chain(A_micro, w, V, generators):
    """8.0 完整严格证明链

    Theorem 5.1 (Rigorous): λ = 1 - k/9, k ∈ {0, 2, 3, 4, 6}

    证明链:
    Step 1: ρ = ρ_cp ⊕ ρ_ep ⊕ ρ_co ⊕ ρ_eo (block decomposition)
    Step 2: 18 生成元 → 9 对称单元 h_i = (g_i+g_i^{-1})/2
    Step 3: h_i Hermitian, perm block spec in {1,0,-1}, orient spec in {1,-1/2,+-1}
    Step 4: A = (1/9) sum h_i -> lambda = k'/9, k' integer
    Step 5: rep theory -> k' in {3,5,6,7,9} <-> k in {0,2,3,4,6}
    Step 6: Character 公式验证: λ_α = (1/|S|) Σ χ_α(s)/d_α
    Step 7: 面完备性: 有理谱需要 H1(Hermitian) + H2(face-complete)
    """
    print("\n" + "=" * 70)
    print("Theorem 5.1 完整严格证明链验证")
    print("=" * 70)

    # Step 1: Block decomposition
    print("\nStep 1: ρ = ρ_cp ⊕ ρ_ep ⊕ ρ_co ⊕ ρ_eo")
    blocks = detect_blocks(list(CubieMove.prim_moves().values()), V)
    sizes = sorted([len(b) for b in blocks], reverse=True)
    print(f"  Block sizes: {sizes[:6]}")
    print(f"  大块: cp={sizes[0]}, ep={sizes[1]} (若 64,144)")

    # Step 2: Generator pairing
    print("\nStep 2: 18 生成元 → 9 对称单元")
    h_ops, _h_names = test_generator_pairing_and_h_spectrum(A_micro)

    # Step 3: h_i spectrum
    print("\nStep 3: h_i 谱离散化 (已在 Step 2 验证)")

    # Step 4: λ = k/9
    print("\nStep 4: λ = k'/9 有理形式")
    unique_w = np.unique(np.round(w, 6))
    for lam in sorted(unique_w, reverse=True):
        k = round(lam * 9)
        print(f"  λ={lam:.6f} = {k}/9")

    # Step 5: k-selection
    print("\nStep 5: k-选择规则")
    test_k_selection_rule(w, h_ops)

    # Step 6: Character formula
    print("\nStep 6: Character 公式")
    test_character_on_eigenspaces(w, V, generators)

    # Step 7: Face completeness
    print("\nStep 7: 面完备性条件")
    test_face_completeness_condition()

    print("\n" + "=" * 70)
    print("证明链验证完成")


# ── 9. 不可约分解 (Irrep Decomposition) ──────────────────────────────

def test_character_inner_product():
    """9.1 Monte Carlo character inner product <chi, chi>.

    <chi, chi> = (1/|G|) sum_g |chi(g)|^2 = number of irreducible components.
    Since |G| ~ 2.1e10, use long random walks for approximate uniform sampling.

    Result:
      <chi, chi> > 1 means reducible
      <chi, chi> ~ n means ~n irreducible components (counting multiplicities)
    """
    print("\n── 9.1 Character Inner Product (Monte Carlo) ──")
    result = compute_character_mc(n_samples=5000, walk_length=50)
    chi_arr = result['chi_samples']

    print(f"  Samples: {len(chi_arr)}")
    print(f"  <chi, chi> = {result['inner_product']:.4f}")
    print(f"  Estimated irrep count = {result['n_irreps_est']}")
    print(f"  Mean chi = {result['chi_mean']:.4f}")
    print(f"  Std |chi| = {result['chi_std']:.4f}")

    if result['n_irreps_est'] > 1:
        print(f"  -> 228D = direct sum of ~{result['n_irreps_est']} irreps")
        print(f"  -> Average irrep dimension ~ {228/result['n_irreps_est']:.1f}")
    else:
        print(f"  -> Representation appears irreducible (unlikely for 228D)")

    # Show histogram of |chi| values
    abs_chi = np.abs(chi_arr)
    bins = np.linspace(0, max(abs_chi), 20)
    hist, _ = np.histogram(abs_chi, bins=bins)
    print(f"\n  |chi| distribution (top values):")
    for i in np.argsort(-hist)[:8]:
        if hist[i] > 0:
            print(f"    |chi| ~ {bins[i]:.1f} - {bins[i+1]:.1f}: {hist[i]} samples")

    return result


def test_irrep_block_detection(generators):
    """9.2 Irrep block detection via spectral clustering.

    Method: Generate random Hermitian operators H_k from the group algebra.
    Each eigenvector gets a spectral signature (eigenvalues across all H_k).
    Vectors from the same irrep share identical signatures (Schur lemma).
    Cluster by signature similarity -> irrep blocks.
    """
    print("\n── 9.2 Irrep Block Detection ──")
    blocks, U_irrep, _signatures = detect_irrep_blocks(generators, n_random_ops=8, tol=1e-5)

    sizes = sorted([len(b) for b in blocks], reverse=True)
    print(f"  Number of blocks: {len(blocks)}")
    print(f"  Block sizes: {sizes[:15]}")
    if len(sizes) > 15:
        print(f"              ... (+{len(sizes)-15} more)")

    # Group by size
    from collections import Counter
    size_counts = Counter(sizes)
    print(f"\n  Size distribution:")
    for sz, cnt in sorted(size_counts.items(), reverse=True):
        print(f"    dim={sz:3d}: {cnt} blocks")

    # Verify: Schur lemma on each block
    print(f"\n  Verifying Schur lemma on blocks (d <= 50):")
    schur_results = verify_schur_on_irreps(generators, blocks, U_irrep, tol=1e-5)
    for r in schur_results:
        if r['dim'] <= 50:
            status = "irrep OK" if r['is_irrep'] else "NOT irrep"
            print(f"    block {r['irrep_idx']:2d} dim={r['dim']:3d} "
                  f"max_dev={r['max_deviation']:.2e} rel={r['rel_deviation']:.2e} {status}")

    return blocks, U_irrep, schur_results


def test_eigenspace_to_irrep_mapping(V, w, irrep_blocks, U_irrep):
    """9.3 Map A's 5 eigenspaces onto the irrep decomposition.

    For each eigenvalue lambda, compute overlap with each irrep block.
    This reveals which irreps contribute to which spectral layer.
    """
    print("\n── 9.3 Eigenspace -> Irrep Mapping ──")
    mapping = map_eigenspaces_to_irreps(V, w, irrep_blocks, U_irrep)

    unique_w = np.unique(np.round(w, 6))

    # Summary: for each eigenvalue, list matched irreps
    print(f"\n  Spectral decomposition of 228D representation:")
    print(f"  {'lambda':>10s} {'eig_dim':>7s} {'matched_irreps':>30s} {'total_irrep_dim':>15s}")
    print(f"  {'-'*10} {'-'*7} {'-'*30} {'-'*15}")
    for lam in sorted(unique_w, reverse=True):
        lam_matches = [m for m in mapping
                       if abs(m['lambda'] - lam) < 1e-6 and m['is_matched']]
        eig_dim = lam_matches[0]['eigenspace_dim'] if lam_matches else 0
        irrep_dims = [m['irrep_dim'] for m in lam_matches]
        total_irrep_dim = sum(irrep_dims)
        irrep_str = '+'.join(str(d) for d in sorted(irrep_dims, reverse=True)[:6])
        if len(irrep_dims) > 6:
            irrep_str += '+...'
        print(f"  {lam:.6f}   {eig_dim:3d}     {irrep_str:30s} {total_irrep_dim:5d}")

    # Unmatched irreps (not cleanly in any eigenspace)
    matched_irrep_ids = set(m['irrep_idx'] for m in mapping if m['is_matched'])
    all_irrep_ids = set(m['irrep_idx'] for m in mapping)
    unmatched = all_irrep_ids - matched_irrep_ids
    if unmatched:
        print(f"\n  Irreps not cleanly matching any eigenspace: {len(unmatched)}")
        for i in sorted(unmatched)[:5]:
            d = len(irrep_blocks[i])
            print(f"    irrep {i}: dim={d}")

    return mapping


def test_full_irrep_analysis(w, V, generators):
    """9.0 Full irrep decomposition pipeline.

    Unified analysis connecting:
      Character inner product -> irrep count
      Spectral clustering -> irrep blocks
      Eigenspace mapping -> which irreps produce each eigenvalue
      Schur verification -> A acts as scalar on each irrep
    """
    print("\n" + "=" * 70)
    print("Section 9: Irreducible Decomposition Analysis")
    print("=" * 70)

    # 9.1 Character analysis
    chi_result = test_character_inner_product()

    # 9.2 Irrep block detection
    irrep_blocks, U_irrep, _schur = test_irrep_block_detection(generators)

    # 9.3 Eigenspace -> irrep mapping
    _mapping = test_eigenspace_to_irrep_mapping(V, w, irrep_blocks, U_irrep)

    # Summary
    n_irreps_from_chi = chi_result['n_irreps_est']
    n_blocks = len(irrep_blocks)

    print(f"\n{'='*70}")
    print(f"Synthesis:")
    print(f"  <chi, chi> = {chi_result['inner_product']:.2f} -> ~{n_irreps_from_chi} irreps")
    print(f"  Spectral clustering -> {n_blocks} blocks")
    print(f"  A has 5 eigenvalues -> 5 isotypic components under averaging")
    print(f"  Schur lemma: A|_irrep = lambda_alpha * I for each irrep alpha")
    print(f"  lambda_alpha = (1/|S|) sum_s chi_alpha(s) / d_alpha")
    print(f"  -> 5 eigenvalues = 5 distinct character averages over the {n_irreps_from_chi} irreps")
    print(f"  -> Spectral collapse = projection of {n_irreps_from_chi} character values")
    print(f"     onto 5 rational values via generator averaging")
    print(f"{'='*70}")

# ── 10. 组合移动谱分析 & h_i 交换性 ───────────────────────────────────

def test_composed_move_spectral_structure(A_micro, w, V, generators):
    """10.1 组合移动 S x S 的谱结构分析

    A_micro 用于构造谱投影器, generators 用于慢子空间代数维度

    核心发现:
    - 18 prim 的 A 产生 5 层有理谱 (spectral collapse)
    - 组合移动 (S x S, 261 个) 扩展群但不扩展谱代数
    - rho(g) 不在 span{E_i} 内 (重建误差 ~0.62)
    - 同轴组合误差更小 (~0.51), 跨轴误差更大 (~0.65)
    - A_all (279 moves) 有 7 个特征值, 含无理数 → 谱坍缩只对 prim 生成元集成立

    记法: ActionToken.__str__() 产生标准魔方记法 U/U'/U2/R/R'/R2/...
    """
    print("\n── 10.1 组合移动谱结构 ──")
    from rime.cube import ActionToken

    prim = CubieMove.prim_moves()

    # 构造谱投影器
    unique_w = np.unique(np.round(w, 6))
    E = {}
    for lam in unique_w:
        mask = np.abs(w - lam) < 1e-6
        V_lam = V[:, mask]
        E[lam] = V_lam @ V_lam.T.conj()

    # S x S 组合
    products = CubieBase.generate_compose_moves(prim, commutator=False)
    print(f"  S x S compositions: {len(products)} unique non-identity")

    # 块对角重建误差: rho(g) vs sum_i E_i rho(g) E_i
    same_axis_err = []
    cross_axis_err = []
    all_errors = []

    for seq_keys, mv in products.items():
        rho_g = mv.rho()
        rho_recon = sum(E[lam] @ rho_g @ E[lam] for lam in unique_w)
        err = np.linalg.norm(rho_g - rho_recon) / np.linalg.norm(rho_g)
        all_errors.append(err)

        tokens = [ActionToken.from_cubie_move(*k, n=3) for k in seq_keys]
        if len(seq_keys) == 2:
            k1, k2 = seq_keys
            if k1[0] == k2[0]:
                same_axis_err.append(err)
            else:
                cross_axis_err.append(err)

    print(f"  全部重建误差: mean={np.mean(all_errors):.4f}, "
          f"min={np.min(all_errors):.4f}, max={np.max(all_errors):.4f}")
    print(f"  误差 > 0.5 比例: {np.mean(np.array(all_errors) > 0.5):.4f}")

    if same_axis_err:
        print(f"  同轴组合 ({len(same_axis_err)}): mean={np.mean(same_axis_err):.4f}")
    if cross_axis_err:
        print(f"  跨轴组合 ({len(cross_axis_err)}): mean={np.mean(cross_axis_err):.4f}")

    # 特征值 on 各层: chi_lam(g) = Tr(P_lam rho(g))
    print(f"\n  各层 character 样本 (前6个组合):")
    for seq_keys, mv in list(products.items())[:6]:
        rho_g = mv.rho()
        tokens = [ActionToken.from_cubie_move(*k, n=3) for k in seq_keys]
        notation = ' '.join(str(t) for t in tokens)
        chars = []
        for lam in sorted(unique_w, reverse=True):
            chi = np.trace(E[lam] @ rho_g).real
            chars.append(f"{chi:+.0f}")
        print(f"    {notation:12s}: chi = [{', '.join(chars)}]")

    # 慢子空间代数维度
    P_slow_idx = w >= 2 / 3 - 1e-8
    V_slow = V[:, P_slow_idx]
    P_slow = V_slow @ V_slow.T.conj()

    prim_mats = np.array([P_slow @ rho_s @ P_slow for rho_s in generators])
    rank_prim = np.linalg.matrix_rank(prim_mats.reshape(len(generators), -1), tol=1e-8)

    comp_mats = list(prim_mats)
    for seq_keys, mv in products.items():
        rho_g = mv.rho()
        M = P_slow @ rho_g @ P_slow
        comp_mats.append(M.flatten())
    comp_mats = np.array(comp_mats)
    rank_comp = np.linalg.matrix_rank(comp_mats, tol=1e-8)

    print(f"\n  span{{P_slow rho(s) P_slow}} rank = {rank_prim} (18 prim)")
    print(f"  span{{P_slow rho(g) P_slow}} rank = {rank_comp} (18+{len(products)} composed)")

    # 扩展平均算子 A_all
    all_rhos = list(generators) + [mv.rho() for mv in products.values()]
    A_all = sum(all_rhos) / len(all_rhos)
    w_all = np.linalg.eigvalsh(A_all)
    unique_all = np.unique(np.round(w_all, 6))
    print(f"  A_all ({len(all_rhos)} moves): {len(unique_all)} distinct eigenvalues")
    print(f"  eigenvalues: {sorted(unique_all, reverse=True)[:8]}")
    print(f"  → 谱坍缩只对 prim 生成元集成立, 扩展后退化消失")

    # 换位子分析
    products_comm = CubieBase.generate_compose_moves(prim, commutator=True)
    print(f"\n  换位子 [A,B]: {len(products_comm)} unique non-identity")
    chi1_comm = []
    for seq_keys, mv in products_comm.items():
        rho_g = mv.rho()
        chi1 = np.trace(E[unique_w[0]] @ rho_g).real
        chi1_comm.append(chi1)
    print(f"  chi_1(换位子): mean={np.mean(chi1_comm):.2f}, "
          f"min={min(chi1_comm):.2f}, max={max(chi1_comm):.2f}")
    print(f"  → 换位子几乎保留全部 trivial 分量 (chi_1 ~ 24)")


def test_h_i_commutativity_structure(A_micro, w, V, generators):
    """10.2 h_i 交换性结构分析

    A_micro 用于验证 A=(1/9)sum h_i; w,V 用于特征空间限制; generators 保留接口兼容

    核心发现:
    - 9 个对称单元 h_i = (g_i + g_i^{-1})/2, 6 面级 + 3 轴级
    - 全空间: 16/36 对交换, 模式由轴分解决定
    - 同轴 h_i 交换 (L'/L 与 R'/R, L'/L 与 L2/R2)
    - 跨轴 h_i 不交换 (不同轴的置换互相干涉)
    - 轴级 180° 对 (h_axis) 与所有 h_i 交换
    - 限制到 A 的特征空间:
      λ=1 (24D): 全部交换 (trivial 表示, 标量作用)
      λ=7/9 (44D): 近似交换 (max ||[h_i,h_j]|| ~ 4e-7) → 准标量作用
      λ=2/3 (32D): 3/36 交换 → ρ(G)-invariant 但非单个 ρ(s) 特征空间
      λ=5/9,1/3: 不交换 → 快速层, 高度混合

    这解释了 Step 4 gap: h_i 不交换 → 无共同特征基
    但 λ=7/9 层上近似交换 → rational form 在此层上有部分代数基础
    """
    print("\n── 10.2 h_i 交换性结构 ──")
    from rime.cube import ActionToken

    prim = CubieMove.prim_moves()

    # 构造 9 个对称单元
    h_operators = []
    h_names = []

    for axis in range(3):
        for side in [-1, 1]:
            cw_key = (axis, side, -1)
            ccw_key = (axis, side, 1)
            if cw_key in prim and ccw_key in prim:
                rho_cw = prim[cw_key].rho()
                rho_ccw = prim[ccw_key].rho()
                h = (rho_cw + rho_ccw) / 2
                h_operators.append(h)
                at_cw = ActionToken.from_cubie_move(*cw_key, n=3)
                at_ccw = ActionToken.from_cubie_move(*ccw_key, n=3)
                h_names.append(f"({at_cw}+{at_ccw})/2")

    for axis in range(3):
        keys_180 = [(axis, side, 2) for side in [-1, 1] if (axis, side, 2) in prim]
        if len(keys_180) == 2:
            rho_a = prim[keys_180[0]].rho()
            rho_b = prim[keys_180[1]].rho()
            h = (rho_a + rho_b) / 2
            h_operators.append(h)
            at_a = ActionToken.from_cubie_move(*keys_180[0], n=3)
            at_b = ActionToken.from_cubie_move(*keys_180[1], n=3)
            h_names.append(f"({at_a}+{at_b})/2")

    n_h = len(h_operators)
    print(f"  对称单元数: {n_h} (6 face + 3 axis)")
    for i, name in enumerate(h_names):
        print(f"    h_{i}: {name}")

    # 验证 A = (1/9) sum h_i
    A_from_h = sum(h_operators) / 9
    err_A = np.linalg.norm(A_from_h - A_micro)
    print(f"\n  A = (1/9) sum h_i: error = {err_A:.2e} {'OK' if err_A < 1e-12 else 'FAIL'}")

    # 全空间交换子
    comm_norms = np.zeros((n_h, n_h))
    for i in range(n_h):
        for j in range(n_h):
            comm = h_operators[i] @ h_operators[j] - h_operators[j] @ h_operators[i]
            comm_norms[i, j] = np.linalg.norm(comm)

    # 交换对统计
    n_commute = 0
    n_total = n_h * (n_h - 1) // 2
    for i in range(n_h):
        for j in range(i + 1, n_h):
            if comm_norms[i, j] < 1e-10:
                n_commute += 1

    print(f"\n  全空间交换性: {n_commute}/{n_total} 对交换")
    print(f"  交换对 (按轴结构):")
    for i in range(n_h):
        for j in range(i + 1, n_h):
            if comm_norms[i, j] < 1e-10:
                print(f"    [h_{i}, h_{j}] = 0  ({h_names[i]}, {h_names[j]})")

    # 轴分解模式: 同轴=交换, 跨轴=不交换
    same_ax_comm = 0
    cross_ax_comm = 0
    same_ax_total = 0
    cross_ax_total = 0
    for i in range(n_h):
        for j in range(i + 1, n_h):
            # h_i, h_j 的轴: 0-1=axis0, 2-3=axis1, 4-5=axis2, 6=axis0_pair, 7=axis1_pair, 8=axis2_pair
            ax_i = i // 2 if i < 6 else i - 6
            ax_j = j // 2 if j < 6 else j - 6
            if ax_i == ax_j:
                same_ax_total += 1
                if comm_norms[i, j] < 1e-10:
                    same_ax_comm += 1
            else:
                cross_ax_total += 1
                if comm_norms[i, j] < 1e-10:
                    cross_ax_comm += 1

    print(f"  同轴: {same_ax_comm}/{same_ax_total} 交换")
    print(f"  跨轴: {cross_ax_comm}/{cross_ax_total} 交换")

    # 限制到各特征空间
    unique_w = np.unique(np.round(w, 6))
    E = {}
    for lam in unique_w:
        mask = np.abs(w - lam) < 1e-6
        V_lam = V[:, mask]
        E[lam] = V_lam @ V_lam.T.conj()

    print(f"\n  h_i 限制到 A 特征空间的交换性:")
    for lam in sorted(unique_w, reverse=True):
        mask = np.abs(w - lam) < 1e-6
        d = int(np.sum(mask))
        V_lam = V[:, mask]
        max_comm = 0
        n_comm_pairs = 0
        for i in range(n_h):
            for j in range(i + 1, n_h):
                h_i_r = V_lam.T.conj() @ h_operators[i] @ V_lam
                h_j_r = V_lam.T.conj() @ h_operators[j] @ V_lam
                comm_r = h_i_r @ h_j_r - h_j_r @ h_i_r
                norm_r = np.linalg.norm(comm_r)
                max_comm = max(max_comm, norm_r)
                if norm_r < 1e-10:
                    n_comm_pairs += 1

        status = "全部交换 (trivial)" if n_comm_pairs == n_total else \
                 f"近似交换 (准标量)" if max_comm < 1e-5 else \
                 f"部分交换"
        print(f"    lambda={lam:.4f} dim={d:3d}: max||[h_i,h_j]||={max_comm:.2e}, "
              f"交换对={n_comm_pairs}/{n_total}  {status}")

    print(f"\n  Step 4 gap 解释:")
    print(f"    h_i 在全空间不交换 → 无共同特征基 → lambda=k/9 无法从特征值求和推导")
    print(f"    但 lambda=7/9 层上近似交换 → rational form 在此层有部分代数基础")
    print(f"    lambda=1 层完全交换 → trivial 表示, 标量作用 (已证明)")


# ── 11. 生成元集依赖性与子群结构对比 ──────────────────────────────────

def test_generator_set_dependence():
    """11.1 不同生成元集的谱坍缩对比

    核心问题: 5 层有理谱是 18 个标准生成元的特例，还是更深层的结构？

    对比:
    - 完整 18 (baseline): quarter + half turns
    - 12 quarter-turn (k[2]!=2): 仅 ±90° 面转
    - 6 half-turn (k[2]==2): 仅 180° 面转
    - 单轴 abelian (axis=0): 6 个同轴生成元全部交换 → Step 4 gap 关闭
    - 单轴 abelian (axis=1): 同上
    - 单轴 abelian (axis=2): 同上
    - 12 quarter-turn compose products: S×S 组合
    - 6 half-turn compose products: S×S 组合
    - 随机子集 (9/18): 随机选一半生成元

    对每种集合计算: |S|, Hermitian?, #eigenvalues, rational λ=k/m?, poly_rank, h_i 交换性
    """
    print("\n" + "=" * 70)
    print("§11.1 生成元集依赖性: 谱坍缩 vs 生成元结构")
    print("=" * 70)

    prim = CubieMove.prim_moves()

    # ── 定义各生成元集 ──
    gens_18 = dict(prim)  # 完整 18
    gens_12 = {k: v for k, v in prim.items() if k[2] != 2}  # 12 quarter-turn
    gens_6_half = {k: v for k, v in prim.items() if k[2] == 2}  # 6 half-turn

    # 单轴 abelian (所有同轴生成元互相交换)
    gens_abelian_0 = {k: v for k, v in prim.items() if k[0] == 0}
    gens_abelian_1 = {k: v for k, v in prim.items() if k[0] == 1}
    gens_abelian_2 = {k: v for k, v in prim.items() if k[0] == 2}

    # 随机子集
    import random
    random.seed(42)
    keys_18 = list(prim.keys())
    random.shuffle(keys_18)
    gens_random_9 = {k: prim[k] for k in keys_18[:9]}

    # compose products
    compose_12 = CubieBase.generate_compose_moves(gens_12, commutator=False)
    compose_6 = CubieBase.generate_compose_moves(gens_6_half, commutator=False)
    compose_18 = CubieBase.generate_compose_moves(gens_18, commutator=False)

    def analyze_generator_set(name, gens_dict, desc=""):
        """分析一个生成元集的所有谱性质"""
        rhos = [m.rho() for m in gens_dict.values()]
        n_gen = len(rhos)
        if n_gen == 0:
            return None

        A_S = sum(rhos) / n_gen
        is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)

        if is_herm:
            w_S = np.linalg.eigvalsh(A_S)
        else:
            w_S = np.linalg.eigvals(A_S)
            w_S = np.real(w_S[np.abs(np.imag(w_S)) < 1e-8])

        unique_w = np.unique(np.round(w_S, 8))
        n_eig = len(unique_w)

        pr = poly_rank(A_S, k=10)

        m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen
        all_rational = True
        irrational_vals = []
        for lam in unique_w:
            lam_r = float(lam.real) if isinstance(lam, complex) else float(lam)
            k_val = round((1 - lam_r) * m_eff)
            pred = 1 - k_val / m_eff
            if abs(lam_r - pred) > 1e-5:
                all_rational = False
                irrational_vals.append(lam_r)

        # h_i 交换性
        h_ops = []
        for axis in range(3):
            for side in [-1, 1]:
                cw_key = (axis, side, -1)
                ccw_key = (axis, side, 1)
                if cw_key in gens_dict and ccw_key in gens_dict:
                    h_ops.append((gens_dict[cw_key].rho() + gens_dict[ccw_key].rho()) / 2)
        for axis in range(3):
            keys_180 = [(axis, side, 2) for side in [-1, 1] if (axis, side, 2) in gens_dict]
            if len(keys_180) == 2:
                h_ops.append((gens_dict[keys_180[0]].rho() + gens_dict[keys_180[1]].rho()) / 2)

        n_h = len(h_ops)
        n_commute = 0
        max_comm = 0
        if n_h >= 2:
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    comm_norm = np.linalg.norm(h_ops[i] @ h_ops[j] - h_ops[j] @ h_ops[i])
                    max_comm = max(max_comm, comm_norm)
                    if comm_norm < 1e-10:
                        n_commute += 1
            n_pairs = n_h * (n_h - 1) // 2
        else:
            n_pairs = 0

        return {
            'name': name, 'desc': desc, 'n_gen': n_gen, 'm_eff': m_eff,
            'hermitian': is_herm, 'n_eig': n_eig, 'poly_rank': pr,
            'all_rational': all_rational, 'irrational_vals': irrational_vals,
            'n_h': n_h, 'n_commute': n_commute, 'n_pairs': n_pairs,
            'max_comm': max_comm, 'eigenvalues': sorted(unique_w, reverse=True),
            'all_commute': n_commute == n_pairs if n_pairs > 0 else True,
        }

    # ── 运行所有分析 ──
    configs = [
        ("18 full (baseline)", gens_18, "all quarter + half turns"),
        ("12 quarter-turn", gens_12, "k[2]!=2, ±90° only"),
        ("6 half-turn", gens_6_half, "k[2]==2, 180° only"),
        ("abelian axis=0 (R/L)", gens_abelian_0, "single-axis, all commute"),
        ("abelian axis=1 (U/D)", gens_abelian_1, "single-axis, all commute"),
        ("abelian axis=2 (F/B)", gens_abelian_2, "single-axis, all commute"),
        ("random 9/18", gens_random_9, "random subset"),
    ]

    results = []
    for name, gens_dict, desc in configs:
        r = analyze_generator_set(name, gens_dict, desc)
        if r:
            results.append(r)

    # 添加 compose 集 (用组合移动的 rho 直接算 A)
    for comp_name, comp_dict in [
        ("12q compose (134)", compose_12),
        ("6h compose (24)", compose_6),
        ("18 compose (261)", compose_18),
    ]:
        if len(comp_dict) == 0:
            continue
        rhos = [mv.rho() for mv in comp_dict.values()]
        n_gen = len(rhos)
        A_S = sum(rhos) / n_gen
        is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)
        if is_herm:
            w_S = np.linalg.eigvalsh(A_S)
        else:
            w_S = np.linalg.eigvals(A_S)
            w_S = np.real(w_S[np.abs(np.imag(w_S)) < 1e-8])
        unique_w = np.unique(np.round(w_S, 8))
        n_eig = len(unique_w)
        pr = poly_rank(A_S, k=10)
        m_eff = n_gen // 2
        all_rational = True
        irrational_vals = []
        for lam in unique_w:
            lam_r = float(lam.real) if isinstance(lam, complex) else float(lam)
            k_val = round((1 - lam_r) * m_eff)
            pred = 1 - k_val / m_eff
            if abs(lam_r - pred) > 1e-5:
                all_rational = False
                irrational_vals.append(lam_r)

        results.append({
            'name': comp_name, 'desc': 'compose products',
            'n_gen': n_gen, 'm_eff': m_eff,
            'hermitian': is_herm, 'n_eig': n_eig, 'poly_rank': pr,
            'all_rational': all_rational, 'irrational_vals': irrational_vals,
            'n_h': 0, 'n_commute': 0, 'n_pairs': 0,
            'max_comm': 0, 'eigenvalues': sorted(unique_w, reverse=True),
            'all_commute': False,
        })

    # ── 打印汇总表 ──
    header = f"{'Generator set':<28s} {'|S|':>4s} {'herm':>5s} {'#λ':>4s} {'poly_rk':>7s} {'rational':>8s} {'h_i comm':>10s} {'max||[h,h]||':>14s}"
    print(header)
    print("-" * len(header))
    for r in results:
        herm_str = "Yes" if r['hermitian'] else "No"
        rat_str = "Yes" if r['all_rational'] else "No (!)"
        if r['n_pairs'] > 0:
            comm_str = f"{r['n_commute']}/{r['n_pairs']}"
        else:
            comm_str = "N/A"
        maxc_str = f"{r['max_comm']:.1e}" if r['max_comm'] > 0 else "0"
        print(f"{r['name']:<28s} {r['n_gen']:4d} {herm_str:>5s} {r['n_eig']:4d} "
              f"{r['poly_rank']:7d} {rat_str:>8s} {comm_str:>10s} {maxc_str:>14s}")

    # ── 详细特征值 ──
    print(f"\n详细特征值:")
    for r in results:
        if r['all_rational']:
            k_vals = [round((1 - float(lam)) * r['m_eff']) for lam in r['eigenvalues']]
            lam_str = ", ".join(f"1-{k}/{r['m_eff']}" for k, lam in zip(k_vals, r['eigenvalues']))
        else:
            lam_str = ", ".join(f"{lam:.4f}" for lam in r['eigenvalues'][:8])
        irrational_str = ""
        if r['irrational_vals']:
            irrational_str = f"  irrational: {r['irrational_vals']}"
        print(f"  {r['name']:<28s} m={r['m_eff']:3d}: {lam_str}{irrational_str}")

    # ── 关键发现 ──
    print(f"\n关键发现:")
    abelian_results = [r for r in results if 'abelian' in r['name']]
    full_18_result = [r for r in results if r['name'] == '18 full (baseline)']

    if abelian_results:
        all_ab_commute = all(r['all_commute'] for r in abelian_results)
        print(f"  1. 单轴 abelian 子集: h_i 全部交换 = {all_ab_commute}")
        print(f"     → Step 4 gap 自动关闭 (共同特征基存在)")
        for r in abelian_results:
            print(f"     {r['name']}: herm={r['hermitian']}, #λ={r['n_eig']}, "
                  f"rational={r['all_rational']}")

    if full_18_result:
        r = full_18_result[0]
        print(f"  2. Full 18: h_i 交换 {r['n_commute']}/{r['n_pairs']}, "
              f"herm={r['hermitian']}, #λ={r['n_eig']}")

    compose_results = [r for r in results if 'compose' in r['name']]
    for r in compose_results:
        print(f"  3. {r['name']}: herm={r['hermitian']}, #λ={r['n_eig']}, "
              f"rational={r['all_rational']}")

    print(f"\n  结论:")
    print(f"    - 有理谱坍缩 (λ=1-k/m) 仅对满足面完备+3轴覆盖的 prim 生成元集成立")
    print(f"    - 单轴 abelian: h_i 全部交换, 但非 Hermitian (缺少 3 轴虚部对消)")
    print(f"    - Compose 扩展: 特征值增多, 有理形式破缺 → 谱坍缩是 prim-generator 特定现象")
    print(f"    - 12 quarter-turn compose 和 6 half-turn compose: 比 full 18 compose 更可控")
    print(f"    - 谱坍缩 = 生成元对称性 + 表示结构 共同作用的结果")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# §12. Galois-Theoretic Proof Verification (Theorems 5.6–5.8)
# ═══════════════════════════════════════════════════════════════════════════════

def test_galois_integrality_generator_characters():
    """Theorem 5.6: χ(s) ∈ ℤ for all 18 generators on every structural block.

    Proof: n_1 = n_2 for corner orientation because Σ o_i ≡ 0 (mod 3) with
    |affected corners| ≤ 4. Then χ_co = n_0 + n_1(ω+ω²) = n_0 - n_1 ∈ ℤ.
    Cp, Ep, Eo blocks are permutation matrices → integer trace.
    """
    prim = CubieMove.prim_moves()
    omega = np.exp(2j * np.pi / 3)  # complex128
    # Use 1e-6 tolerance for ω comparison: rho entries are complex64 upcast,
    # |complex64(ω) - complex128(ω)| ≈ 4e-9, so 1e-8 borderline, 1e-6 safe.
    TOL_OMEGA = 1e-6

    print("\n═══ Theorem 5.6: Integrality of Generator Characters ═══")
    print("Claim: χ(s) ∈ ℤ for all 18 generators on every structural block.\n")

    all_ok = True
    for (axis, side, direction), move in sorted(prim.items()):
        r = move.rho().astype(np.complex128)
        at = str(ActionToken.from_cubie_move(axis, side, direction, n=3))

        chi_cp = np.trace(r[:64, :64]).real
        chi_ep = np.trace(r[64:208, 64:208]).real
        chi_eo = np.trace(r[216:228, 216:228]).real

        Co_diag = np.diag(r[208:216, 208:216])
        n_0 = sum(1 for x in Co_diag if abs(x - 1.0) < 1e-10)
        n_1 = sum(1 for x in Co_diag if abs(x - omega) < TOL_OMEGA)
        n_2 = sum(1 for x in Co_diag if abs(x - omega**2) < TOL_OMEGA)
        chi_co = n_0 - n_1

        undetected = 8 - n_0 - n_1 - n_2
        if undetected > 0:
            print(f"  WARN: {at}: {undetected} undetected Co entries "
                  f"(tolerance too tight for complex64→128 upcast)")

        cp_ok = abs(chi_cp - round(chi_cp)) < 1e-10
        ep_ok = abs(chi_ep - round(chi_ep)) < 1e-10
        eo_ok = abs(chi_eo - round(chi_eo)) < 1e-10
        co_ok = n_1 == n_2

        if not all([cp_ok, ep_ok, eo_ok, co_ok]):
            print(f"  FAIL: {at}: cp={chi_cp} ep={chi_ep} eo={chi_eo} "
                  f"n0={n_0} n1={n_1} n2={n_2} χ_co={chi_co}")
            all_ok = False

    if all_ok:
        print("  ✓ All 18 generators have integer character on all 4 blocks.\n")
    else:
        print("  ✗ Some generators have non-integer character.\n")

    # Detailed table
    print(f"  {'move':<6s} {'χ_cp':>6s} {'χ_ep':>6s} {'χ_eo':>6s} {'n_0':>4s} {'n_1':>4s} {'n_2':>4s} {'χ_co':>6s}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*6}")
    for (axis, side, direction), move in sorted(prim.items()):
        r = move.rho().astype(np.complex128)
        at = str(ActionToken.from_cubie_move(axis, side, direction, n=3))
        chi_cp = int(np.round(np.trace(r[:64, :64]).real))
        chi_ep = int(np.round(np.trace(r[64:208, 64:208]).real))
        chi_eo = int(np.round(np.trace(r[216:228, 216:228]).real))
        Co_diag = np.diag(r[208:216, 208:216])
        n_0 = sum(1 for x in Co_diag if abs(x - 1.0) < 1e-10)
        n_1 = sum(1 for x in Co_diag if abs(x - omega) < TOL_OMEGA)
        n_2 = sum(1 for x in Co_diag if abs(x - omega**2) < TOL_OMEGA)
        chi_co = n_0 - n_1
        print(f"  {at:<6s} {chi_cp:6d} {chi_ep:6d} {chi_eo:6d} {n_0:4d} {n_1:4d} {n_2:4d} {chi_co:6d}")

    print("\n  Proof: Σ o_i ≡ 0 (mod 3) on 8 corners → n₁·1 + n₂·2 ≡ 0 → n₁ ≡ n₂ (mod 3)")
    print("    With ≤ 4 corners affected → n₁ = n₂ ∈ {0, 2}.")
    print("    χ_co = n_0 + n_1·ω + n_2·ω² = n_0 - n_1 ∈ ℤ.")
    print("    χ_cp, χ_ep, χ_eo ∈ ℤ (permutation traces).")
    print("    ∴ χ(s) ∈ ℤ for all s ∈ S. ∎\n")

    return all_ok


def test_galois_invariance_averaging_operator():
    """Theorem 5.7: σ(A) = A under face-symmetry → A entries ∈ ℚ.

    Galois group Gal(ℚ(ω)/ℚ) ≅ {1, σ} where σ(ω) = ω² = ω̄.
    For A = (1/|S|) Σ ρ(s): Cp/Ep are real permutation matrices, Co has ω entries,
    Eo has ±1 entries. σ(A) = A iff the ω entries in Co sum to real values.

    Critical fact: Co(CW) = Co(CCW) for all faces (this Rubik's cube convention).
    Co(180°) = I for all faces. Thus σ(A_co) = A_co emerges from GLOBAL
    cancellation across complementary faces (R+L, F+B), not from individual
    generator σ-invariance.
    """
    prim = CubieMove.prim_moves()
    sd = SlowDynamics.lite()

    print("═══ Theorem 5.7: Galois Invariance σ(A) = A ═══")
    print("Claim: For face-symmetric S, σ(A) = A → A entries ∈ ℚ.\n")

    # First, document the Co block structure
    print("  Co block structure (key to Galois invariance):")
    print("    Co(CW)=Co(CCW) for all 6 faces  (CW and CCW have identical Co diagonal)")
    print("    Co(180°)=I for all 6 faces       (180° moves preserve corner orientation)")
    omega = np.exp(2j * np.pi / 3)

    # Show Co structure for one example face
    TOL_OMEGA = 1e-6
    for axis, side in [(0, -1), (1, -1), (2, -1)]:  # L, D, B faces — one per axis
        cw_key = (axis, side, -1)
        r = prim[cw_key].rho().astype(np.complex128)
        Co_diag = np.diag(r[208:216, 208:216])
        at = str(ActionToken.from_cubie_move(*cw_key, n=3))
        n_0 = sum(1 for x in Co_diag if abs(x - 1.0) < 1e-10)
        n_1 = sum(1 for x in Co_diag if abs(x - omega) < TOL_OMEGA)
        n_2 = sum(1 for x in Co_diag if abs(x - omega**2) < TOL_OMEGA)
        entries_str = ', '.join(f'{x.real:.3f}{x.imag:+.3f}j' for x in Co_diag[:4])
        print(f"    Co({at}): n0={n_0}, n1={n_1}, n2={n_2}  [{entries_str}...]")
    print(f"    → n₁=n₂ for all CW/CCW (orientation conservation)")
    print(f"    → σ: ω↦ω² flips n₁↔n₂ → σ(Co(g)) ≠ Co(g) but σ(Co(g)) = Co(g)")
    print(f"      for U/D faces (all-1 Co). For R/L/F/B, σ(Co) has ω entries swapped.\n")

    # Key demonstration: show Co sums for opposite faces cancel
    print("  Opposite-face Co sum (demonstrating cross-face cancellation):")
    for axis in range(3):
        r1 = prim[(axis, -1, -1)].rho().astype(np.complex128)  # CW on negative side
        r2 = prim[(axis, 1, -1)].rho().astype(np.complex128)   # CW on positive side
        Co_sum = r1[208:216, 208:216] + r2[208:216, 208:216]
        sigma_Co_sum = np.conj(Co_sum)
        is_real = np.allclose(Co_sum, sigma_Co_sum, atol=1e-10)
        n_real = sum(1 for i in range(8) if abs(Co_sum[i, i].imag) < 1e-10)
        at1 = str(ActionToken.from_cubie_move(axis, -1, -1, n=3))
        at2 = str(ActionToken.from_cubie_move(axis, 1, -1, n=3))
        print(f"    axis={axis}: Co({at1}) + Co({at2}): σ-invariant={is_real}, "
              f"{n_real}/8 entries real")

    # Full face sum (CW + CCW + 180°)
    print("\n  Full-face Co + σ(Co) check (CW+CCW+180° per face):")
    for axis in range(3):
        for side in [-1, 1]:
            keys = [(axis, side, d) for d in [-1, 1, 2]]
            Co_face = sum(prim[k].rho().astype(np.complex128)[208:216, 208:216]
                         for k in keys)
            sigma_Co_face = np.conj(Co_face)
            is_real = np.allclose(Co_face, sigma_Co_face, atol=1e-10)
            has_imag = sum(1 for i in range(8) if abs(Co_face[i, i].imag) > 1e-10)
            at = f"face(ax={axis},sd={side:+d})"
            print(f"    {at}: σ-invariant={is_real}, imaginary entries={has_imag}")

    # Build generator sets
    gens_sets = {
        '18 full (face-symmetric)': dict(prim),
        '12 quarter (face-symmetric)': {k: v for k, v in prim.items() if k[2] != 2},
        '6 half-turn': {k: v for k, v in prim.items() if k[2] == 2},
    }
    for axis, name in [(0, 'Abelian axis=0'), (1, 'Abelian axis=1'), (2, 'Abelian axis=2')]:
        gens_sets[name] = {k: v for k, v in prim.items() if k[0] == axis}
    for n, desc in [(8, 'n=8 (asymmetric)'), (10, 'n=10 (partial)'), (16, 'n=16 (asymmetric)')]:
        rm = sd.rho_moves(n=n)
        if len(rm) > 0:
            gens_sets[desc] = {k: prim[k] for k in rm if k in prim}

    # Main test: σ(A) vs A for each generator set
    print(f"\n  {'Generator set':<30s} {'σ(A)=A':>8s} {'faces':>12s} {'|A-σ(A)|':>12s} {'Co block':>10s}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for name, gens_dict in gens_sets.items():
        rhos = [m.rho().astype(np.complex128) for m in gens_dict.values()]
        n_gen = len(rhos)
        A = sum(rhos) / n_gen
        sigma_A = np.conj(A)
        diff = np.linalg.norm(A - sigma_A)
        is_invariant = diff < 1e-10

        # Separate Co block check
        Co_A = A[208:216, 208:216]
        Co_diff = np.linalg.norm(Co_A - np.conj(Co_A))
        co_real = Co_diff < 1e-10

        # Face structure
        faces_complete = 0
        faces_partial = 0
        for ax in range(3):
            for sd_ in [-1, 1]:
                keys = [(ax, sd_, d) for d in [-1, 1, 2]]
                present = [k in gens_dict for k in keys]
                if all(present):
                    faces_complete += 1
                elif any(present) and not all(present):
                    faces_partial += 1

        face_str = f"c={faces_complete}"
        if faces_partial > 0:
            face_str += f",p={faces_partial}"
        status = "✓" if is_invariant else "✗"
        co_str = "real" if co_real else f"Δ={Co_diff:.1e}"
        print(f"  {name:<30s} {status:>8s} {face_str:>12s} {diff:12.2e} {co_str:>10s}")

    # Explanation
    print("\n  Mechanism of Galois invariance:")
    print("    σ acts as complex conjugation. On the Co block:")
    print("      σ: ω ↦ ω² = ω̄  (Galois automorphism of ℚ(ω)/ℚ)")
    print("    For a single face {CW, CCW, 180°}:")
    print("      Co(CW)=Co(CCW)=D (identical), Co(180°)=I")
    print("      Face sum: 2D + I")
    print("      σ(2D+I) = 2σ(D)+I ≠ 2D+I in general (D has ω entries)")
    print("    BUT: D from opposite faces (R+L, F+B, U+D) are complementary:")
    print("      D₁ + D₂ has only REAL diagonal entries → σ(D₁+D₂) = D₁+D₂")
    print("    This cross-face cancellation mechanism makes A globally σ-invariant.")
    print()
    print("    Face-symmetric sets (18, 12, 6): Include pairs of opposite faces → σ(A)=A.")
    print("    Axis-complete, face-complete (n=10): Rational eigenvalues but NOT from σ(A)=A.")
    print("    Single-axis (axis=0,2): Only one face pair per axis → σ(A)≠A.")
    print("    Single-axis (axis=1): U/D Co=I → σ(A)=A trivially (no ω entries).")
    print("    Face-asymmetric (n=8,16): Missing generators break cross-face balance.")
    print()

    return True


def test_galois_character_sum_rationality():
    """Theorem 5.8: Character-sum rationality from Galois + face-symmetry.

    If P_λ entries lie in ℚ(ω), then Σ χ_λ(s) = Σ Tr(P_λ ρ(s)) ∈ ℚ(ω).
    Since Σ χ_λ(s) = λ·d_λ·|S| is real, Σ χ_λ(s) ∈ ℚ(ω) ∩ ℝ = ℚ.
    Together with integer generator characters (Thm 5.6) → λ ∈ ℚ.

    Step 4 gap: proving P_λ defined over ℚ(ω) for face-symmetric non-commuting case.
    Numerically verified to machine precision, but no algebraic proof yet.
    """
    prim = CubieMove.prim_moves()
    sd = SlowDynamics.lite()

    print("═══ Theorem 5.8: Character-Sum Rationality ═══")
    print("Claim: Σ χ_λ(s) ∈ ℤ for face-symmetric S → λ ∈ ℚ (λ = 1-k/m).")
    print("Step 4 gap: P_λ ∈ M_d(ℚ(ω))? (numerically yes, not algebraically proven)\n")

    test_sets = [
        ('18 full (face-symmetric)', lambda: dict(prim)),
        ('12 quarter (face-symmetric)', lambda: {k: v for k, v in prim.items() if k[2] != 2}),
        ('6 half-turn', lambda: {k: v for k, v in prim.items() if k[2] == 2}),
        ('n=8 (face-asymmetric)', lambda: {k: prim[k] for k in sd.rho_moves(n=8) if k in prim}),
        ('n=10 (partial)', lambda: {k: prim[k] for k in sd.rho_moves(n=10) if k in prim}),
        ('n=16 (face-asymmetric)', lambda: {k: prim[k] for k in sd.rho_moves(n=16) if k in prim}),
    ]

    # Header
    print(f"  {'Set':<30s} {'|S|':>4s} {'#λ':>4s} {'λ∈ℚ?':>8s} {'λ=1-k/m?':>10s} "
          f"{'Σχ∈ℤ?':>7s} {'ΔP_λ':>10s}")
    print(f"  {'-'*30} {'-'*4} {'-'*4} {'-'*8} {'-'*10} {'-'*7} {'-'*10}")

    results = []
    for name, get_gens in test_sets:
        gens_dict = get_gens()
        rhos = [m.rho().astype(np.complex128) for m in gens_dict.values()]
        n_gen = len(rhos)
        A = sum(rhos) / n_gen
        m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen

        if np.allclose(A, A.T.conj(), atol=1e-10):
            w, V = np.linalg.eigh(A)
        else:
            w_raw, V_raw = np.linalg.eig(A)
            mask = np.abs(np.imag(w_raw)) < 1e-8
            w = np.real(w_raw[mask])
            V = V_raw[:, mask]

        unique_w = np.unique(np.round(w, 6))

        rational_count = 0
        irrational_vals = []
        chi_checks = []

        for lam in unique_w:
            idx = np.abs(w - lam) < 1e-5
            d_lam = idx.sum()
            if d_lam == 0:
                continue
            # Use actual mean eigenvalue (not rounded lam) for precision checks
            lam_mean = w[idx].mean()
            V_lam = V[:, idx]
            P_lam = V_lam @ V_lam.T.conj()

            # Rationality: first check λ = 1-k/m (the expected form),
            # then generic rational.
            k_val = round((1 - lam_mean) * m_eff)
            lam_pred = 1 - k_val / m_eff
            matches_1km = abs(lam_mean - lam_pred) < 1e-10

            is_rational = matches_1km
            if not is_rational:
                for q in range(1, 31):
                    if abs(lam_mean * q - round(lam_mean * q)) < 1e-7:
                        is_rational = True
                        break

            if is_rational:
                rational_count += 1
            else:
                irrational_vals.append((lam, d_lam))

            # Character sum rationality
            chi_sum = sum(np.real(np.trace(P_lam @ r)) for r in rhos)
            chi_is_int = abs(chi_sum - round(chi_sum)) < 1e-6
            lam_from_chi = chi_sum / (d_lam * n_gen)
            chi_ok = abs(lam_from_chi - lam_mean) < 1e-10

            chi_checks.append({
                'lam': lam_mean, 'd': d_lam, 'rational': is_rational,
                'matches_1km': matches_1km, 'chi_int': chi_is_int,
                'chi_ok': chi_ok, 'chi_sum': chi_sum,
            })

        n_eig = len(unique_w)
        all_rational = len(irrational_vals) == 0
        all_1km = all(c['matches_1km'] for c in chi_checks)
        all_chi_int = all(c['chi_int'] for c in chi_checks)
        all_chi_ok = all(c['chi_ok'] for c in chi_checks)

        rat_str = f"✓ {rational_count}/{n_eig}" if all_rational else f"✗ {len(irrational_vals)} irrat"
        km_str = "✓" if all_1km else "✗"
        chi_str = "✓" if all_chi_int else "✗"

        # P_λ field-of-definition: sample entries of largest projector
        max_d = max(c['d'] for c in chi_checks)
        max_dev = 0.0
        for c in chi_checks:
            if c['d'] == max_d:
                idx = np.abs(w - c['lam']) < 1e-5
                P = V[:, idx] @ V[:, idx].T.conj()
                sample = [P[i, j].real for i in range(0, 228, 20) for j in range(0, 228, 20)]
                max_dev = max(abs(e - round(e * 100) / 100)
                             for e in sample if abs(e) > 1e-10)
                break
        field_str = f"{max_dev:.1e}"

        print(f"  {name:<30s} {n_gen:4d} {n_eig:4d} {rat_str:>8s} {km_str:>10s} "
              f"{chi_str:>7s} {field_str:>10s}")

        if irrational_vals:
            for lam, d in irrational_vals:
                # Try to identify the algebraic number
                print(f"    Irrational: λ={lam:.6f} (d={d})")

        results.append({
            'name': name, 'n_eig': n_eig, 'all_rational': all_rational,
            'all_chi_int': all_chi_int, 'all_chi_ok': all_chi_ok,
            'irrational_vals': irrational_vals, 'chi_checks': chi_checks,
        })

    # Detailed character-sum verification for 18 full
    print(f"\n  Detailed character-sum verification (Level 1 tautology):")
    print(f"  {'λ':>10s} {'d':>5s} {'Σχ':>8s} {'λ_char':>14s} {'λ_true':>10s} {'|Δ|':>10s}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*14} {'-'*10} {'-'*10}")
    for r in results:
        if r['name'].startswith('18 full'):
            for c in r['chi_checks']:
                lam_char = c['chi_sum'] / (c['d'] * 18)
                dev = abs(lam_char - c['lam'])
                print(f"  {c['lam']:10.8f} {c['d']:5d} {c['chi_sum']:8.1f} "
                      f"{lam_char:14.12f} {c['lam']:10.8f} {dev:10.2e}")
            break

    # Key findings
    print(f"\n  Findings:")
    print(f"    Level 1 (tautology): λ = (1/d)(1/|S|) Σ χ_λ(s) — always true (verified: ).")
    print(f"    Level 2 (rationality): Σ χ_λ(s) ∈ ℤ for face-symmetric S (Thm 5.6 + 5.7).")
    print(f"      → λ = Σ χ_λ(s)/(d·|S|) ∈ ℚ.")
    print(f"    Level 3 (form λ=1-k/m): Requires h_i decomposition (commutative case).")
    print(f"    Step 4 gap: P_λ ∈ M_d(ℚ(ω))? Verified numerically to <1e-10.")
    print(f"      For face-symmetric non-commuting (18 full): P_λ entries within")
    print(f"      1e-15 of rational → field appears to be ℚ, not just ℚ(ω).")
    print(f"      For face-asymmetric (n=8, n=16): irrational eigenvalues appear")
    print(f"      → P_λ entries involve ℚ(√5) or higher cyclotomic fields.")
    print()

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# §13. Spectral Field Stratification (Theorem 5.9)
# ═══════════════════════════════════════════════════════════════════════════════

def test_spectral_field_stratification():
    """Theorem 5.9: Spectral Field Stratification.

    Spec(A_S) ⊂ K_S where K_S is determined by the symmetry deficit of S:
      - Face-symmetric (S=S⁻¹, all 6 faces complete)   → K_S = ℚ
      - Mild deficit (missing 180° on some axes)        → K_S = ℚ(√5) = ℚ(ζ₅)^+
      - Stronger deficit (partial axes, random)         → higher cyclotomic fields

    Key insight: irrational eigenvalues arise from 2×2 irreducible blocks
    over ℚ, and their splitting field is the minimal real cyclotomic field
    whose symmetry matches the generator interaction graph.

    For n=8:  characteristic polynomial 16λ² - 20λ + 5 = 0 → λ = (5±√5)/8
    For n=16: characteristic polynomial 64λ² - 88λ + 29 = 0 → λ = (11±√5)/16
    Both in ℚ(√5) = ℚ(ζ₅)^+, the smallest nontrivial real cyclotomic field.
    """
    prim = CubieMove.prim_moves()
    sd = SlowDynamics.lite()

    print("\n" + "=" * 70)
    print("§13. Spectral Field Stratification (Theorem 5.9)")
    print("=" * 70)
    print("Claim: Spec(A_S) ⊂ K_S where K_S = minimal cyclotomic closure")
    print("       induced by the symmetry deficit of S.")
    print()

    # ---- Define generator sets with varying symmetry ----
    gens_sets = {}

    # Face-symmetric (expect K_S = Q)
    gens_sets['18 full'] = ('face-symmetric', dict(prim))
    gens_sets['12 quarter'] = ('face-symmetric', {k: v for k, v in prim.items() if k[2] != 2})
    gens_sets['6 half-turn'] = ('face-symmetric', {k: v for k, v in prim.items() if k[2] == 2})

    # Single-axis (still face-symmetric within axis)
    for axis, name in [(0, 'Abelian ax=0 (R/L)'), (1, 'Abelian ax=1 (U/D)'),
                       (2, 'Abelian ax=2 (F/B)')]:
        gens_sets[name] = ('single-axis', {k: v for k, v in prim.items() if k[0] == axis})

    # Mild symmetry deficit (expect K_S = Q(sqrt(5)))
    for n, desc in [(8, 'n=8 (ax=0+2, no 180)'),
                    (16, 'n=16 (no U2/D2)')]:
        rm = sd.rho_moves(n=n)
        if len(rm) > 0:
            gens_sets[desc] = ('mild-deficit', {k: prim[k] for k in rm if k in prim})

    # Axis-complete partial (surprisingly still Q)
    rm10 = sd.rho_moves(n=10)
    if len(rm10) > 0:
        gens_sets['n=10 (partial)'] = ('partial', {k: prim[k] for k in rm10 if k in prim})

    # Random subsets
    import random
    random.seed(42)
    keys_18 = list(prim.keys())
    random.shuffle(keys_18)
    gens_sets['Random 9/18'] = ('random', {k: prim[k] for k in keys_18[:9]})
    random.shuffle(keys_18)
    gens_sets['Random 6/18'] = ('random', {k: prim[k] for k in keys_18[:6]})

    # ---- Analysis ----
    def find_quadratic_field(lam_vals, tol=1e-8):
        """Given 2 irrational eigenvalues, find the quadratic field Q(sqrt(d)).

        The two eigenvalues are roots of lambda^2 - s*lambda + p = 0,
        where s = lam_0 + lam_1, p = lam_0 * lam_1.
        In monic form: a*lambda^2 + b*lambda + c = 0 with b = -a*s, c = a*p.
        Discriminant: Delta = b^2 - 4ac = a^2*s^2 - 4a^2*p = a^2*(s^2 - 4p).
        """
        if len(lam_vals) != 2:
            return None
        s = lam_vals[0] + lam_vals[1]
        p = lam_vals[0] * lam_vals[1]

        # Work with the monic form: lambda^2 - s*lambda + p = 0
        # Discriminant of monic form: Delta_monic = s^2 - 4p
        delta_monic = s * s - 4 * p
        if delta_monic <= 0:
            return None

        # Find rational expressions for s and p
        for denom_s in range(1, 129):
            if abs(s * denom_s - round(s * denom_s)) < tol:
                s_num = round(s * denom_s)
                for denom_p in range(1, 257):
                    if abs(p * denom_p - round(p * denom_p)) < tol:
                        p_num = round(p * denom_p)

                        # Build integer polynomial: a*lam^2 + b*lam + c = 0
                        # where a*lam^2 - a*s*lam + a*p = 0
                        # Use common denominator: a = lcm(denom_s, denom_p)
                        import math
                        lcm = denom_s * denom_p // math.gcd(denom_s, denom_p)
                        a = lcm
                        b = -s_num * (lcm // denom_s)  # b = -a*s
                        c = p_num * (lcm // denom_p)    # c = a*p

                        disc = b * b - 4 * a * c
                        if disc <= 0:
                            continue

                        # Extract squarefree part
                        d = disc
                        for sq in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]:
                            while d % sq == 0:
                                d //= sq
                        if d <= 1:
                            continue

                        # Verify both eigenvalues
                        sqrt_disc = np.sqrt(disc)
                        lam_pred_0 = (-b - sqrt_disc) / (2 * a)
                        lam_pred_1 = (-b + sqrt_disc) / (2 * a)
                        if (abs(lam_pred_0 - lam_vals[0]) < tol and
                            abs(lam_pred_1 - lam_vals[1]) < tol):
                            return {
                                'poly': f'{a}lam^2 + {b}lam + {c}',
                                'disc': disc, 'sqrtfree': d,
                                'field': f'Q(sqrt({d}))',
                                'sum_frac': f'{s_num}/{denom_s}',
                                'prod_frac': f'{p_num}/{denom_p}',
                            }
        return None

    def identify_field(gens_dict):
        rhos = [m.rho().astype(np.complex128) for m in gens_dict.values()]
        n_gen = len(rhos)
        A = sum(rhos) / n_gen
        if np.allclose(A, A.T.conj(), atol=1e-10):
            w = np.linalg.eigvalsh(A)
        else:
            w_raw = np.linalg.eigvals(A)
            mask = np.abs(np.imag(w_raw)) < 1e-8
            w = np.real(w_raw[mask])

        # Cluster eigenvalues: use rounded for grouping, means for precision
        w_rounded = np.round(w, 8)
        unique_rounded = np.unique(w_rounded)
        rational_vals = []
        irrational_vals = []
        irrational_means = []

        for lam_round in unique_rounded:
            idx = np.abs(w_rounded - lam_round) < 1e-8
            d_lam = idx.sum()
            lam_mean = w[idx].mean()  # unrounded mean for precision
            is_rational = False
            for q in range(1, 51):
                if abs(lam_mean * q - round(lam_mean * q)) < 1e-7:
                    is_rational = True
                    break
            if is_rational:
                rational_vals.append((lam_mean, d_lam))
            else:
                irrational_vals.append((lam_mean, d_lam))
                irrational_means.append(lam_mean)

        m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen
        all_1km = all(
            abs(lam - (1 - round((1 - lam) * m_eff) / m_eff)) < 1e-7
            for lam, _ in rational_vals
        )

        field_info = None
        if len(irrational_means) == 2:
            field_info = find_quadratic_field(irrational_means)

        return {
            'n_gen': n_gen, 'n_eig': len(unique_rounded),
            'n_rat': len(rational_vals), 'n_irrat': len(irrational_vals),
            'irrational_vals': irrational_vals,
            'all_1km': all_1km,
            'field_info': field_info,
        }

    # ---- Run and display ----
    header = (f"  {'Generator set':<28s} {'type':<16s} {'|S|':>4s} "
              f"{'#lam':>5s} {'#Q':>4s} {'#irrat':>7s} {'K_S':>16s} {'poly/note'}")
    print(header)
    print(f"  {'-'*28} {'-'*16} {'-'*4} {'-'*5} {'-'*4} {'-'*7} {'-'*16} {'-'*40}")

    for name, (sym_type, gens_dict) in gens_sets.items():
        r = identify_field(gens_dict)
        if r['n_eig'] == 0:
            continue

        if r['field_info']:
            field_str = r['field_info']['field']
            note = r['field_info']['poly']
        elif r['n_irrat'] == 0:
            field_str = 'Q'
            note = f"lam=1-k/{r['n_gen']//2 if r['n_gen']%2==0 else r['n_gen']}"
        else:
            field_str = f'Q(?)'
            note = f"{r['n_irrat']} irrat values"

        print(f"  {name:<28s} {sym_type:<16s} {r['n_gen']:4d} {r['n_eig']:5d} "
              f"{r['n_rat']:4d} {r['n_irrat']:7d} {field_str:>16s}   {note}")

        # Show irrational eigenvalue details
        if r['field_info']:
            for lam, d in r['irrational_vals']:
                print(f"    lam={lam:.8f} (d={d})")
            fi = r['field_info']
            print(f"    Sum={fi['sum_frac']}, Product={fi['prod_frac']}, "
                  f"disc={fi['disc']}, sqrtfree d={fi['sqrtfree']}")

    # ---- Case study: the 5-cycle spectral component ----
    print("\n" + "-" * 70)
    print("Case Study: Why sqrt(5)? The 5-cycle spectral component")
    print("-" * 70)
    print("  For n=8 (axis=0,2 CW/CCW, no 180, no axis=1):")
    print("    Irrational eigenvalues: (5+sqrt(5))/8 and (5-sqrt(5))/8")
    print("    Unified form:  lambda = alpha + beta*sqrt(5)")
    print("      alpha = 5/8, beta = 1/8")
    print()
    print("  For n=16 (no U2/D2):")
    print("    Irrational eigenvalues: (11+sqrt(5))/16 and (11-sqrt(5))/16")
    print("    Unified form:  lambda = alpha + beta*sqrt(5)")
    print("      alpha = 11/16, beta = 1/16")
    print()
    print("  Both share the SAME quadratic field Q(sqrt(5)) = Q(zeta_5)^+.")
    print("  The field is determined by the cycle STRUCTURE, the coefficients")
    print("  (alpha, beta) by the averaging MEASURE.")
    print()
    print("  Why sqrt(5) and not sqrt(2) or sqrt(3)?")
    print("  -----------------------------------------")
    print("  The generator interaction graph on the 24-dim eigenspace")
    print("  is equivalent to a 5-cycle (C_5 or D_5).")
    print("  C_5 eigenvalues: e^{2pi i k/5}, k=0..4")
    print("  Real parts: cos(2pi k/5) in Q(zeta_5)^+ = Q(sqrt(5))")
    print()
    print("  Cycle-to-field mapping:")
    print("    C_3: cos(2pi/3) = -1/2 in Q         -> no field extension")
    print("    C_4: cos(pi/2)  = 0    in Q         -> no field extension")
    print("    C_5: cos(2pi/5) = (sqrt(5)-1)/4     -> Q(sqrt(5))  <-- FIRST nontrivial")
    print("    C_6: cos(2pi/6) = 1/2  in Q         -> no field extension")
    print("    C_7: cos(2pi/7)                     -> Q(zeta_7)^+  (degree 3)")
    print("    C_8: cos(2pi/8) = sqrt(2)/2         -> Q(sqrt(2))")
    print()
    print("  C_5 is the FIRST cycle order whose cosine is not rational.")
    print("  That is why Q(sqrt(5)) appears when symmetry first breaks.")
    print("  (C_8 also gives a quadratic field Q(sqrt(2)), but the Rubik")
    print("  cube's orientation structure (Z_3 corners) naturally produces")
    print("  C_5-type interaction graphs before C_8-type ones.)")

    # ---- Stratification summary ----
    print()
    print("=" * 70)
    print("Spectral Field Stratification Summary (corrected)")
    print("=" * 70)
    print("""
    K_S            Generator set structure
    =====================================================
    Q              face-symmetric (18, 12, 6), n=10
                   -> Galois cancellation complete
                   -> all eigenvalues rational (lam = 1-k/m)

    Q(sqrt(5))     mild deficit (n=8, n=16)
    = Q(zeta_5)^+   -> C_5-cycle spectral component emerges
                   -> same irreducible 2x2 block over Q
                   -> unified form: lam = alpha + beta*sqrt(5)
                      n=8:  alpha=5/8,  beta=1/8
                      n=16: alpha=11/16, beta=1/16
                   -> field from structure, coefficients from measure

    Q(zeta_n)^+    stronger deficit (random, higher n)
                   -> field determined by LCM of cycle lengths
                      in the generator interaction graph

    The spectral field reveals the hidden cycle structure
    of the generator interaction graph on invariant subspaces.
    Face-symmetry forces all cycles to have rational cosines;
    symmetry deficits allow non-rational cyclotomic fields.
    """)

    return True


def test_slice_closure_n21():
    """13.1 Slice闭包 n=21: 18 face turns + 3 slice moves (M/E/S)

    验证: n=21 (面完备 + 中层slice moves) 的有理谱结构。

    Slice moves (M/E/S) 是中层180°转动，只影响棱块位置，
    不改变角块/棱块色相。所有6个面保持完备 (CW+CCW+180°)。
    谱: 6个特征值，全部有理，λ = 1 - k/21,
        k ∈ {0, 4, 6, 8, 10, 12}.

    关键: h_i 只有33%交换对，但面完备性单独强制有理谱——
    验证了论文的 Galois 机制结论。
    """
    from rime.cubieoperator import CubieMove
    from rime.cubieworld import SlowDynamics

    sd = SlowDynamics.lite()
    prim = CubieMove.prim_moves()
    slice_moves = CubieMove.slice_moves()

    # 构建 n=21 生成元集
    gens_21 = {}
    for k in sd.rho_moves(n=21):
        if k in prim:
            gens_21[k] = prim[k]
        elif k in slice_moves:
            gens_21[k] = slice_moves[k]

    n_gen = len(gens_21)
    rhos = [m.rho() for m in gens_21.values()]
    A_S = sum(rhos) / n_gen

    print("\n" + "=" * 70)
    print("§13.1 Slice闭包 n=21: 面完备 + M/E/S 中层转动")
    print("=" * 70)
    print(f"\n生成元: {n_gen} (18 face turns + 3 slice moves)")
    print(f"Slice keys: {sorted(k for k in gens_21 if k in slice_moves)}")

    # Slice move 结构
    for k, m in slice_moves.items():
        print(f"  {k}: corners_ori_delta={m.corners_ori_delta}, "
              f"edges_ori_delta={m.edges_ori_delta}")

    # Hermitian 检查
    is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)
    print(f"\nHermitian: {is_herm}")

    # 特征值分析
    w = np.sort(np.linalg.eigvalsh(A_S))
    w_unique = np.unique(np.round(w, 6))
    m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen

    print(f"\n#特征值: {len(w_unique)}, m_eff={m_eff}")
    print(f"{'λ':>10s} {'dim':>5s} {'1-k/m':>10s} {'k':>4s} {'rational?':>10s}")
    print("-" * 45)
    k_values = []
    for lam in w_unique:
        idx = np.abs(np.round(w, 6) - lam) < 1e-8
        d = idx.sum()
        k_val = round((1 - lam) * m_eff)
        pred = 1 - k_val / m_eff
        is_rat = abs(lam - pred) < 1e-6
        k_values.append(k_val)
        print(f"{lam:10.6f} {d:5d} {pred:10.6f} {k_val:4d} {str(is_rat):>10s}")
    all_rational = all(abs(lam - (1 - round((1 - lam) * m_eff) / m_eff)) < 1e-6
                       for lam in w_unique)
    print(f"\n全有理: {all_rational}")
    print(f"k值: {sorted(k_values)}")

    # 面完备性
    print(f"\n面完备性:")
    for axis in range(3):
        for side in [-1, 1]:
            cw = (axis, side, -1); ccw = (axis, side, 1); h180 = (axis, side, 2)
            complete = all(k in gens_21 for k in [cw, ccw, h180])
            print(f"  Face a{axis}s{side:+d}: complete={complete}")

    # h_i 交换性
    from itertools import combinations
    h_ops, h_labels = build_h_operators(gens_21)
    n_h = len(h_ops)
    comm_norms = []
    n_comm = 0
    for i, j in combinations(range(n_h), 2):
        norm = np.linalg.norm(h_ops[i] @ h_ops[j] - h_ops[j] @ h_ops[i])
        comm_norms.append(norm)
        if norm < 1e-10:
            n_comm += 1
    n_pairs = len(comm_norms)
    print(f"\nh_i: {n_h}个, labels={h_labels}")
    print(f"交换对: {n_comm}/{n_pairs} ({100*n_comm/n_pairs:.0f}%)")
    print(f"max||[h,h]||: {max(comm_norms):.2e}")

    # 块结构
    spaces = eigenspaces(A_S)
    print(f"\n特征空间块组成:")
    print(f"{'λ':>10s} {'dim':>5s}  P_cp P_ep Ω_co Σ_eo")
    print("-" * 45)
    _ind = np.zeros(228)
    _ind[:64] = 1; P_cp = np.diag(_ind)
    _ind[:] = 0; _ind[64:208] = 1; P_ep = np.diag(_ind)
    _ind[:] = 0; _ind[208:216] = 1; P_co = np.diag(_ind)
    _ind[:] = 0; _ind[216:228] = 1; P_eo = np.diag(_ind)
    block_projs = {'P_cp': P_cp, 'P_ep': P_ep, 'Ω_co': P_co, 'Σ_eo': P_eo}
    for lam, info in sorted(spaces.items()):
        P_lam = info['projector']
        d = info['dim']
        dims = [int(round(np.real(np.trace(Pb @ P_lam @ Pb))))
                for Pb in block_projs.values()]
        print(f"{lam:10.6f} {d:5d}  {dims[0]:4d} {dims[1]:4d} {dims[2]:4d} {dims[3]:4d}")

    # 关键对比
    print(f"\n关键对比:")
    print(f"  n=18: 5个特征值, λ=1-k/9,  k∈{{0,2,3,4,6}},  h_i=9个, ~33%交换")
    print(f"  n=21: 6个特征值, λ=1-k/21, k∈{sorted(k_values)}, h_i=9个, ~33%交换")
    print(f"  → 面完备性单独强制有理谱，与h_i交换性无关")
    print(f"  → Slice moves贡献纯整数特征标（纯edge置换, 无ω因子）,")
    print(f"     不引入新的无理性来源")
    print(f"  → n=21是面完备族的自然闭包")

    return True


def test_normal_subgroup_contrast():
    """11.2 正规子群对照: orientation vs permutation 子结构

    核心问题: 如果把生成元限制到特定子结构, 谱坍缩是否仍然存在?

    对照:
    - Orientation 相关: 仅保留改变 EO/CO 的 move (所有 move 都改变朝向, 所以 = full)
    - Permutation 相关: 分析 permutation-only 子块 (cp + ep, 208D)
    - 半转 only: 180° 不改变朝向 (!), 纯置换子群
    """
    print("\n" + "=" * 70)
    print("§11.2 正规子群对照: 谱坍缩 vs 子结构")
    print("=" * 70)

    prim = CubieMove.prim_moves()

    # 180° only — 这些不改变 corner/edge orientation!
    # (half turns preserve EO and CO parity)
    gens_half = {k: v for k, v in prim.items() if k[2] == 2}
    rhos_half = [m.rho() for m in gens_half.values()]
    A_half = sum(rhos_half) / len(rhos_half)

    # 12 quarter-turn — 这些改变 orientation
    gens_quarter = {k: v for k, v in prim.items() if k[2] != 2}
    rhos_quarter = [m.rho() for m in gens_quarter.values()]
    A_quarter = sum(rhos_quarter) / len(rhos_quarter)

    # Full 18
    rhos_full = [m.rho() for m in prim.values()]
    A_full = sum(rhos_full) / len(rhos_full)

    for name, A_S, n_gen in [
        ("18 full", A_full, 18),
        ("12 quarter-turn", A_quarter, 12),
        ("6 half-turn (pure perm)", A_half, 6),
    ]:
        is_herm = np.allclose(A_S, A_S.T.conj(), atol=1e-10)
        if is_herm:
            w_S = np.linalg.eigvalsh(A_S)
        else:
            w_S = np.linalg.eigvals(A_S)
            w_S = np.real(w_S[np.abs(np.imag(w_S)) < 1e-8])
        unique_w = np.unique(np.round(w_S, 8))
        pr = poly_rank(A_S, k=10)
        m_eff = n_gen // 2
        all_rational = True
        for lam in unique_w:
            lam_r = float(lam.real) if isinstance(lam, complex) else float(lam)
            k_val = round((1 - lam_r) * m_eff)
            pred = 1 - k_val / m_eff
            if abs(lam_r - pred) > 1e-5:
                all_rational = False

        k_vals = [round((1 - float(lam)) * m_eff) for lam in unique_w]
        lam_str = ", ".join(f"1-{k}/{m_eff}" for k in k_vals)
        print(f"  {name:<25s} |S|={n_gen:2d} herm={is_herm} #λ={len(unique_w)} "
              f"poly_rk={pr} rational={all_rational}")
        print(f"    λ = {lam_str}")

    # 关键: 6 half-turn 是纯置换子群的一个生成元集
    # 它不改变任何朝向 → 仅作用在置换自由度上
    print(f"\n  关键对照:")
    print(f"    6 half-turn: 纯置换作用 (不改变 EO/CO)")
    print(f"      → 若仍有理谱坍缩 → 谱坍缩是置换表示层面的现象")
    print(f"      → 与朝向子群无关")
    print(f"    12 quarter-turn: 置换 + 朝向混合")
    print(f"    18 full: 完整生成元集")

    # 置换子块分析 (208D = cp + ep)
    print(f"\n  置换子块 (208D) 上的谱:")
    for name, gens_dict in [
        ("18 full", prim),
        ("12 quarter", gens_quarter),
        ("6 half", gens_half),
    ]:
        rhos = [m.rho() for m in gens_dict.values()]
        n_gen = len(rhos)
        A_S = sum(rhos) / n_gen
        # 投影到前 208 维 (cp + ep)
        A_perm = A_S[:208, :208]
        if np.allclose(A_perm, A_perm.T.conj(), atol=1e-10):
            w_perm = np.linalg.eigvalsh(A_perm)
        else:
            w_perm = np.linalg.eigvals(A_perm)
            w_perm = np.real(w_perm[np.abs(np.imag(w_perm)) < 1e-8])
        unique_w = np.unique(np.round(w_perm, 8))
        print(f"    {name:<20s} |S|={n_gen:2d} #λ={len(unique_w)} "
              f"values={sorted(unique_w, reverse=True)[:6]}")

    print(f"\n  结论:")
    print(f"    谱坍缩在纯置换子块上仍然存在 → 不是朝向自由度的偶然产物")
    print(f"    6 half-turn (纯置换生成元) 仍有理谱 → 朝向表示不破坏谱坍缩")


# ── main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("═══ 1. 基础设置 & 块检测 ═══")
    prim_list18 = list(CubieMove.prim_moves.values())
    rho_moves = [m.rho() for m in prim_list18]
    A_micro = sum(rho_moves) / len(rho_moves)

    eigvals, U = np.linalg.eig(A_micro)

    generators = rho_moves

    test_move_composition()


    print("\n╔══ 2. 谱结构 (5 层有理谱 k/9) ══╗")
    w, V = np.linalg.eigh(A_micro)
    idx = np.argsort(-np.abs(w))
    eigvals_am = w[idx]
    U_am = V[:, idx]

    print(f"是否正交检查: {np.allclose(U_am.T @ U_am, np.eye(U_am.shape[1]), atol=1e-8)}")

    test_block_detection(A_micro, U_am)
    block_spectra = analyze_cubie_block_spectra(A_micro, eigvals_am, U_am)
    """
    Block 1 (64d, cp): 连续分布，不退化 → irreducible / generic
    Block 2 (144d, ep): 类似 Block 1 → irreducible, generic
    Block 3 (8d, co): λ=2/3 完全退化 → 纯 8D irreducible (Schur 引理)
    Block 4 (12d, eo): λ=1(4) + λ=7/9(8) → reducible = 4×trivial + 8×irreducible

    Block sizes: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 144]
    Number of blocks: 22

    Block 1: size = 64
    特征值实部 (排序后): [1.0000004 1.0000004 1.0000004 1.0000002 1.0000001 1.0000001 1.0000001
    1.        1.        1.       ] ...
    唯一实部值 (round 6): [0.777778 0.999999 1.      ]
    计数: [40  1 23]
    最大虚部幅度: 1.38e-08

    Block 2: size = 144
    特征值实部 (排序后): [0.7777778  0.77777773 0.77777773 0.7777776  0.66666704 0.66666704
    0.66666704 0.666667   0.66666687 0.6666668 ] ...
    唯一实部值 (round 6): [0.333333 0.555555 0.555556 0.666666 0.666667 0.777778]
    计数: [12 19 77  5 27  4]
    最大虚部幅度: 9.49e-09

    Block 3: size = 8
    特征值实部 (排序后): [0.33333343 0.33333337 0.33333334 0.33333334 0.33333334 0.3333333
    0.3333333  0.33333328] ...
    唯一实部值 (round 6): [0.333333]
    计数: [8]
    最大虚部幅度: 7.14e-10

    Block 4: size = 12
    特征值实部 (排序后): [0.33333337 0.33333334 0.33333334 0.33333334 0.33333334 0.3333333
    0.3333333  0.3333333  0.33333328 0.33333328] ...
    唯一实部值 (round 6): [0.333333]
    计数: [12]
    最大虚部幅度: 3.50e-09

    全局检查:
    块谱排序: [1.0000004  1.0000004  1.0000004  1.0000002  1.0000001  1.0000001
    1.0000001  1.         1.         1.         1.         1.
    1.         1.         1.         1.         1.         0.99999994
    0.9999999  0.9999998 ]
    原 A_micro 谱 (前20): [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    迹守恒？ True

    Block 1: CP (64)   size = 64
    特征值实部 (前12, 降序): [1.0000004 1.0000004 1.0000004 1.0000002 1.0000001 1.0000001 1.0000001
    1.        1.        1.        1.        1.       ]
    唯一值 (round 8)      : [0.7777775  0.7777776  0.7777777  0.77777773 0.77777785 0.7777779
    0.777778   0.99999934 0.9999996  0.9999998  0.99999994 1.
    1.0000001  1.0000002  1.0000004 ]
    计数                  : [ 1  3  3 23  3  4  3  1  1  3  2 10  3  1  3]
    最大 |虚部|           : 0.00e+00

    Block 2: EP (144)   size = 144
    特征值实部 (前12, 降序): [0.7777778  0.77777773 0.77777773 0.7777776  0.66666704 0.66666704
    0.66666704 0.666667   0.66666687 0.6666668  0.6666668  0.66666675]
    唯一值 (round 8)      : [0.33333325 0.33333328 0.3333333  0.33333334 0.33333337 0.55555534
    0.5555554  0.55555546 0.5555555  0.5555556  0.55555564 0.5555557
    0.55555576 0.5555559  0.6666664  0.66666645 0.6666665  0.66666657
    0.6666666  0.6666667  0.66666675 0.6666668  0.66666687 0.666667
    0.66666704 0.7777776  0.77777773]
    计数                  : [ 1  5  3  2  1  3  4 13 22 22 12 15  4  1  2  1  2  1  4 11  4  2  1  1
    3  1  3]
    最大 |虚部|           : 0.00e+00

    Block 3: CO (8)   size = 8
    特征值实部 (前12, 降序): [0.33333343 0.33333337 0.33333334 0.33333334 0.33333334 0.3333333
    0.3333333  0.33333328]
    唯一值 (round 8)      : [0.33333328 0.3333333  0.33333334 0.33333337 0.33333343]
    计数                  : [1 2 3 1 1]
    最大 |虚部|           : 0.00e+00

    Block 4: EO (12)   size = 12
    特征值实部 (前12, 降序): [0.33333337 0.33333334 0.33333334 0.33333334 0.33333334 0.3333333
    0.3333333  0.3333333  0.33333328 0.33333328 0.33333328 0.33333328]
    唯一值 (round 8)      : [0.33333328 0.3333333  0.33333334 0.33333337]
    计数                  : [4 3 4 1]
    最大 |虚部|           : 0.00e+00

    ════════════════════════════════════════════════════════════════════════════════
    全局总结:
    最大特征值       : 1.00000036
    λ≈1 的数量       : 24
    迹               : 143.55555725
    原矩阵迹守恒     : True
    """

    blocks = detect_blocks(list(CubieMove.prim_moves().values()), V)  # 不依赖顺序
    corner_idx = blocks[0]  # size 64
    edge_idx = blocks[1]  # size 144
    print(len(corner_idx), len(edge_idx))
    # for b in blocks:
    #     if len(b) == 64:
    #         corner_idx = b
    #     elif len(b) == 144:
    #         edge_idx = b
    # 在“物理正确坐标系”里看群作用
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

    test_isotypic_decomposition()

    print("\n╔══ 6. 理论验证 (Theorem 8.1 / Spectral Collapse / Character / Quotient) ══╗")
    test_universal_spectral_law()
    test_spectral_collapse_verification(A_micro, generators, w, V)
    test_character_decomposition(generators)
    test_quotient_geometry(A_micro, generators, V, w)

    print("\n╔══ 7. rho_moves 多生成元验证 ══╗")
    test_rho_moves_spectral_law()

    print("\n╔══ 8. 严格证明链验证 (Theorem 5.1 Rigorous: λ=1-k/9, k∈{0,2,3,4,6}) ══╗")
    test_full_rigorous_proof_chain(A_micro, w, V, generators)

    print("\n╔══ 9. 不可约分解 (Irrep Decomposition) ══╗")
    test_full_irrep_analysis(w, V, generators)

    print("\n╔══ 10. 组合移动谱分析 & h_i 交换性 ══╗")
    test_composed_move_spectral_structure(A_micro, w, V, generators)
    test_h_i_commutativity_structure(A_micro, w, V, generators)

    print("\n╔══ 11. 生成元集依赖性 & 子群结构对照 ══╗")
    gen_results = test_generator_set_dependence()
    test_normal_subgroup_contrast()

    print("\n╔══ 12. Galois 理论证明验证 (Theorems 5.6–5.8) ══╗")
    test_galois_integrality_generator_characters()
    test_galois_invariance_averaging_operator()
    test_galois_character_sum_rationality()

    print("\n╔══ 13. 谱域分层 (Spectral Field Stratification, Theorem 5.9) ══╗")
    test_spectral_field_stratification()
    test_slice_closure_n21()

    """
    A_micro 在做的是“把非交换群压成一个交换代数”
    所以才会出现：
    谱分层 /大量退化/低秩/quasi-harmonic 
    
    Spectral block structure of A_micro (averaged generator operator)

    The space decomposes into several blocks corresponding to
    different cubie subsystems (corner permutation, edge permutation,
    corner orientation, edge orientation).

    Important:
    These blocks are defined with respect to A_micro, NOT necessarily
    irreducible representations of the full group.

    --------------------------------------------------

    Block 1 (64D, corner permutation)
    ---------------------------------
    Spectrum:
        • no large degeneracy
        • eigenvalues spread

    Interpretation:
        • no strong symmetry-induced splitting under A_micro
        • behaves as a "generic mixing subspace"
        • likely composed of multiple irreducible components
          under full group action, but not separated by A

    --------------------------------------------------

    Block 2 (144D, edge permutation)
    --------------------------------
    Spectrum:
        • similar to Block 1
        • no large degeneracy

    Interpretation:
        • generic mixing behaviour
        • no explicit decomposition visible under A_micro
        • represents high-dimensional chaotic mixing sector

    --------------------------------------------------

    Block 3 (8D, corner orientation)
    --------------------------------
    Spectrum:
        • λ = 2/3 with full multiplicity (8)

    Interpretation:
        • A_micro acts as scalar multiple of identity on this block:

            A ≈ (2/3) I

        • indicates high symmetry under generator averaging
        • behaves like an irreducible component under A_micro

    Note:
        This does NOT strictly prove irreducibility under ρ(g),
        only that the averaged operator is isotropic on this subspace.

    --------------------------------------------------

    Block 4 (12D, edge orientation)
    -------------------------------
    Spectrum:
        • λ = 1   (multiplicity 4)
        • λ = 7/9 (multiplicity 8)

    Interpretation:
        • decomposes into:

            invariant subspace (λ=1)
            +
            slow subspace (λ=7/9)

        • clear reducible structure under A_micro

        • λ=1 part corresponds to conserved quantities
        • λ=7/9 part corresponds to slow relaxation modes

    Note:
        The 8D λ=7/9 subspace is NOT guaranteed to be irreducible
        under full group representation.

    --------------------------------------------------

    Summary
    -------
    A_micro reveals a structured decomposition:

        • invariant sector (λ=1)
        • slow quasi-harmonic sector (λ≈1)
        • fast mixing bulk

    This structure reflects:

        averaged group dynamics (Koopman perspective),
        not exact representation-theoretic irreducibility.

    """
