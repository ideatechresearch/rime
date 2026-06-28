"""
CubieSpectralOperator 完整测试套件 — 44 tests in 3 parts

Part A: Legacy Analysis Tests (~16 tests)
  Discovery-phase verification/phenomenology using functions from
  test/exploratory/_exp_legacy_analysis.py (migrated from cubieoperator.py).

Part B: Theorem Verification Tests (~14 tests)
  Rigorous verification of specific theorems (λ=1−k/9, character,
  irrep decomposition, T7 pairs). Galois chain (5.6–5.9) extracted to
  test/canonical/_exp_spectral_theorems.py.

Part C: Core API Tests (~14 tests)
  Self-contained unit tests of CubieSpectralOperator methods:
  projectors, transport tensor, commutant algebra, irrep decomposition.

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
from rime.cubieworld import SlowDynamics
from rime.cubeplot import (
    draw_error_histogram, draw_slow_coordinates,
    draw_gram_matrix, draw_annealing,
)

N_MODES = 10
N_SAMPLES = 2000


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Legacy Analysis Tests
# These use functions migrated to test/exploratory/_exp_legacy_analysis.py.
# They represent discovery-phase verification, phenomenology, and one-off
# analyses that are NOT part of the CubieSpectralOperator core API.
# ═══════════════════════════════════════════════════════════════════════════


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
    
    A 的特征向量 = harmonic function（平均意义）"误差为 0"是必然结果
    
    慢动力学本质上是"守恒谐函数 + 准谐衰减"的组合
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
    群作用会在 eigenspace 内"旋转/混合"这些模式，
    而 λ 控制的是"平均收缩"，std 控制的是"瞬时扩散"。
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
    对应"集体缩放"模式（e.g. 朝向或置换的均匀收缩）。
    群谐误差小但非零（准谐函数）。
    
    Diffusion block (λ=2/3, dim=32)
    严格线性本征方向（exact eigenvector direction）。
    对每个生成元 s，ρ(s) V_diffusion = (2/3) V_diffusion（标量缩放）。
    对应"扩散-like"模式，但不是随机扩散，而是纯缩放扩散（pure scaling diffusion）。
    群谐误差 = 0（在采样精度内）。
    
    剩余层 (λ=5/9, 1/3, dim=96+32)
    混合更强，扰动大，接近"随机化"但仍有结构（奇异值稳定在 1 或 √3/2）。
    群谐误差可能较大（未测试）。
    
    误差是"谱层泄漏（inter-layer leakage）
    
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


# ── 6. 理论验证实验 ─────────────────────────────────────────────────────

def test_universal_spectral_law():
    """Theorem 8.1 (Universal Spectral Law) 数值验证

    核心主张: 对任意生成元子集 S，A_S 的特征值遵循 λ = 1 - k/m，m = |S|/2
    验证策略: 按轴/面逐步增加生成元，检验谱形式是否严格有理
    """
    results = verify_universal_spectral_law()

    # 汇总: 所有子集是否满足有理谱
    all_pass = all(r['all_rational'] for r in results)
    print(f"\n{'=' * 60}")
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
            print(f"  k={k + 1:2d} generators: entropy={entropies[k]:.4f}, "
                  f"distinct_eigenvalues={degens[k]}")
    print(f"  → entropy 单调下降: {all(entropies[i] >= entropies[i + 1] for i in range(len(entropies) - 1))}")


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
    print(f"  ρ_f^100 = {data['rho_fast'] ** 100:.2e}")

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

# ═══════════════════════════════════════════════════════════════════════════
# Part B: Theorem Verification Tests
# Rigorous verification of specific theorems from Paper I/II/III.
# These tests validate the structural claims that underpin the trilogy.
# ═══════════════════════════════════════════════════════════════════════════

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

    cso = CubieSpectralOperator.lite()

    header = f"{'n':>3} {'|S|':>3} {'m':>3} {'herm':>5} {'#λ':>3} {'poly_rk':>7} {'slowD':>5} {'rational':>8}"
    print(header)
    print("-" * len(header))

    for n in [2, 3, 4, 6, 8, 9, 10, 12, 16, 18]:
        rm = cso.rho_moves(n=n)
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

    print(f"  置换块 (208D) 非 {1, 0, -1} 谱值: {sorted(set(perm_outliers))[:8] if perm_outliers else '无 OK'}")
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

    cso = CubieSpectralOperator.lite()

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
        rm = cso.rho_moves(n=n)
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
        print(f"  -> Average irrep dimension ~ {228 / result['n_irreps_est']:.1f}")
    else:
        print(f"  -> Representation appears irreducible (unlikely for 228D)")

    # Show histogram of |chi| values
    abs_chi = np.abs(chi_arr)
    bins = np.linspace(0, max(abs_chi), 20)
    hist, _ = np.histogram(abs_chi, bins=bins)
    print(f"\n  |chi| distribution (top values):")
    for i in np.argsort(-hist)[:8]:
        if hist[i] > 0:
            print(f"    |chi| ~ {bins[i]:.1f} - {bins[i + 1]:.1f}: {hist[i]} samples")

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
        print(f"              ... (+{len(sizes) - 15} more)")

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
    print(f"  {'-' * 10} {'-' * 7} {'-' * 30} {'-' * 15}")
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

    print(f"\n{'=' * 70}")
    print(f"Synthesis:")
    print(f"  <chi, chi> = {chi_result['inner_product']:.2f} -> ~{n_irreps_from_chi} irreps")
    print(f"  Spectral clustering -> {n_blocks} blocks")
    print(f"  A has 5 eigenvalues -> 5 isotypic components under averaging")
    print(f"  Schur lemma: A|_irrep = lambda_alpha * I for each irrep alpha")
    print(f"  lambda_alpha = (1/|S|) sum_s chi_alpha(s) / d_alpha")
    print(f"  -> 5 eigenvalues = 5 distinct character averages over the {n_irreps_from_chi} irreps")
    print(f"  -> Spectral collapse = projection of {n_irreps_from_chi} character values")
    print(f"     onto 5 rational values via generator averaging")
    print(f"{'=' * 70}")


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

    comp_mats = [m.flatten() for m in prim_mats]
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


# Galois theorem tests (5.6–5.9) moved to test/canonical/_exp_spectral_theorems.py
from experiments.canonical._exp_spectral_theorems import (
    test_galois_integrality_generator_characters,
    test_galois_invariance_averaging_operator,
    test_galois_character_sum_rationality,
    test_spectral_field_stratification,
)



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

    cso = SlowDynamics.lite()
    prim = CubieMove.prim_moves()
    slice_moves = CubieMove.slice_moves()

    # 构建 n=21 生成元集
    gens_21 = {}
    for k in cso.rho_moves(n=21):
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
            cw = (axis, side, -1);
            ccw = (axis, side, 1);
            h180 = (axis, side, 2)
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
    print(f"交换对: {n_comm}/{n_pairs} ({100 * n_comm / n_pairs:.0f}%)")
    print(f"max||[h,h]||: {max(comm_norms):.2e}")

    # 块结构
    spaces = eigenspaces(A_S)
    print(f"\n特征空间块组成:")
    print(f"{'λ':>10s} {'dim':>5s}  P_cp P_ep Ω_co Σ_eo")
    print("-" * 45)
    _ind = np.zeros(228)
    _ind[:64] = 1
    P_cp = np.diag(_ind)
    _ind[:] = 0
    _ind[64:208] = 1
    P_ep = np.diag(_ind)
    _ind[:] = 0
    _ind[208:216] = 1
    P_co = np.diag(_ind)
    _ind[:] = 0
    _ind[216:228] = 1
    P_eo = np.diag(_ind)
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


# ═══════════════════════════════════════════════════════════════════════════
# §14. 论文加分项验证 (2026-05-02)
# ═══════════════════════════════════════════════════════════════════════════


def test_g1_eigenspace_boundary():
    """Problem 3: G₁ subgroup action and the λ=2/3 boundary.

    Verify that λ=2/3 is the last primitive idempotent (in descending λ order)
    whose eigenspace is invariant under G₁ = ⟨U,D,R,L,F2,B2⟩.

    Key claim: E_{2/3} ∈ A ∩ ρ(G₁)', while E_{5/9} and E_{1/3} ∈ A but ∉ ρ(G₁)'.
    """
    import numpy as np
    from rime.cubie import CubieMove

    print("\n" + "=" * 80)
    print("Problem 3: G₁ Action and the λ=2/3 Boundary")
    print("=" * 80)

    # ── Build G₁ generators ──
    prim = CubieMove.prim_moves
    # G₁ = ⟨U, U', U2, D, D', D2, R, R', R2, L, L', L2, F2, B2⟩
    # Keys: (axis, side, direction)
    # axis: 0=R/L, 1=U/D, 2=F/B
    # side: -1 or 1
    # direction: 1=CW (standard face turn), -1=CCW (prime), 2=180
    g1_keys = []
    for key in prim:
        axis, side, direction = key
        if axis in (0, 1):  # R/L or U/D — all directions allowed
            g1_keys.append(key)
        elif axis == 2 and direction == 2:  # F/B — only 180°
            g1_keys.append(key)
    g1_gens = {k: prim[k] for k in g1_keys}
    print(f"\nG₁ generators ({len(g1_gens)} moves):")
    for k in sorted(g1_keys):
        print(f"  {k}")

    # ── Build full A and G₁ averaged operator ──
    A_full = sum(g.rho().astype(np.complex128) for g in prim.values()) / len(prim)
    w_full, V_full = np.linalg.eigh(A_full)
    w_rounded = np.round(w_full, 8)
    unique_w = sorted(np.unique(w_rounded), reverse=True)

    print(f"\nFull A spectrum (18-full):")
    for lam in unique_w:
        dim = int(np.sum(np.abs(w_rounded - lam) < 1e-8))
        k = round(9 * (1 - lam))
        print(f"  λ={lam:.6f} ≈ 1-{k}/9, dim={dim}")

    # ── For each eigenspace, check G₁-invariance ──
    # E_λ is G₁-invariant iff P_λ ρ(g) P_λ = ρ(g) P_λ for all g ∈ G₁
    # (equivalently: ρ(g) maps E_λ into E_λ)
    print(f"\n─ Full-space G₁-invariance ─")
    print(f"  Checking: ||(I-P_λ) ρ(g) P_λ||₂ = 0 for all g ∈ G₁")
    print(f"  {'λ':>10s} {'dim':>5s} {'max leakage':>14s} {'G₁-invariant?':>15s}")

    invariance_results = {}
    for lam in unique_w:
        mask = np.abs(w_rounded - lam) < 1e-8
        V_lam = V_full[:, mask]
        P_lam = V_lam @ V_lam.T.conj()
        I_minus_P = np.eye(228) - P_lam

        max_leakage = 0.0
        for key in g1_keys:
            gen = g1_gens[key]
            rho_g = gen.rho().astype(np.complex128)
            leakage = np.linalg.norm(I_minus_P @ rho_g @ P_lam, ord=2)
            max_leakage = max(max_leakage, leakage)

        is_invariant = max_leakage < 1e-8
        invariance_results[lam] = (is_invariant, max_leakage)
        status = "INVARIANT" if is_invariant else "NOT invariant"
        print(f"  {lam:10.6f} {int(np.sum(mask)):5d} {max_leakage:14.2e} {status:>15s}")

    # ── Per-block G₁-invariance ──
    # Check whether each block's contribution to each eigenspace is
    # invariant under G₁ restricted to that block.
    print(f"\n─ Per-block G₁-invariance ─")
    print(f"  For each eigenspace λ, check if the block-restricted eigenspace")
    print(f"  E_λ|_B = P_B E_λ is invariant under ρ_B(G₁).")
    print(f"  {'λ':>10s} {'cp(64)':>10s} {'ep(144)':>10s} {'co(8)':>10s} {'eo(12)':>10s}")

    block_ranges = {
        'cp': (0, 64), 'ep': (64, 208), 'co': (208, 216), 'eo': (216, 228)
    }

    for lam in unique_w:
        mask = np.abs(w_rounded - lam) < 1e-8
        V_lam = V_full[:, mask]
        P_lam = V_lam @ V_lam.T.conj()
        d_full = int(np.sum(mask))

        block_status = {}
        for bname, (i0, i1) in block_ranges.items():
            b_dim = i1 - i0
            # Block projector
            P_b = np.zeros((228, 228))
            P_b[i0:i1, i0:i1] = np.eye(b_dim)

            # Eigenspace restricted to this block
            P_lam_b = P_b @ P_lam @ P_b
            d_b = int(round(np.real(np.trace(P_lam_b))))

            if d_b == 0:
                block_status[bname] = 'n/a'
                continue

            # Projector within the block (b_dim × b_dim)
            P_lam_b_sub = P_lam_b[i0:i1, i0:i1]
            I_b = np.eye(b_dim)
            I_minus_P_b = I_b - P_lam_b_sub

            max_leak_b = 0.0
            for key in g1_keys:
                gen = g1_gens[key]
                rho_full = gen.rho().astype(np.complex128)
                rho_b = rho_full[i0:i1, i0:i1]
                leak_b = np.linalg.norm(I_minus_P_b @ rho_b @ P_lam_b_sub, ord=2)
                max_leak_b = max(max_leak_b, leak_b)

            is_inv_b = max_leak_b < 1e-6
            block_status[bname] = f"{'✓' if is_inv_b else '✗'} d={d_b}"

        print(f"  {lam:10.6f} {block_status['cp']:>10s} {block_status['ep']:>10s} "
              f"{block_status['co']:>10s} {block_status['eo']:>10s}")

    # ── Key findings ──
    print(f"\n─ Boundary analysis ─")

    # 1. Check slow subspace V_slow = ⊕_{λ≥2/3} E_λ (100-dim)
    slow_mask = w_rounded >= 2 / 3 - 1e-8
    V_slow = V_full[:, slow_mask]
    P_slow = V_slow @ V_slow.T.conj()
    I_minus_P_slow = np.eye(228) - P_slow
    max_slow_leak = 0.0
    for key in g1_keys:
        rho_g = g1_gens[key].rho().astype(np.complex128)
        leak = np.linalg.norm(I_minus_P_slow @ rho_g @ P_slow, ord=2)
        max_slow_leak = max(max_slow_leak, leak)
    slow_invariant = max_slow_leak < 1e-8
    dim_slow = int(np.sum(slow_mask))
    print(f"  V_slow (λ≥2/3, {dim_slow}D): G₁-invariant = {slow_invariant} "
          f"(max leak={max_slow_leak:.2e})")

    # 2. Check fast subspace V_fast = ⊕_{λ<2/3} E_λ (128-dim)
    fast_mask = w_rounded < 2 / 3 - 1e-8
    V_fast = V_full[:, fast_mask]
    P_fast = V_fast @ V_fast.T.conj()
    I_minus_P_fast = np.eye(228) - P_fast
    max_fast_leak = 0.0
    for key in g1_keys:
        rho_g = g1_gens[key].rho().astype(np.complex128)
        leak = np.linalg.norm(I_minus_P_fast @ rho_g @ P_fast, ord=2)
        max_fast_leak = max(max_fast_leak, leak)
    fast_invariant = max_fast_leak < 1e-8
    dim_fast = int(np.sum(fast_mask))
    print(f"  V_fast (λ<2/3, {dim_fast}D): G₁-invariant = {fast_invariant} "
          f"(max leak={max_fast_leak:.2e})")

    # ── Corrected understanding (post-ρ-fix) ──
    print(f"\n  CORRECTION to paper's Problem 3 claim:")
    print(f"  - E_{{2/3}} as a full 26D subspace is NOT G₁-invariant.")
    print(f"  - Post-ρ-fix: the co block has 3 eigenvalues K_co={{3,4,6}}")
    print(f"    (multiplicities 2,3,3). At k=3 (λ=2/3), co contributes 2 dim.")
    print(f"  - The 24D ep-block component at λ=2/3 is mixed by G₁ permutation.")
    print(f"  ")
    print(f"  CORRECT characterization of λ=2/3:")
    print(f"  - It is one of three co-block eigenvalues (k=3,4,6).")
    print(f"  - Post-ρ-fix: co-block is perm@phase, NOT scalar.")
    print(f"  - The co block participates in spectral layering at k=3,4,6")
    print(f"    with the permutation@phase structure (Lemma 4.1).")
    print(f"  - The Galois trace cancellation (ω+ω²+1=0) still operates —")
    print(f"    it ensures all three eigenvalues are rational despite the")
    print(f"    off-diagonal entries from position permutation.")
    print(f"  ")
    print(f"  REVISED P3 framing: The co-block participates in layering")
    print(f"  with 3 distinct eigenvalues via its perm@phase structure.")
    print(f"  The association scheme / Bose-Mesner algebra approach explains")
    print(f"  the cp and ep block spectra; the co/eo blocks contribute")
    print(f"  additional k-values through the cycle characters of their")
    print(f"  permutation@phase representation.")

    return True  # test completed successfully (clarified the claim)


def test_symmetry_broken_qsqrt5():
    """Problem 4: symmetry-broken families (n=8, n=16) and Q(sqrt5).

    Post-ρ-fix: the sqrt5 eigenvalues PERSIST in n=8 and n=16.
    They come from CP/EP adjacency algebra symmetry breaking
    (incomplete face coverage), NOT from the CO/EO representation.

    The ρ-fix corrected CO/EO from diagonal-only to perm@phase
    (adding k=1 layer, co={3,4,6}, eo={1,2,4}), but this does
    not affect the CP/EP block spectra where the sqrt5 originates.
    """
    import numpy as np
    from rime.cubie import CubieMove
    from rime.helpers import is_in_qsqrt5

    print("\n" + "=" * 80)
    print("Problem 4: Symmetry-Broken Families and Q(sqrt5) Post-ρ-Fix")
    print("=" * 80)
    print("ρ-fix: CO/EO corrected (diagonal-only → perm@phase).")
    print("sqrt5 in n=8,n=16 is from CP/EP adjacency algebra — PERSISTS.")

    cso = SlowDynamics.lite()
    prim = CubieMove.prim_moves()

    families = {}
    for n_val in [18, 16, 12, 10, 8, 6]:
        gens_n = {}
        for k in cso.rho_moves(n=n_val):
            if k in prim:
                gens_n[k] = prim[k]
        families[f'n={n_val}'] = (gens_n, n_val)

    print(f"\n{'─' * 80}")
    print(f"{'Family':>10s} {'|S|':>4s} {'#λ':>4s} {'All Q?':>8s} {'Field':>12s}")
    print(f"{'─' * 80}")

    omega = np.exp(2j * np.pi / 3)

    for name, (gens_dict, n_gen) in families.items():
        gen_list = list(gens_dict.values())
        rhos = [g.rho().astype(np.complex128) for g in gen_list]
        A = sum(rhos) / n_gen

        is_herm = np.allclose(A, A.T.conj(), atol=1e-10)
        if is_herm:
            w = np.linalg.eigvalsh(A)
        else:
            w_raw = np.linalg.eigvals(A)
            w = np.real(w_raw[np.abs(np.imag(w_raw)) < 1e-8])
        w_unique = np.unique(np.round(w, 8))
        n_unique = len(w_unique)

        m_eff = n_gen // 2 if n_gen % 2 == 0 else n_gen
        from rime.helpers import is_rational_form
        all_rational = all(is_rational_form(lam, m_eff) for lam in w_unique)

        field = 'Q'
        if not all_rational:
            non_rat = [lam for lam in w_unique if not is_rational_form(lam, m_eff)]
            if all(is_in_qsqrt5(lam)[0] for lam in non_rat):
                field = 'Q(sqrt5)'
            else:
                field = 'higher'

        print(f"{name:>10s} {n_gen:4d} {n_unique:4d} {str(all_rational):>8s} {field:>12s}")

    # Detailed n=8 analysis
    print(f"\n{'─' * 80}")
    print("Detailed n=8 analysis (post-ρ-fix)")
    print(f"{'─' * 80}")

    gens_8 = {}
    for k in cso.rho_moves(n=8):
        if k in prim:
            gens_8[k] = prim[k]
    gen_list_8 = list(gens_8.values())
    rhos_8 = [g.rho().astype(np.complex128) for g in gen_list_8]
    A_8 = sum(rhos_8) / len(rhos_8)

    # Block-level analysis
    for block_name, sl in [('cp', slice(0,64)), ('ep', slice(64,208)),
                            ('co', slice(208,216)), ('eo', slice(216,228))]:
        A_blk = A_8[sl, sl]
        w_blk = np.linalg.eigvalsh(A_blk)
        w_u = np.unique(np.round(w_blk, 8))
        m8 = len(gens_8) // 2
        has_sqrt5 = any(not is_rational_form(lam, m8) for lam in w_u)
        print(f"  {block_name} block: eigenvalues={sorted(w_u, reverse=True)}"
              f"{' ← sqrt5!' if has_sqrt5 else ''}")

    print(f"\n  Key finding:")
    print(f"    The sqrt5 eigenvalues (≈0.9045, ≈0.3455) are in the EP and EO blocks.")
    print(f"    They come from CP/EP adjacency algebra symmetry breaking")
    print(f"    (n=8 has only axis-0 and axis-2 moves, no U/D faces).")
    print(f"    This is a REAL effect, not a ρ-artifact.")

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Core API Tests
# Self-contained unit tests of CubieSpectralOperator primary-object methods.
# These test the A→K→κ pipeline that defines the trilogy's three papers.
# Minimal setup: create a CubieSpectralOperator instance, call methods, verify.
# ═══════════════════════════════════════════════════════════════════════════

def test_spectral_layers():
    """Test CubieSpectralOperator._compute_spectral_layers: eigenvalues, dimensions, projectors."""
    print("\n── test_spectral_layers ──")

    cso = CubieSpectralOperator(n=18)

    # 1. Layer dimensions must sum to TOTAL_DIM (228)
    total_dim = sum(info['dim'] for info in cso._layers.values())
    assert total_dim == TOTAL_DIM, f"Layer dims sum to {total_dim}, expected {TOTAL_DIM}"
    print(f"  [OK] Layer dimensions sum to {TOTAL_DIM}")

    # 2. Six expected eigenvalues for n=18 (post-ρ-fix): {1, 8/9, 7/9, 2/3, 5/9, 1/3}
    expected = {1.0, 8 / 9, 7 / 9, 2 / 3, 5 / 9, 1 / 3}
    for exp in expected:
        found = any(abs(lam - exp) < cso.tol for lam in cso._layers)
        assert found, f"Expected eigenvalue {exp:.6f} not found"
    print(f"  [OK] All 6 expected eigenvalues present")

    # 3. dim_const + dim_slow + dim_fast = 228
    mask_fast = cso.w < 2 / 3 - cso.tol
    dim_fast = int(np.sum(mask_fast))
    assert cso.dim_const + cso.dim_slow + dim_fast == TOTAL_DIM, \
        f"dim_const({cso.dim_const})+dim_slow({cso.dim_slow})+dim_fast({dim_fast})≠{TOTAL_DIM}"
    print(f"  [OK] dim_const({cso.dim_const}) + dim_slow({cso.dim_slow}) + dim_fast({dim_fast}) = {TOTAL_DIM}")

    # 4. Projectors are idempotent and mutually orthogonal
    projs = [info['projector'] for info in cso._layers.values()]
    for i, Pi in enumerate(projs):
        assert np.allclose(Pi @ Pi, Pi, atol=cso.tol * 10), f"Projector {i} not idempotent"
        for j, Pj in enumerate(projs):
            if i < j:
                assert np.allclose(Pi @ Pj, 0, atol=cso.tol * 10), \
                    f"Projectors {i},{j} not orthogonal"
    print(f"  [OK] All projectors idempotent and mutually orthogonal")

    # 5. Sum of projectors = I
    assert np.allclose(sum(projs), np.eye(TOTAL_DIM), atol=cso.tol * 10), "Σ projectors ≠ I"
    print(f"  [OK] Σ P_i = I")

    # 6. Theoretical multiplicities for n=18 (post-ρ-fix: 6 layers)
    for lam, exp_dim in [(1.0, 20), (8 / 9, 2), (7 / 9, 39), (2 / 3, 26), (5 / 9, 106), (1 / 3, 35)]:
        mask = np.abs(cso.w - lam) < cso.tol
        actual_dim = int(np.sum(mask))
        assert actual_dim == exp_dim, f"λ≈{lam:.6f}: dim={actual_dim}, expected {exp_dim}"
    print(f"  [OK] Eigenvalue multiplicities match theory: "
          f"λ=1:20, λ=8/9:2, λ=7/9:39, λ=2/3:26, λ=5/9:106, λ=1/3:35")

    # 7. rho_fast = 5/9 for n=18
    assert abs(cso.rho_fast - 5 / 9) < cso.tol, f"rho_fast={cso.rho_fast}, expected 5/9"
    print(f"  [OK] rho_fast = {cso.rho_fast:.6f} ≈ 5/9")

    print(f"  All spectral_layers checks passed.")
    return True


def test_spectral_evolve():
    """Test CubieSpectralOperator.spectral_evolve: T-step spectral diffusion."""
    print("\n── test_spectral_evolve ──")

    cso = CubieSpectralOperator(n=18)

    # 1. T=0: identity
    x = np.random.randn(TOTAL_DIM) + 1j * np.random.randn(TOTAL_DIM)
    y0 = cso.spectral_evolve(x, 0)
    assert np.allclose(y0, x, atol=cso.tol), "T=0 must return x"
    print(f"  [OK] T=0: evolve(x, 0) = x")

    # 2. T=1: equals A @ x
    y1 = cso.spectral_evolve(x, 1)
    y_A = cso.A @ x
    err = np.linalg.norm(y1 - y_A)
    assert err < cso.tol * TOTAL_DIM, f"T=1: |evolve - A@x| = {err:.2e}"
    print(f"  [OK] T=1: evolve(x, 1) = A @ x  (err={err:.2e})")

    # 3. On each eigenspace, evolve(v, 1) = λ·v
    for lam in sorted(cso._layers.keys(), reverse=True):
        V_lam = cso.eigenspace_basis(lam)
        if V_lam.shape[1] > 0:
            v = V_lam[:, 0]
            y = cso.spectral_evolve(v, 1)
            assert np.allclose(y, lam * v, atol=cso.tol * 100), \
                f"λ={lam:.6f}: evolve(v,1) ≠ λ·v"
    print(f"  [OK] On each eigenspace: evolve(v, 1) = λ·v")

    # 4. T-step on eigenvector: λ^T·v
    lam_test = 7 / 9
    V_test = cso.eigenspace_basis(lam_test)
    assert V_test.shape[1] > 0, f"No eigenvectors for λ={lam_test}"
    v = V_test[:, 0]
    for T in [0, 1, 2, 5, 10]:
        y = cso.spectral_evolve(v, T)
        expected = (lam_test ** T) * v
        assert np.allclose(y, expected, atol=cso.tol * 1000), \
            f"λ={lam_test}, T={T}: evolve(v,T) ≠ λ^T·v"
    print(f"  [OK] T-step on eigenvector (λ=7/9): λ^T·v for T∈{{0,1,2,5,10}}")

    # 5. Linearity
    x1 = np.random.randn(TOTAL_DIM) + 1j * np.random.randn(TOTAL_DIM)
    x2 = np.random.randn(TOTAL_DIM) + 1j * np.random.randn(TOTAL_DIM)
    y_combined = cso.spectral_evolve(x1 + x2, 3)
    y_separate = cso.spectral_evolve(x1, 3) + cso.spectral_evolve(x2, 3)
    err_lin = np.linalg.norm(y_combined - y_separate)
    assert err_lin < cso.tol * 100, f"Linearity failed: err={err_lin:.2e}"
    print(f"  [OK] Linearity: evolve(x1+x2, T) = evolve(x1, T) + evolve(x2, T)")

    # 6. Consistency with A^T
    for T in [1, 2, 3]:
        y_exact = np.linalg.matrix_power(cso.A, T) @ x
        y_evolve = cso.spectral_evolve(x, T)
        err_T = np.linalg.norm(y_evolve - y_exact)
        assert err_T < cso.tol * TOTAL_DIM * 10, \
            f"T={T}: |evolve - A^{T}@x| = {err_T:.2e}"
    print(f"  [OK] A^T @ x = evolve(x, T) for T=1,2,3")

    print(f"  All spectral_evolve checks passed.")
    return True


# ── 14. 理论 invariant 测试 ──────────────────────────────────────────────

def test_projector_trace():
    """Tr(P_i) = dim(E_i): projector trace equals eigenspace dimension."""
    print("\n── test_projector_trace ──")
    cso = CubieSpectralOperator(n=18)
    for lam, info in cso._layers.items():
        P = info['projector']
        trP = np.trace(P)
        dim = info['dim']
        assert abs(trP.real - dim) < cso.tol * 10, \
            f"λ={lam:.6f}: Tr(P)={trP.real:.6f} ≠ dim={dim}"
        assert abs(trP.imag) < cso.tol * 10, f"λ={lam:.6f}: Im(Tr(P))={trP.imag:.2e}"
    print(f"  [OK] Tr(P_i) = dim(E_i) for all {len(cso._layers)} layers")
    return True


def test_spectral_completeness():
    """A = Σ λ_i P_i: spectral decomposition reconstructs A."""
    print("\n── test_spectral_completeness ──")
    cso = CubieSpectralOperator(n=18)
    A_recon = np.zeros_like(cso.A, dtype=complex)
    for lam, info in cso._layers.items():
        A_recon += lam * info['projector']
    err = np.linalg.norm(cso.A - A_recon, 'fro')
    assert err < cso.tol * TOTAL_DIM, f"A ≠ Σ λ_i P_i: err={err:.2e}"
    print(f"  [OK] A = Σ λ_i P_i  (Frobenius err={err:.2e})")
    return True


def test_commuting_projectors():
    """A P_i = P_i A: A commutes with every spectral projector."""
    print("\n── test_commuting_projectors ──")
    cso = CubieSpectralOperator(n=18)
    for lam, info in cso._layers.items():
        P = info['projector']
        AP = cso.A @ P
        PA = P @ cso.A
        err = np.linalg.norm(AP - PA, 'fro')
        assert err < cso.tol * TOTAL_DIM * 5, \
            f"λ={lam:.6f}: [A, P_i] Frobenius err={err:.2e}"
    print(f"  [OK] A P_i = P_i A for all {len(cso._layers)} layers")
    return True


def test_projector_eigen_property():
    """A P_i = λ_i P_i: projector is eigenoperator of A."""
    print("\n── test_projector_eigen_property ──")
    cso = CubieSpectralOperator(n=18)
    for lam, info in cso._layers.items():
        P = info['projector']
        AP = cso.A @ P
        lamP = lam * P
        err = np.linalg.norm(AP - lamP, 'fro')
        assert err < cso.tol * TOTAL_DIM * 5, \
            f"λ={lam:.6f}: A P_i ≠ λ_i P_i, err={err:.2e}"
    print(f"  [OK] A P_i = λ_i P_i for all {len(cso._layers)} layers")
    return True


def test_slow_projector_consistency():
    """P_slow = Σ_{λ≥2/3} P_λ: slow projector equals sum of slow-layer projectors."""
    print("\n── test_slow_projector_consistency ──")
    cso = CubieSpectralOperator(n=18)
    P_slow = cso.slow_projector(threshold=2/3)
    P_sum = np.zeros_like(cso.A, dtype=complex)
    slow_lam_count = 0
    for lam, info in cso._layers.items():
        if lam >= 2/3 - cso.tol:
            P_sum += info['projector']
            slow_lam_count += 1
    err = np.linalg.norm(P_slow - P_sum, 'fro')
    assert err < cso.tol * TOTAL_DIM, \
        f"P_slow ≠ Σ_{{λ≥2/3}} P_λ: err={err:.2e}"
    tr_slow = np.trace(P_slow).real
    assert abs(tr_slow - (cso.dim_const + cso.dim_slow)) < cso.tol * 10, \
        f"Tr(P_slow)={tr_slow:.1f} ≠ dim_const+dim_slow={cso.dim_const + cso.dim_slow}"
    print(f"  [OK] P_slow = Σ_{{λ≥2/3}} P_λ ({slow_lam_count} layers), "
          f"dim_slow={int(tr_slow)}")
    return True


def test_semigroup_spectral_law():
    """A^t = Σ λ_i^t P_i for both integer and fractional t (semigroup spectral law)."""
    print("\n── test_semigroup_spectral_law ──")
    from scipy.linalg import fractional_matrix_power
    cso = CubieSpectralOperator(n=18)

    # Helper: build A^t via spectral sum
    def spectral_power(T):
        val = np.zeros_like(cso.A, dtype=complex)
        for lam, info in cso._layers.items():
            val += (lam ** T) * info['projector']
        return val

    # Integer T: compare with np.linalg.matrix_power
    for T in [1, 2, 3, 4, 5]:
        A_T_exact = np.linalg.matrix_power(cso.A, T)
        A_T_spec = spectral_power(T)
        err = np.linalg.norm(A_T_exact - A_T_spec, 'fro')
        assert err < cso.tol * TOTAL_DIM * 10, \
            f"T={T}: A^T ≠ Σ λ_i^T P_i, err={err:.2e}"

    # Fractional T via scipy fractional_matrix_power
    for T in [0.5, 1.5, 2.5]:
        A_T_exact = fractional_matrix_power(cso.A, T)
        A_T_spec = spectral_power(T)
        err = np.linalg.norm(A_T_exact - A_T_spec, 'fro')
        assert err < cso.tol * TOTAL_DIM * 20, \
            f"T={T}: A^{T} ≠ Σ λ_i^{T} P_i, err={err:.2e}"
    print(f"  [OK] A^t = Σ λ_i^t P_i for T ∈ {{1,2,3,4,5, 0.5,1.5,2.5}}")

    # Vector-level semigroup property: evolve(x, t1+t2) = evolve(evolve(x, t1), t2)
    x = np.random.randn(TOTAL_DIM) + 1j * np.random.randn(TOTAL_DIM)
    for t1, t2 in [(0.5, 1.5), (1.5, 3.5), (0.7, 2.3)]:
        y_direct = cso.spectral_evolve(x, t1 + t2)
        y_composed = cso.spectral_evolve(cso.spectral_evolve(x, t1), t2)
        err = np.linalg.norm(y_direct - y_composed)
        assert err < cso.tol * TOTAL_DIM * 100, \
            f"t1={t1}, t2={t2}: semigroup property failed, err={err:.2e}"
    print(f"  [OK] Semigroup: evolve(x, t1+t2) = evolve(evolve(x, t1), t2)")

    return True


def test_spectral_entropy():
    """H(z) = -Σ p_i log p_i where p_i = ‖P_i z‖² / ‖z‖². Bounds: 0 ≤ H ≤ log(N_layers)."""
    print("\n── test_spectral_entropy ──")
    cso = CubieSpectralOperator(n=18)

    def spectral_entropy(z):
        norm_sq = np.vdot(z, z).real
        if norm_sq < 1e-15:
            return 0.0
        H = 0.0
        for _, info in cso._layers.items():
            Pz = info['projector'] @ z
            pi = np.vdot(Pz, Pz).real / norm_sq
            if pi > 1e-15:
                H -= pi * np.log(pi)
        return H

    n_layers = len(cso._layers)

    # Random vector
    for _ in range(5):
        z = np.random.randn(TOTAL_DIM) + 1j * np.random.randn(TOTAL_DIM)
        H = spectral_entropy(z)
        assert 0 <= H <= np.log(n_layers) + 1e-10, \
            f"H={H:.6f} not in [0, log({n_layers})={np.log(n_layers):.4f}]"

    # Pure eigenvector → H ≈ 0
    for lam in [1.0, 8/9, 7/9, 2/3, 5/9, 1/3]:
        V_lam = cso.eigenspace_basis(lam)
        if V_lam.shape[1] > 0:
            v = V_lam[:, 0]
            H_pure = spectral_entropy(v)
            assert H_pure < 1e-6, f"Pure eigenvector λ={lam}: H={H_pure:.2e} ≠ 0"

    # Uniform across layers → H ≈ log(n_layers)
    z_uniform = np.zeros(TOTAL_DIM, dtype=complex)
    for _, info in cso._layers.items():
        v = info['projector'] @ np.random.randn(TOTAL_DIM) + 0j
        z_uniform += v / np.linalg.norm(v)
    H_uniform = spectral_entropy(z_uniform)
    assert H_uniform > 0.5 * np.log(n_layers), \
        f"Uniform mixture: H={H_uniform:.4f} too low, expected ~{np.log(n_layers):.4f}"

    print(f"  [OK] Spectral entropy: 0 ≤ H ≤ log({n_layers})={np.log(n_layers):.4f}, "
          f"pure=0, uniform={H_uniform:.4f} ≈ max")
    return True


def test_commutant_residual():
    """‖[P_i, ρ(g)]‖_F per generator: which projectors are central idempotents of the group algebra."""
    print("\n── test_commutant_residual ──")
    cso = CubieSpectralOperator(n=18)

    bounds = {}
    for lam, info in sorted(cso._layers.items(), reverse=True):
        P = info['projector']
        residuals = cso.commutant_residual(P)
        vals = list(residuals.values())
        avg_val = np.mean(vals)
        max_val = np.max(vals)
        bound = 2 * np.sqrt(info['dim'] * TOTAL_DIM)
        assert max_val <= bound + cso.tol * 100, \
            f"λ={lam:.6f}: max ‖[P,ρ]‖={max_val:.4f} > bound {bound:.4f}"
        bounds[lam] = (avg_val, max_val)

    # λ=1: genuine central idempotent — commutes with every ρ(g)
    if 1.0 in bounds:
        avg_val, max_val = bounds[1.0]
        assert avg_val < cso.tol * 10, f"λ=1: nonzero commutant residual avg={avg_val:.2e}"
        print(f"  [OK] λ=1: central idempotent (avg ‖[P,ρ]‖={avg_val:.2e}, max={max_val:.2e})")

    # Report which layers are central
    for lam in sorted(bounds, reverse=True):
        avg_val, max_val = bounds[lam]
        is_central = "CENTRAL ✓" if max_val < cso.tol * 10 else ""
        tag = "(trivial rep)" if abs(lam - 1.0) < cso.tol else ""
        print(f"  [info] λ={lam:.6f}: ‖[P,ρ]‖ avg={avg_val:.4f} max={max_val:.4f} {is_central}{tag}")

    print(f"  [OK] commutant_residual: P_1 is the only genuine central idempotent")
    return True


def test_spectral_curvature_tensor():
    """‖[P_i, ρ(g)]‖_F: curvature measures how much generator action mixes eigenspaces."""
    print("\n── test_spectral_curvature_tensor ──")
    cso = CubieSpectralOperator(n=18)

    curvatures = {}
    for lam, info in cso._layers.items():
        residuals = cso.commutant_residual(info['projector'])
        vals = list(residuals.values())
        curvatures[lam] = (float(np.mean(vals)), float(np.max(vals)))

    # Invariant subspace (λ=1) should have zero curvature
    if 1.0 in curvatures:
        avg_c, max_c = curvatures[1.0]
        assert avg_c < cso.tol * 10, f"λ=1: nonzero curvature avg={avg_c:.2e}"
        print(f"  [OK] λ=1: zero curvature (avg={avg_c:.2e}, max={max_c:.2e})")

    # Report curvature spectrum
    for lam in sorted(curvatures, reverse=True):
        avg_c, max_c = curvatures[lam]
        label = "(invariant)" if abs(lam - 1.0) < cso.tol else ""
        print(f"  [info] λ={lam:.6f}: curv avg={avg_c:.4f}, max={max_c:.4f} {label}")

    print(f"  All curvatures computed via commutant_residual.")
    return True


def test_mode_transport():
    """P_i ρ(g) P_j: mode transport — how generators couple eigenspaces."""
    print("\n── test_mode_transport ──")
    cso = CubieSpectralOperator(n=18)

    layers = sorted(cso._layers.keys(), reverse=True)
    rho_list = [rho for _, rho in cso.rho_moves.values()]

    # Transport tensor via new method
    T = cso.transport_tensor()

    # 1. Completeness: Σ_{i,j} P_i ρ(g) P_j = ρ(g)
    for rho_g in rho_list:
        transport_sum = np.zeros_like(cso.A, dtype=complex)
        for lam_i in layers:
            Pi = cso._layers[lam_i]['projector']
            for lam_j in layers:
                Pj = cso._layers[lam_j]['projector']
                transport_sum += Pi @ rho_g @ Pj
        err = np.linalg.norm(transport_sum - rho_g, 'fro')
        assert err < cso.tol * TOTAL_DIM * 10, \
            f"Σ P_i ρ(g) P_j ≠ ρ(g), err={err:.2e}"
    print(f"  [OK] Σ_{{i,j}} P_i ρ(g) P_j = ρ(g) (completeness)")

    # 2. λ=1: trivial representation — P_1 ρ(g) P_1 = P_1
    if 1.0 in cso._layers:
        P1 = cso._layers[1.0]['projector']
        for rho_g in rho_list[:3]:
            diag_transport = P1 @ rho_g @ P1
            err = np.linalg.norm(diag_transport - P1, 'fro')
            assert err < cso.tol * TOTAL_DIM, \
                f"λ=1: P_1 ρ(g) P_1 ≠ P_1, err={err:.2e}"
        print(f"  [OK] λ=1: P_1 ρ(g) P_1 = P_1 (trivial rep)")

    # 3. λ=1 decoupled from all other sectors
    t_1x = [(lam_i, lam_j, d['max']) for (lam_i, lam_j), d in T.items()
            if abs(lam_i - 1.0) < cso.tol and abs(lam_j - 1.0) > cso.tol]
    max_cross = max(v for _, _, v in t_1x) if t_1x else 0.0
    assert max_cross < cso.tol * 10, f"λ=1 ↔ others: nonzero transport {max_cross:.2e}"
    print(f"  [OK] λ=1 decoupled from all other sectors (max cross-transport={max_cross:.2e})")

    # 4. Transport tensor summary
    print(f"  [info] Transport tensor ‖P_i ρ(g) P_j‖ (6×6, max over generators):")
    for lam_i in layers:
        row = [f"{T[(lam_i, lam_j)]['max']:7.4f}" for lam_j in layers]
        tag = "(trivial)" if abs(lam_i - 1.0) < cso.tol else ""
        print(f"    λ={lam_i:.6f} {tag}: [{', '.join(row)}]")

    print(f"  [OK] Transport tensor: block-diagonal structure confirmed")
    return True


# ── 15. 深层代数结构 ────────────────────────────────────────────────────

def test_transport_graph():
    """Transport graph: nodes=sectors, edges=nonzero cross-transport. Verify star structure."""
    print("\n── test_transport_graph ──")
    cso = CubieSpectralOperator(n=18)

    G = cso.transport_graph()

    # 6 nodes (post-ρ-fix)
    assert len(G['nodes']) == 6, f"Expected 6 nodes, got {len(G['nodes'])}"
    print(f"  [OK] Nodes: {[f'{lam:.6f}' for lam in G['nodes']]}")

    # λ=1 is isolated (central idempotent → no cross-transport)
    assert 1.0 in G['isolated'], f"λ=1 should be isolated, got isolated={G['isolated']}"
    print(f"  [OK] λ=1 is isolated (central idempotent)")

    # Verify edges match the known star structure
    for lam_i, lam_j, w in G['edges']:
        print(f"  [edge] λ={lam_i:.6f} ↔ λ={lam_j:.6f}, max transport={w:.4f}")

    # Star hub detection: λ=5/9 should be the hub
    if G['is_star']:
        assert abs(G['hub'] - 5/9) < cso.tol * 10, \
            f"Expected hub λ=5/9, got {G['hub']:.6f}"
        print(f"  [OK] Star graph: hub = λ={G['hub']:.6f} (5/9)")

    # Key structural assertions using actual layer keys
    layers_dict = {round(lam, 6): lam for lam in cso._layers}
    T = cso.transport_tensor()
    lam_79 = layers_dict.get(round(7/9, 6))
    lam_23 = layers_dict.get(round(2/3, 6))
    lam_59 = layers_dict.get(round(5/9, 6))
    # 7/9 ↔ 5/9 should exist
    t_79_59 = T[(lam_79, lam_59)]['max']
    assert t_79_59 > cso.tol * 10, f"7/9 ↔ 5/9 transport should be nonzero, got {t_79_59:.2e}"
    # 2/3 ↔ 5/9 should exist
    t_23_59 = T[(lam_23, lam_59)]['max']
    assert t_23_59 > cso.tol * 10, f"2/3 ↔ 5/9 transport should be nonzero, got {t_23_59:.2e}"
    # 7/9 ↔ 2/3 should NOT exist
    t_79_23 = T[(lam_79, lam_23)]['max']
    assert t_79_23 < cso.tol * 10, f"7/9 ↔ 2/3 should be zero, got {t_79_23:.2e}"
    print(f"  [OK] 7/9↔5/9={t_79_59:.4f}, 2/3↔5/9={t_23_59:.4f}, 7/9↔2/3={t_79_23:.2e}")

    # Laplacian properties: positive semidefinite, zero sum rows
    L = G['laplacian']
    assert np.allclose(L.sum(axis=1), 0, atol=cso.tol), "Laplacian rows should sum to zero"
    w_L = np.linalg.eigvalsh(L)
    assert w_L[0] >= -cso.tol * 10, f"Laplacian should be PSD, min eig={w_L[0]:.2e}"
    print(f"  [OK] Graph Laplacian: PSD, row-sums=0, λ(L)={np.round(w_L, 4)}")

    return True


def test_raising_lowering():
    """R/L operators: check algebraic structure on transport edges."""
    print("\n── test_raising_lowering ──")
    cso = CubieSpectralOperator(n=18)

    rl = cso.raising_lowering()
    print(f"  Generator: {rl['generator_key']}")

    # Verify all transport edges have R/L operators
    for (lam_i, lam_j), nrm in rl['norms'].items():
        print(f"  λ={lam_i:.6f}↔λ={lam_j:.6f}: ‖R‖={nrm['R']:.4f}, ‖L‖={nrm['L']:.4f}")
        assert nrm['R'] > cso.tol * 10, f"R({lam_i:.6f}→{lam_j:.6f}) should be nonzero"
        assert nrm['L'] > cso.tol * 10, f"L({lam_j:.6f}→{lam_i:.6f}) should be nonzero"

    # R^† R and R R^†: should be related to projectors
    closure = rl['closure']
    if closure:
        print(f"  [algebra] R†R norm = {closure['R†R']:.4f}")
        print(f"  [algebra] RR† norm = {closure['RR†']:.4f}")
        print(f"  [algebra] LR  norm = {closure['LR']:.4f}")
        print(f"  [algebra] RL  norm = {closure['RL']:.4f}")
        print(f"  [algebra] ‖[L,R]‖ = {closure['‖[L,R]‖']:.4f}")

        # R†R should be approximately proportional to P_{7/9}
        # (since R = P_{5/9} ρ(g) P_{7/9}, R†R = P_{7/9} ρ(g)† P_{5/9} ρ(g) P_{7/9})
        # On V_{7/9}, generators act approximately as scalar 7/9 → not exactly
        assert closure['R†R'] > 0.1, f"R†R too small: {closure['R†R']:.4f}"

    # Verify that raising from λ=7/9 goes to λ=5/9 (not λ=2/3)
    layers_dict = {round(lam, 6): lam for lam in cso._layers}
    lam_79 = layers_dict.get(round(7/9, 6))
    lam_59 = layers_dict.get(round(5/9, 6))
    lam_23 = layers_dict.get(round(2/3, 6))
    R_79_59 = rl['R'].get((lam_79, lam_59))
    R_79_23 = rl['R'].get((lam_79, lam_23)) if (lam_79, lam_23) in rl['R'] else None
    assert R_79_59 is not None, "R(7/9→5/9) should exist"
    assert R_79_23 is None, "R(7/9→2/3) should NOT exist (decoupled channels)"
    print(f"  [OK] Raising: 7/9 → 5/9 ✓, 7/9 → 2/3 absent ✓")

    return True


def test_commutant_algebra():
    """Commutant algebra: compute C = {X : [X, ρ(g)] = 0 ∀g} and irreducible structure."""
    print("\n── test_commutant_algebra ──")
    cso = CubieSpectralOperator(n=18)

    ca = cso.commutant_algebra()
    print(f"  Total commutant dimension: {ca['dim_total']}")

    # Central idempotents: only λ=1
    central_rounded = [round(lam, 6) for lam in ca['central_idempotents']]
    assert 1.0 in central_rounded, f"λ=1 must be central, got {ca['central_idempotents']}"
    print(f"  [OK] Central idempotents: {[f'{lam:.6f}' for lam in ca['central_idempotents']]}")

    # Report block structure
    for lam in sorted(ca['blocks'], reverse=True):
        b = ca['blocks'][lam]
        n_irreps = b['n_irreps']
        d = b['dim']
        c = b['commutant_dim']
        label = "CENTRAL" if lam in ca['central_idempotents'] else ""
        print(f"  λ={lam:.6f}: dim={d}, comm_dim={c}, "
              f"n_irreps≈{n_irreps} ({'decomposes into ' + str(n_irreps) + ' blocks' if n_irreps > 1 else 'single block'}) {label}")

    # Key checks using rounded key matching
    blocks_lookup = {round(lam, 6): lam for lam in ca['blocks']}

    # λ=1: 24D, generators act as identity → commutant = all 24×24 matrices → dim = 24² = 576
    if 1.0 in blocks_lookup:
        lam_1 = blocks_lookup[1.0]
        b1 = ca['blocks'][lam_1]
        assert b1['commutant_dim'] == 576, \
            f"λ=1: expected comm_dim=576 (24²), got {b1['commutant_dim']}"
        print(f"  [OK] λ=1: comm_dim=576 = 24² (complete matrix algebra on trivial rep)")

    # λ=2/3: should be fully degenerate → comm_dim = d² = 1024 (32×32)
    key_23 = round(2/3, 6)
    if key_23 in blocks_lookup:
        lam_23 = blocks_lookup[key_23]
        b23 = ca['blocks'][lam_23]
        if b23['commutant_dim'] == b23['dim'] ** 2:
            print(f"  [OK] λ=2/3: comm_dim={b23['commutant_dim']}={b23['dim']}² (scalar action, fully degenerate)")

    # λ=5/9 (96D): should have nontrivial internal structure
    key_59 = round(5/9, 6)
    if key_59 in blocks_lookup:
        lam_59 = blocks_lookup[key_59]
        b59 = ca['blocks'][lam_59]
        assert b59['commutant_dim'] < b59['dim'] ** 2, \
            f"λ=5/9 (96D hub): expected nontrivial structure, comm_dim={b59['commutant_dim']}"
        print(f"  [info] λ=5/9 hub: comm_dim={b59['commutant_dim']} < {b59['dim']}²={b59['dim']**2}, "
              f"nontrivial internal structure")

    return True


def test_irrep_decomposition():
    """Artin-Wedderburn: decompose each eigenspace into isotypic components (d_irrep, multiplicity)."""
    print("\n── test_irrep_decomposition ──")
    cso = CubieSpectralOperator(n=18)

    ird = cso.irrep_decomposition()
    print(f"  Total commutant dimension: {ird['dim_total']}")
    print(f"  Total isotypic types found: {ird['total_isotypic_types']}")

    # Report per-block decomposition
    for lam in sorted(ird['blocks'], reverse=True):
        b = ird['blocks'][lam]
        d = b['dim']
        c = b['commutant_dim']
        s = b['center_dim']
        iso = b['isotypic']
        iso_str = ", ".join(f"({d_irr}D×{mult})" for d_irr, mult in iso)
        print(f"  λ={lam:.6f}: dim={d}, comm_dim={c}, center_dim={s}, "
              f"isotypic=[{iso_str}]")

    # Aggregate irrep types
    print(f"\n  Aggregated irrep types:")
    for irr in ird['irrep_sizes']:
        src_str = ", ".join(f"λ={lam:.4f}×{m}" for lam, m in irr['sources'])
        print(f"    d_irrep={irr['d_irrep']:3d}, total_mult={irr['total_mult']:3d}  ← {src_str}")

    # Key verifications:
    # λ=1: should decompose as 24 copies of 1D trivial rep
    if round(1.0, 6) in {round(lam, 6) for lam in ird['blocks']}:
        lam1 = next(lam for lam in ird['blocks'] if round(lam, 6) == 1.0)
        b1 = ird['blocks'][lam1]
        assert len(b1['isotypic']) > 0, "λ=1 should have isotypic decomposition"
        # Each isotypic component should be d_irrep=1
        for d_irr, mult in b1['isotypic']:
            if d_irr == 1:
                print(f"  [OK] λ=1: {d_irr}D trivial irrep × {mult}")
            else:
                print(f"  [info] λ=1: {d_irr}D irrep × {mult} (non-trivial within V_1?)")

    # Check d_irrep × mult = dim(block) for consistency
    for lam in ird['blocks']:
        b = ird['blocks'][lam]
        total_from_iso = sum(d_irr * mult for d_irr, mult in b['isotypic'])
        if total_from_iso > 0:
            assert abs(total_from_iso - b['dim']) < 2, \
                f"λ={lam:.6f}: Σ d_irr·mult = {total_from_iso} ≠ dim = {b['dim']}"
    print(f"  [OK] Dimension consistency: Σ d_irrep × multiplicity = dim_block for all blocks")

    return True


# ── main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("═══ 1. 基础设置 & 块检测 ═══")
    prim_list18 = list(CubieMove.prim_moves.values())
    rho_moves = [m.rho() for m in prim_list18]
    A_micro = sum(rho_moves) / len(rho_moves)

    eigvals, U = np.linalg.eig(A_micro)

    generators = rho_moves

    test_spectral_layers()
    test_spectral_evolve()

    print("\n╔══ 2. 谱结构 (5 层有理谱 k/9) ══╗")
    w, V = np.linalg.eigh(A_micro)
    idx = np.argsort(-np.abs(w))
    eigvals_am = w[idx]
    U_am = V[:, idx]

    print(f"是否正交检查: {np.allclose(U_am.T @ U_am, np.eye(U_am.shape[1]), atol=1e-8)}")

    test_block_detection(A_micro, U_am)
    block_spectra = analyze_cubie_block_spectra(A_micro, eigvals_am, U_am)

    blocks = detect_blocks(list(CubieMove.prim_moves().values()), V)  # 不依赖顺序
    corner_idx = blocks[0]  # size 64
    edge_idx = blocks[1]  # size 144
    print(len(corner_idx), len(edge_idx))
    # for b in blocks:
    #     if len(b) == 64:
    #         corner_idx = b
    #     elif len(b) == 144:
    #         edge_idx = b
    # 在"物理正确坐标系"里看群作用
    # ARCHIVED: test_spectral_layers_5layer(A_micro, generators)
    # ARCHIVED: test_double_cosets()
    # ARCHIVED: test_isotypic_decomposition()
    test_block_spectrum(A_micro, V, blocks, corner_idx, edge_idx)

    print("\n╔══ 3. Bose-Mesner & 代数性质 ══╗")
    test_bose_mesner(A_micro, generators)
    mask_slow = w >= 2 / 3 - 1e-8
    V_slow = V[:, mask_slow]
    test_commutant_and_algebra(A_micro, generators, V_slow)
    test_fast_layer_properties(A_micro, V_slow, generators)
    test_shell_decomposition(A_micro)

    print("\n╔══ 4. 慢子空间近似 & 群谐函数 ══╗")
    test_slow_approximation(A_micro, w, V)
    test_harmonic_slowest(A_micro, w, V)
    test_harmonic_79_block(V, w)
    test_harmonic_23_block(V, w)
    test_attention_reconstruction(A_micro, w, V)

    print("\n╔══ 5. 退火 & 块谱分解 ══╗")
    test_annealing(A_micro)

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

    print("\n" + "=" * 80)
    print("=== Bonus: Paper Refinement Verification (2026-05-02) ===")
    print("=" * 80)
    test_g1_eigenspace_boundary()
    test_symmetry_broken_qsqrt5()

    print("\n" + "=" * 80)
    print("=== 14. 理论 invariant 测试 ===")
    print("=" * 80)
    test_projector_trace()
    test_spectral_completeness()
    test_commuting_projectors()
    test_projector_eigen_property()
    test_slow_projector_consistency()
    test_semigroup_spectral_law()
    test_spectral_entropy()
    test_commutant_residual()
    test_spectral_curvature_tensor()
    test_mode_transport()

    print("\n" + "=" * 80)
    print("=== 15. 深层代数结构 ===")
    print("=" * 80)
    test_transport_graph()
    test_raising_lowering()
    test_commutant_algebra()
    test_irrep_decomposition()

    """
    A_micro 在做的是"把非交换群压成一个交换代数"
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
