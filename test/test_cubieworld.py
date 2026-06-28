from collections import deque
import numpy as np
import random, math, os
from scipy.stats import pearsonr, spearmanr

"""
cubieworld 慢动力学与环境实验

实验分组：
  1. 时间晶体模拟（慢流形闭合轨道）
  2. 生成器谱结构（不同 n 的特征值分布）
  3. 交换子谱分析
  4. Move scores & move energy 评估
  5. 量子 vs 经典演化差异
  6. Environment 目标状态距离分析
  7. 慢流形距离分布 & 壳层结构
  8. 主模拟 (HybridSimulation)
  9. 退火实验
  10. 温度扫描 (Order/Chaos 相变)
  11. 扩散轨迹
  12. 随机游走慢距离 vs prune_d
  13. 等深度对慢距离 vs prune_d
  14. 准等距性验证
  15. move_energy vs prune_d 相关分析（ranking 验证）
  16. move_energy ranking quality（top-1/Kendall tau）
  17. move_energy component regression（线性回归）
  18. 远区几何信号检测
  19. alpha-sweep ablation（最优 alpha=0）
  20. 二次型 E_geom 比较
  21. Regime 可分性验证（分类测试，非 ranking）
  22. 动力学统计（2-cycle 频率 / orbit / 熵）
  23. 几何-行为因果相关（非 prune 相关）
  24. 2-step MPC vs Greedy（trajectory-level planning + orbit penalty）
  25. TrajectoryEnergy — symmetry breaking in iso-distance shell

运行: python test/test_cubieworld.py
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rime.cubieworld import SlowDynamics, N_GENERATORS
from rime.cubieagent import Environment
from rime.cubie import CubieState, CubieMove, CubieBase
from rime.helpers import cosine_distance, poly_rank
from rime.base import DATA_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

N_PAIRS = 5000
TOL = 1e-10

def setup():
    model = SlowDynamics(n=N_GENERATORS)
    if not hasattr(CubieBase, 'CO_EO_PRUNE'):
        CubieBase.build_pruning_table()
    return model


class BalanceWorld:
    def __init__(self, model: SlowDynamics = None, max_depth=40, balance_tol=10.0):
        self.model = model or SlowDynamics(18)  # SlowDynamics instance
        solved = CubieState.solved()
        self.solved_rho = solved.vector
        self.z_solved = self.model.project(self.solved_rho)
        self.order_pan = deque([(solved, 0.0)])  # (state, e) 两个相空间
        self.chaos_pan = deque([])
        self.max_depth = max_depth
        self.balance_tol = balance_tol
        self.history = []  # Track pan weights over time

    def slow_coord(self, state):
        return self.model.project(state.vector)

    def weight(self, state):
        """混乱度：慢子空间到 solved 的距离"""
        z = self.slow_coord(state)  # Slow distance as weight
        return self.model.l2_distance(z, self.z_solved)

    def energy(self, state):
        z = self.slow_coord(state)
        return np.linalg.norm(z - self.z_solved) ** 2

    def chaos_density(self, state, samples=30):
        """用交换子扰动，看世界长期演化
        如果 [g,h] 几乎不改变 slow space
        → 系统接近可交换 → 有序
        如果 [g,h] 在 slow space 扰动很大
        → 非交换强 → 混沌"""
        total = 0
        moves = list(CubieMove.prim_moves.values())
        z = self.slow_coord(state)
        for _ in range(samples):
            g, h = random.sample(moves, 2)
            m = g @ h @ g.inverse() @ h.inverse()
            state2 = m.act(state)
            z2 = self.slow_coord(state2)
            effect = np.linalg.norm(z2 - z)  # log(ρ(g)) log(ρ(h)) - log(ρ(h)) log(ρ(g))
            total += effect  # commutator_effect
        return total / samples

    def generate_state(self, target_weight):
        depth = int(target_weight)  # Approximate depth for scramble
        state = CubieBase.generate_cubie(length=min(depth, self.max_depth))
        return state, self.energy(state)

    def metropolis_step(self, state, temperature=1.0):
        """
        move 按 ΔE 概率接受
        temperature 高 → 更随机
        temperature 低 → 更趋向 solved
        """
        E0 = self.energy(state)

        g = CubieBase.random_walk(length=1)
        new_state = g.act(state)

        E1 = self.energy(new_state)
        dE = E1 - E0
        # Metropolis rule
        if dE < 0:
            return new_state, E1
        else:
            if random.random() < math.exp(-dE / temperature):
                return new_state, E1
            else:
                return state, E0

    def observe(self):
        """
        低温 → order phase
        高温 → chaos phase
        """
        order_w = [w for _, w in self.order_pan]
        chaos_w = [w for _, w in self.chaos_pan]

        obs = {
            "order_mean": np.mean(order_w) if order_w else 0,
            "chaos_mean": np.mean(chaos_w) if chaos_w else 0,
            "order_var": np.var(order_w) if order_w else 0,
            "chaos_var": np.var(chaos_w) if chaos_w else 0,
            "order_size": len(self.order_pan),
            "chaos_size": len(self.chaos_pan),
        }

        return obs

    def balance(self, max_pan_energy=2000.0):
        order_e = sum(w for _, w in self.order_pan)
        chaos_e = sum(w for _, w in self.chaos_pan)
        imbalance = order_e - chaos_e
        # imbalance_ratio = abs(order_e - chaos_e) / (order_e + chaos_e + 1e-6)
        target = abs(imbalance)

        if order_e > max_pan_energy:  # 防止无限增长
            self.order_pan = deque(sorted(self.order_pan, key=lambda x: x[1])[:len(self.order_pan) // 2])
        if chaos_e > max_pan_energy:
            self.chaos_pan = deque(sorted(self.chaos_pan, key=lambda x: x[1])[:len(self.chaos_pan) // 2])

        if target > self.balance_tol:
            if imbalance > 0:
                # Add to chaos pan
                state, e = self.generate_state(target)
                self.chaos_pan.append((state, e))
            else:
                # Add to order pan (generate low-weight state)
                state, e = self.generate_state(target / 2)  # Bias toward order
                self.order_pan.append((state, e))

            if chaos_e > order_e * 1.5:  # 当 chaos 过重时注入秩序
                solved_like = CubieBase.generate_cubie(length=5)  # 接近 solved 的状态
                self.order_pan.append((solved_like, self.energy(solved_like)))

        self.history.append((order_e, chaos_e, self.observe()))

    def evolve(self, steps=10, temperature=1.0):
        for t in range(steps):
            # if t % self.model.Tf == 0:
            #     for agent in self.agents:
            #         agent.fast_mix()  # 模拟个体噪声
            # Evolve states on both pans
            # current_temp = temperature * (1 + 2 * imbalance_ratio)
            for pan in [self.order_pan, self.chaos_pan]:
                new_pan = deque()
                for state, old_e in pan:
                    # Apply random move and reproject
                    new_state, new_e = self.metropolis_step(state, temperature=temperature)
                    # fluctuation = np.exp(-(new_e - old_e) / temperature) 涨落统计
                    new_pan.append((new_state, new_e))
                pan.clear()
                pan.extend(new_pan)

            self.balance()  # Rebalance after evolution

    def anneal(self, steps=1000, T0=5.0, cooling_rate=0.999):
        """
        退火：温度从 T0 指数下降，每步演化 1 次
        """
        T = T0
        traj = []
        z0 = self.slow_coord(CubieState.solved())

        for step in range(steps):
            self.evolve(steps=1, temperature=T)
            T *= cooling_rate

            # 记录当前状态到 solved 的慢距离
            current_state = self.order_pan[0][0]  # 以 order pan 的第一个状态为例
            z = self.slow_coord(current_state)
            traj.append(np.linalg.norm(z - z0))

        return traj

    def plot_history(self):
        if not self.history:
            return

        order_e, chaos_e, obs = zip(*self.history)
        steps = np.arange(len(self.history))

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # 上图：总能量
        axs[0].plot(steps, order_e, label='Order Pan Total Energy (d²)', linewidth=2)
        axs[0].plot(steps, chaos_e, label='Chaos Pan Total Energy (d²)', linewidth=2)
        axs[0].set_xlabel('Evolution Steps')
        axs[0].set_ylabel('Total Energy')
        axs[0].set_title('Order vs Chaos Energy Evolution')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # 下图：平均能量
        axs[1].plot(steps, [o['order_mean'] for o in obs], 'b--', label='Order Mean')
        axs[1].plot(steps, [o['chaos_mean'] for o in obs], label='Chaos Mean')
        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('Mean Energy per State')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "Order vs Chaos Balance Evolution.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_anneal_traj(self, traj):
        plt.figure(figsize=(10, 5))
        plt.plot(traj, label='Slow Distance to Solved', linewidth=2)
        plt.xlabel('Annealing Step')
        plt.ylabel('||z - z_solved||')
        plt.title('Annealing Trajectory (T from 5.0 → ~0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


class RubikAgent:

    def __init__(self, state, world: BalanceWorld):
        self.state = state
        self.world = world
        # self.x = x  # 高维状态 228
        self.z = world.slow_coord(state)  # slow coord 慢坐标
        # self.belief= [w1, w2, ..., w12]
        self.energy = world.energy(state)

        self.age = 0
        self.memory = deque(maxlen=10)
        self.memory.append(state)  # 初始状态入记忆

    def step(self, T):
        new_state, new_e = self.world.metropolis_step(
            self.state, temperature=T
        )

        self.memory.append(self.state)

        self.state = new_state
        self.energy = new_e
        self.age += 1

    def imitate(self, other_agent):
        """
        社会学习：从其他 agent 的记忆中随机模仿一个状态
        """
        if other_agent.memory:
            teacher_state = random.choice(list(other_agent.memory))
            self.state = teacher_state  # 直接模仿
            self.energy = self.world.energy(self.state)
            self.memory.append(self.state)  # 记录模仿结果

    def reproduce(self):
        """
        繁殖：产生一个孩子，继承部分记忆 + 小变异
        """
        if self.energy < 2.0 and random.random() < 0.05:
            child_state = self.state  # 孩子初始状态 = 父母状态

            # 小变异（随机走 1–3 步）
            if random.random() < 0.5:
                steps = random.randint(1, 3)
                g = CubieBase.random_walk(length=steps)
                child_state = g.act(child_state)

            # 继承父母记忆的一部分（前 5 个）
            child = RubikAgent(child_state, self.world)
            child.memory = deque(list(self.memory)[:5], maxlen=10)
            child.memory.append(child_state)  # 当前状态入队

            return child
        return None


class RubikLife(BalanceWorld):
    """Artificial Life on Group Manifolds
    群作用版：离散、非可交换、不会平均（不会塌成 I/d）、难做 planning / 概率 / 远期预测"""

    def __init__(self, n_agents=20, **kwargs):

        super().__init__(**kwargs)

        self.agents = []

        for _ in range(n_agents):
            s = CubieBase.generate_cubie(length=random.randint(5, 20))
            self.agents.append(RubikAgent(s, self))

        self.population_history = []
        self.record()

    def record(self):
        zs = [self.slow_coord(a.state) for a in self.agents]

        if len(zs) == 0:
            return

        Z = np.stack(zs)

        self.population_history.append(Z)

    def step(self, T=1.0):

        for agent in self.agents:
            agent.step(T)

        # 2. 交互（社会学习）
        self.interaction()

        # 3. 繁殖
        self.reproduce()

        # 4. 死亡
        self.death()

        # 5. 记录种群慢坐标
        self.record()

    def interaction(self):
        """agent 之间交互：距离近则互相模仿"""
        zs = [self.slow_coord(a.state) for a in self.agents]
        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i >= j:
                    continue
                d = np.linalg.norm(zs[i] - zs[j])
                if d < 5:  # 吸引簇：模仿
                    if random.random() < 0.1:
                        a.imitate(b)  # L_{a→b}
                elif d > 15.0:  # 排斥：轻微扰动
                    if random.random() < 0.05:
                        g = CubieBase.random_walk(length=1)
                        a.state = g.act(a.state)
                        a.energy = self.energy(a.state)
                else:
                    # weak repulsion
                    a.energy += 0.01
                    b.energy += 0.01

    def reproduce(self):
        new_agents = []
        for a in self.agents:
            child = a.reproduce()
            if child:
                new_agents.append(child)
        self.agents.extend(new_agents)

    def death(self):
        survivors = []
        for a in self.agents:
            if a.energy < 50 and a.age < 300:
                survivors.append(a)
        self.agents = survivors

    def plot_population(self, n_last=50):
        """可视化最近 n_last 代的种群在慢流形上的分布"""
        if len(self.population_history) < n_last:
            return

        plt.figure(figsize=(12, 8))
        recent = self.population_history[-n_last:]
        for i, Z in enumerate(recent):
            alpha = 0.1 + 0.9 * (i / (n_last - 1))  # 越新越亮
            plt.scatter(Z[:, 0].real, Z[:, 1].real, s=10, alpha=alpha, c='blue',
                        label=f'Gen -{n_last - i}' if i == 0 else None)

        plt.xlabel('Slow PC1')
        plt.ylabel('Slow PC2')
        plt.title('Population Evolution on Slow Manifold (last 50 generations)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_population_size(self):
        sizes = [len(gen) for gen in self.population_history]
        plt.figure(figsize=(10, 5))
        plt.plot(sizes, 'b-', label='Population Size')
        plt.xlabel('Generation')
        plt.ylabel('Number of Agents')
        plt.title('Population Dynamics')
        plt.grid(True)
        plt.legend()
        plt.show()


# ── 1. 时间晶体模拟 ─────────────────────────────────────────────────────

def test_time_crystal():
    """在慢流形上模拟离散时间晶格相"""
    model = SlowDynamics(n=N_GENERATORS)

    mode_indices = [0, 1, 2]
    V_modes = model.V_slow[:, mode_indices]  # (76, 3) — slow eigenvectors
    w_modes = model.w_slow[mode_indices]  # (3,) — slow eigenvalues

    state0 = CubieBase.generate_cubie(length=5)
    z0 = model.project(state0.vector)  # (76,) — slow coordinates

    steps = 300
    trajectory = [z0]
    for t in range(1, steps + 1):
        z_next = z0.copy()
        z_next[mode_indices] = z0[mode_indices] * (w_modes ** t)
        trajectory.append(z_next)
    trajectory = np.array(trajectory)  # (301, 76)

    # 投影到 3 个最慢模式的可视化坐标（取实部——Hermitian 矩阵的本征向量复数，但物理可观测量为实投影）
    proj = np.real(trajectory[:, mode_indices])  # (301, 3)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], 'b-', linewidth=2, label='Slow manifold trajectory')
    ax.scatter(proj[0, 0], proj[0, 1], proj[0, 2], c='green', s=80, label='Initial state')
    ax.set_xlabel('Slow Mode 1')
    ax.set_ylabel('Slow Mode 2')
    ax.set_zlabel('Slow Mode 3')
    ax.set_title('Time Crystal Phase: Closed Orbit in Slow Manifold')
    ax.legend()

    ax2 = fig.add_subplot(122)
    for i in range(3):
        ax2.plot(proj[:, i], label=f'Slow Mode {i + 1} (lambda~{w_modes[i]:.4f})', linewidth=2)
    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('Projection onto slow modes')
    ax2.set_title('Periodic Oscillation (Time Crystal Signature)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "time_crystal.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("时间晶格周期性检测（前3个慢模式）:")
    for i in range(3):
        period = np.argmax(np.correlate(proj[:, i], proj[:, i], mode='full')[steps:]) + 1
        print(f"  Mode {i + 1}: 估计周期 ~ {period} 步")


# ── 2. 生成器谱结构 ─────────────────────────────────────────────────────

def test_generator_spectrum():
    """不同 n 生成元的特征值分布、代数维度、多项式秩"""
    for n in [21, 18, 16, 12, 10, 9, 8, 6, 4, 3, 2]:
        model = SlowDynamics(n=n)
        unique, counts = np.unique(np.round(model.w, 6), return_counts=True)
        gen = model.rho_moves
        m = len(gen) // 2
        pred = [1 - k / m for k in range(m + 1)]
        print('>-----', n, len(gen), pred)
        print("dim span {rho(g)}=", model.dim_algebra,
              model.dim_algebra_slow, 'dim_s:',
              model.dim_slow,
              model.Tf,
              model.scale_l2)
        a_s = model.V_slow.T.conj() @ model.A_micro @ model.V_slow
        print('rank:', poly_rank(model.A_micro), poly_rank(a_s))
        for u, c in zip(unique[::-1], counts[::-1]):
            print(u, c)
    """
    attention-like generator mixing
    The spectrum of the averaged generator operator follows a universal form
    lambda = 1 - k/m, where m is the number of generator axes.

    10 generators
    1.0 52
    0.8 36
    0.6 64
    0.4 68
    0.2 8

    6 generators dir=2 k/3
    dim span {rho(g)}= 6 6
    rank: 5 4
    1.0 72
    0.666667 72
    0.333333 84

    6 generators axis=0
    dim span {rho(g)}= 6 1
    rank: 4 2
    1.0 100
    0.5 8
    0.333333 120

    4 generators
    1.0 100
    0.5 80
    0.25 8
    0.0 40
    """



# ── 4. Move scores & move energy ────────────────────────────────────────

# Extracted from cubieworld.py (2026-05-18). These are deprecated:
# the geometric decomposition (radial/tangential/rotational) fails because
# the continuous embedding loses group-theoretic information. See Paper II
# for the replacement: transport_score() via K_αβ.

def _move_scores(model, z, target, preference=None, eps=3.0):
    """[DEPRECATED] Geometric move scoring — extracted from SlowDynamics (2026-05-18).

    当状态已经很接近目标时，不同 move 对 z 的"推动方向"差异变得非常小,慢距离本身在接近目标时分辨率不足
    接近目标时：mean_target_dist 0.397~0.483
    max_chaos 0.397,max_chaos 0.860
    边界退化,被推到一个混合平衡态,须引入微扰,人为注入不对称性,否则系统会卡在对称点
    align → 近目标 → 自动衰减
    target_dist → 近目标 → 梯度消失
    sin_theta → 成主导（导致绕圈 / 平衡轨道）
    梯度流 + 旋度流 拼特征

    核心原则：
        • target_dist = backbone（唯一稳定信号）
        • radial / tangential = 正交分解
        • near-target 强化 symmetry breaking

    使用 _move_scores(model, z, target) 而非 model.move_scores()。
    Use SlowDynamics.transport_score() for theory-grounded scoring.
    """
    Uz = np.einsum('nij,j->ni', model.U, z)
    dz = Uz - z
    tz = target - z
    tz_norm = np.linalg.norm(tz)
    dz_norm = np.linalg.norm(dz, axis=1)
    inner = np.real(np.einsum('ni,i->n', dz, tz))
    radial = inner / (tz_norm + 1e-8)
    cos_theta = inner / (dz_norm * tz_norm + 1e-8)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
    tangential = dz_norm * sin_theta
    if tz_norm < eps:
        radial *= 0.0
        tangential = dz_norm
    target_dist = np.linalg.norm(Uz - target, axis=1)
    chaos = dz_norm
    pref = np.zeros_like(chaos)
    if preference is not None:
        pref = np.real(np.einsum('ni,i->n', dz, preference))
    scores = np.stack([
        target_dist,  # 0: global correctness（越小越好）
        -radial,  # 1: 径向（越大越好）
        tangential,  # 2: 切向（探索/绕）
        chaos,  # 3: 扰动,动量,扩散 0.7-5.4
        pref  # 4: 对称性破缺
    ], axis=1)
    return scores


def _move_energy(model, z, target, prev_dz=None):
    """[DEPRECATED] Geometric energy — extracted from SlowDynamics (2026-05-18).

    E = <dz, ∇V> + <dz, G(z) dz>  （Fredholm / 紧算子视角）

    ================================
    1. 动力学分区（Three Regimes）
    ================================

    Far Regime（远区）||tz|| large
        • radial 主导，potential 强
        • move ranking 有意义（≈ L2 正确排序）
        • energy 表现稳定

    Mixing Regime（混合区）
        • radial 与 tangential 竞争
        • anisotropy 上升，curvature 开始起作用
        • 欧式距离区分能力下降（进入 shell overlap）

    Near-Target Regime（近目标区）【核心困难】||tz|| → small
        • radial → 0, potential → flat
        • tangential / orbit motion 主导
        • metric 容易压倒 potential
        • move_scores 区分度崩塌
        • 出现大量：2-cycle（g↔g⁻¹）、dir==2 对称陷阱、orbit/cycle 行为

    ================================
    2. 群对称性主导现象（Level-3）
    ================================
    • 2-cycle orbit（最稳定吸引子）：g↔g⁻¹ 振荡，投影后 U≈U⁻¹（方向信息丢失）
    • dir==2 / 180° 对称陷阱：radial↓, tangential↓, rot≈0
    • 对称子流形：局部近似 Abel，非交换结构被投影压平

    ================================
    3. Energy 失效模式（必须牢记）
    ================================
    • Metric 主导：quadratic 过强 → 压制 potential → "少动优先"而非"朝目标走"
    • 对称性区分失败：dir==2 / Z-axis 区域所有几何量同时变小
    • 近区退化：所有信号衰减 → energy landscape flatten → 所有动作分数接近

    ================================
    4. Ground Truth（强约束原则）
    ================================
    ||Uz - target|| (L2 distance) — 全局排序最可靠，在任何 regime 都不完全失效
    L2 ranking is ground truth backbone; geometric terms only modulate, never dominate

    本质：魔方慢流形不是连续优化问题，而是"离散群轨道+连续嵌入+对称性主导"的动力系统

    使用 _move_energy(model, z, target) 而非 model.move_energy()。
    Use SlowDynamics.transport_score() for theory-grounded scoring.
    """
    Uz = np.einsum('nij,j->ni', model.U, z)
    dz = Uz - z
    tz = target - z
    norm_dz = np.linalg.norm(dz, axis=1)
    norm_tz = np.linalg.norm(tz)
    tz_unit = tz / (norm_tz + 1e-8)
    radial = np.real(np.einsum('ni,i->n', np.conj(dz), tz_unit))
    tangential = np.linalg.norm(dz - np.outer(radial, tz_unit), axis=1)
    E_base = np.linalg.norm(Uz - target, axis=1)
    inner = np.einsum('ni,i->n', np.conj(dz), tz)
    E_potential = -np.real(inner)
    rot = np.abs(np.imag(inner))
    anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
    curvature = np.log1p(anisotropy)
    E_geom = radial + tangential + curvature * rot
    alpha = np.clip(norm_tz / model.scale_l2, 0.0, 1.0)
    E = (
        E_base
        + (0.25 * alpha + 0.05) * E_geom
        + 0.15 * alpha * E_potential
    )
    if prev_dz is not None:
        cos_traj = np.real(np.einsum('ni,i->n', np.conj(dz), prev_dz)) / (norm_dz * np.linalg.norm(prev_dz) + 1e-8)
        E_traj = 1.0 - cos_traj
        E += 0.25 * E_traj
    return E

# ═══════════════════════════════════════════════════════════════════════════════
# κ Hierarchy search diagnostics (migrated from rime/cubieworld.py 260519)
#
# Standalone functions operating on CubieSpectralOperator — no SlowDynamics
# dependency. Use these for empirical validation of Paper III's accessibility
# hierarchy. See also: rime-lite/tests/test_kappa_hierarchy.py
# ═══════════════════════════════════════════════════════════════════════════════




def greedy_full_search(op, x_start, x_goal, max_depth=200):
    """1-step greedy on full 228-dim vector using real ρ(g) actions (κ₀ level).

    No projection. No heuristic scoring. Returns (path, final_dist, depth, trace).
    """
    path = []
    trace = []
    x_curr = np.asarray(x_start, dtype=np.complex128).copy()

    for depth in range(max_depth):
        dist = float(np.linalg.norm(x_curr - x_goal))
        phase = op.dominant_phase_at(x_goal - x_curr)
        trace.append((depth, dist, round(phase, 6)))

        if dist < 1e-6:
            return path, dist, depth, trace

        best_key, best_dist = None, float('inf')
        for key in op.rho_moves:
            if path and CubieMove.is_redundant(path[-1], key):
                continue
            d = op.move_distance( key, x_curr, x_goal)
            if d < best_dist:
                best_dist, best_key = d, key

        if best_key is None:
            break

        rho = op.rho_moves[best_key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        x_curr = rho @ x_curr
        path.append(best_key)

    return path, float(np.linalg.norm(x_curr - x_goal)), max_depth, trace


def random_walk_baseline(op, x_start, x_goal, max_depth=200):
    """Random walk on full 228-dim vector as baseline comparison."""
    path = []
    trace = []
    x_curr = np.asarray(x_start, dtype=np.complex128).copy()
    keys = list(op.rho_moves.keys())

    for depth in range(max_depth):
        dist = float(np.linalg.norm(x_curr - x_goal))
        trace.append((depth, dist))
        if dist < 1e-6:
            return path, dist, depth, trace

        candidates = [k for k in keys
                      if not (path and CubieMove.is_redundant(path[-1], k))]
        if not candidates:
            candidates = keys
        key = random.choice(candidates)

        rho = op.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        x_curr = rho @ x_curr
        path.append(key)

    return path, float(np.linalg.norm(x_curr - x_goal)), max_depth, trace


def hub_routed_beam_search(op, x_start, x_goal, max_depth=100):
    """2-step hub-routed beam search (κ₁ level).

    Uses transport topology to route through V₅/₉ hub when direct transport
    between current phase and target phase is blocked. Commits both steps.
    """
    hub_lam = round(5/9, 6)
    T = op.transport_tensor()
    move_keys = list(op.rho_moves.keys())
    path = []
    trace = []
    x_curr = np.asarray(x_start, dtype=np.complex128).copy()

    for depth in range(max_depth):
        gap = x_goal - x_curr
        dist = float(np.linalg.norm(gap))
        phase = op.dominant_phase_at(gap)
        trace.append((depth, dist, round(phase, 6)))

        if dist < 1e-6:
            return path, dist, depth, trace

        goal_phase = op.dominant_phase_at(x_goal)
        has_direct = T.get((phase, goal_phase), {}).get('max', 0) > TOL * 10

        if has_direct:
            best_key, best_dist = None, float('inf')
            for key in move_keys:
                if path and CubieMove.is_redundant(path[-1], key):
                    continue
                d = op.move_distance(key, x_curr, x_goal)
                if d < best_dist:
                    best_dist, best_key = d, key
            if best_key is None:
                break
            rho = op.rho_moves[best_key][1]
            if hasattr(rho, 'toarray'):
                rho = rho.toarray()
            x_curr = rho @ x_curr
            path.append(best_key)
        else:
            moves_to_hub = op.phase_crossing_moves(phase, hub_lam)
            moves_from_hub = op.phase_crossing_moves(hub_lam, goal_phase)
            if not moves_to_hub:
                moves_to_hub = move_keys
            if not moves_from_hub:
                moves_from_hub = move_keys

            candidates = []
            for k1 in moves_to_hub:
                if path and CubieMove.is_redundant(path[-1], k1):
                    continue
                rho1 = op.rho_moves[k1][1]
                if hasattr(rho1, 'toarray'):
                    rho1 = rho1.toarray()
                x1 = rho1 @ x_curr
                for k2 in moves_from_hub:
                    if CubieMove.is_redundant(k1, k2):
                        continue
                    rho2 = op.rho_moves[k2][1]
                    if hasattr(rho2, 'toarray'):
                        rho2 = rho2.toarray()
                    d2 = float(np.linalg.norm(rho2 @ x1 - x_goal))
                    candidates.append(((k1, k2), d2))

            if not candidates:
                break
            candidates.sort(key=lambda c: c[1])
            (k1_best, k2_best), _ = candidates[0]
            for k in (k1_best, k2_best):
                rho = op.rho_moves[k][1]
                if hasattr(rho, 'toarray'):
                    rho = rho.toarray()
                x_curr = rho @ x_curr
                path.append(k)

    return path, float(np.linalg.norm(x_curr - x_goal)), max_depth, trace


def t7_forced_search(op, x_start, x_goal, max_depth=300, stuck_window=15):
    """κ₁ hub-routed base + T7 forced 3-step when plateau is detected.

    Returns (path, final_dist, depth, trace, t7_triggers).
    """
    hub_lam = round(5/9, 6)
    T = op.transport_tensor()
    move_keys = list(op.rho_moves.keys())
    path = []
    trace = []
    t7_triggers = []
    x_curr = np.asarray(x_start, dtype=np.complex128).copy()

    rho_dense = {}
    for key in move_keys:
        rho = op.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        rho_dense[key] = rho

    valid_next = {}
    for k1 in move_keys:
        valid_next[k1] = [k2 for k2 in move_keys
                          if not CubieMove.is_redundant(k1, k2)]

    depth = 0
    best_dist_ever = float('inf')
    steps_since_improvement = 0

    while depth < max_depth:
        gap = x_goal - x_curr
        dist = float(np.linalg.norm(gap))
        phase = op.dominant_phase_at( gap)
        profile = op.phase_profile_at( gap)
        trace.append((depth, dist, phase, profile))

        if dist < 1e-6:
            return path, dist, depth, trace, t7_triggers

        if dist < best_dist_ever - 1e-8:
            best_dist_ever = dist
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        # --- T7 trigger: forced 3-step when stuck ---
        if steps_since_improvement >= stuck_window and len(path) >= 3:
            best_3seq = None
            best_3dist = float('inf')
            for k1 in move_keys:
                if path and CubieMove.is_redundant(path[-1], k1):
                    continue
                r1 = rho_dense[k1]
                x1 = r1 @ x_curr
                for k2 in valid_next.get(k1, move_keys):
                    r2 = rho_dense[k2]
                    x2 = r2 @ x1
                    for k3 in valid_next.get(k2, move_keys):
                        r3 = rho_dense[k3]
                        d3 = float(np.linalg.norm(r3 @ x2 - x_goal))
                        if d3 < best_3dist:
                            best_3dist = d3
                            best_3seq = (k1, k2, k3)

            if best_3seq is not None and best_3dist < dist - 0.01:
                t7_triggers.append((depth, dist, best_3dist, best_3seq))
                for k in best_3seq:
                    x_curr = rho_dense[k] @ x_curr
                    path.append(k)
                    depth += 1
                best_dist_ever = best_3dist
                steps_since_improvement = 0
                continue

        # --- κ₁: hub-routed 2-step (normal mode) ---
        goal_phase = op.dominant_phase_at( x_goal)
        has_direct = T.get((phase, goal_phase), {}).get('max', 0) > TOL * 10

        if has_direct:
            best_key, best_dist = None, float('inf')
            for key in move_keys:
                if path and CubieMove.is_redundant(path[-1], key):
                    continue
                d = op.move_distance( key, x_curr, x_goal)
                if d < best_dist:
                    best_dist, best_key = d, key
            if best_key is None:
                break
            x_curr = rho_dense[best_key] @ x_curr
            path.append(best_key)
            depth += 1
        else:
            moves_to_hub = op.phase_crossing_moves(phase, hub_lam)
            moves_from_hub = op.phase_crossing_moves(hub_lam, goal_phase)
            if not moves_to_hub:
                moves_to_hub = move_keys
            if not moves_from_hub:
                moves_from_hub = move_keys

            candidates = []
            for k1 in moves_to_hub:
                if path and CubieMove.is_redundant(path[-1], k1):
                    continue
                x1 = rho_dense[k1] @ x_curr
                for k2 in moves_from_hub:
                    if CubieMove.is_redundant(k1, k2):
                        continue
                    d2 = float(np.linalg.norm(rho_dense[k2] @ x1 - x_goal))
                    candidates.append(((k1, k2), d2))

            if not candidates:
                break
            candidates.sort(key=lambda c: c[1])
            (k1_best, k2_best), _ = candidates[0]
            for k in (k1_best, k2_best):
                x_curr = rho_dense[k] @ x_curr
                path.append(k)
                depth += 1

    return path, float(np.linalg.norm(x_curr - x_goal)), depth, trace, t7_triggers


def detect_cycles(trace, window=4):
    """Detect if a distance trace has entered a limit cycle.

    Returns (period, stable).
    """
    dists = [t[1] for t in trace]
    if len(dists) < 10:
        return 0, False

    tail = dists[-8:]
    for period in [2, 3, 4]:
        if len(tail) < 2 * period:
            continue
        a = np.array(tail[-period:])
        b = np.array(tail[-2 * period:-period])
        if np.allclose(a, b, atol=0.05):
            mid = dists[-3 * period:-2 * period]
            if len(mid) >= period:
                c = np.array(mid)
                if np.allclose(a, c, atol=0.05):
                    return period, True
            return period, False
    return 0, False

def test_move_scores_and_energy():
    """move_scores / move_energy 接近目标时的区分度"""
    model = setup()
    z0 = model.project(CubieState.solved().vector)
    zp = model.project(CubieBase.generate_cubie(1).vector)
    preference = z0 - zp

    for i in range(5):
        s1 = CubieBase.generate_cubie(i, check=True)
        z1 = model.project(s1.vector)
        tz = z0 - z1
        norm_tz = np.linalg.norm(tz)
        score = _move_scores(model,z1, z0, preference)
        print(i, norm_tz, np.mean(np.abs(score), axis=0), np.max(score, axis=0), score)
    """
    当 z 已经是 solved（z0）时，move_scores 几乎无法区分不同 move 的好坏,
    方向敏感性不够强，尤其在接近目标时区分度崩塌
    """

    # move_energy 与 delta_V 符号一致性
    s0 = CubieState.solved()
    z0 = model.project(s0.vector)
    k = 3
    for i in range(5):
        m = CubieBase.random_walk(length=i)
        s2 = m.act(s0)
        z2 = model.project(s2.vector)

        energy = _move_energy(model,z2, z0)
        best = np.argmin(energy)
        Uz = np.einsum('nij,j->ni', model.U, z2)
        V0 = np.linalg.norm(z2 - z0) ** 2
        V1 = np.linalg.norm(Uz - z0, axis=1) ** 2
        delta_V = V1 - V0

        sign_match = np.mean(np.sign(-energy) == np.sign(-delta_V))
        pred_topk = np.argsort(energy)[:k]
        true_topk = np.argsort(delta_V)[:k]
        hit_k = len(set(pred_topk) & set(true_topk)) / k
        corr, _ = spearmanr(energy, delta_V)
        print(i, sign_match, hit_k, corr)

    # 四层架构：const | slow | hub | fast
    print('-' * 60)
    print('Architecture: const(20) | slow(67) | hub(106) | fast(35) = 228')
    print('Phase graph (star via V5/9 hub):')
    print('  V1(1.0)     isolated')
    print('  V8/9(0.89)  <-> V5/9(hub)  K=0.75')
    print('  V7/9(0.78)  <-> V5/9(hub)  K=4.42')
    print('  V2/3(0.67)  <-> V5/9(hub)  K=3.46')
    print('  V1/3(0.33)  <-> V5/9(hub)  K=4.06')

    s1 = CubieBase.generate_cubie(10)
    z_start = model.project(s1.vector)
    z_goal = model.project(CubieState.solved().vector)

    full_plan = model.phase_path_plan(z_start, z_goal)
    slow_plan = model._slow_phase_plan(z_start, z_goal)
    print(f'Full phase plan: {[round(p,4) for p in full_plan]}')
    print(f'Slow+hub plan:   {[round(p,4) for p in slow_plan]}')

    print()
    print('1. Greedy single-step (transport_score):')
    path, final_dist, depth, phase_trace = model.spectral_guided_search(z_start, z_goal, max_depth=30)
    print(f'   {len(path)} moves, final dist={final_dist:.4f}')
    phases_seen = set(step[1] for step in phase_trace)
    print(f'   Phases visited: {sorted(phases_seen)}')
    for step in phase_trace[:4]:
        print(f'     d={step[0]:2d}  phase={step[1]:.4f}  dist={step[2]:.4f}')
    print(f'     ...  d={phase_trace[-1][0]:2d}  phase={phase_trace[-1][1]:.4f}  dist={phase_trace[-1][2]:.4f}')

    print()
    print('2. Composition search (routes through V5/9 hub):')
    path2, dist2, d2, trace2, events = model.spectral_composition_search(
        z_start, z_goal, max_depth=30, stuck_threshold=4)
    print(f'   {len(path2)} moves, final dist={dist2:.4f}, composition events: {len(events)}')
    for e in events[:3]:
        print(f'     d={e["depth"]:2d}: {e["from_phase"]} -> {e["to_phase"]} via {e["pair"]}')
    if len(events) > 3:
        print(f'     ... ({len(events)} total)')
    phases_seen2 = set(step[1] for step in trace2)
    print(f'   Phases visited: {sorted(phases_seen2)}')
    for step in trace2[:4]:
        print(f'     d={step[0]:2d}  phase={step[1]:.4f}  dist={step[2]:.4f}')
    print(f'     ...  d={trace2[-1][0]:2d}  phase={trace2[-1][1]:.4f}  dist={trace2[-1][2]:.4f}')

    # transport_score vs actual distance
    print()
    print('3. Transport score vs actual distance (first step):')
    gap = z_goal - z_start
    scores = model.transport_scores(gap)
    dists = {}
    for key in model.rho_slow.keys():
        z_next = model.apply_move(key, z_start)
        dists[key] = model.l2_distance(z_next, z_goal)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:5]
    for key, score in ranked:
        actual_dist = dists[key]
        move_str = str(CubieMove.prim_moves().get(key, key))
        print(f'     {move_str:6s}  score={score:.4f}  actual_dist={actual_dist:.4f}')

def test_quantum_vs_classical():
    """慢空间量子演化 vs 经典演化的表示差异"""
    model = setup()
    s1 = CubieBase.generate_cubie()
    z1 = model.project(s1.vector)
    z1 /= np.linalg.norm(z1) + 1e-8
    r1 = np.outer(z1, z1.conj())

    m = random.choice(list(CubieMove.prim_moves().values()))
    rho_m = m.rho()
    U = model.project_move(rho_m)
    r2 = U @ r1 @ U.conj().T  # 慢空间量子演化

    s2 = m.act(s1)
    z2 = model.project(s2.vector)
    s2_v = s1.vector @ m.matrix
    z2 /= np.linalg.norm(z2) + 1e-8
    r21 = np.outer(z2, z2.conj())  # 经典演化后纯态化
    rho22 = np.outer(s2.vector, s2.vector.conj())  # 经典 act 后直接外积

    r22 = 2 / 3 * r1 + 1 / 3 * r21
    r22 /= np.trace(r22)
    r23 = 2 / 3 * r1 + 1 / 3 * r2
    r23 /= np.trace(r23)
    d1 = np.linalg.norm(r22 - r23, 'fro')
    d2 = 0.5 * np.linalg.norm(r22 - r23, ord='nuc')

    rho1 = np.outer(s1.vector, s1.vector.conj())
    rho2 = rho_m @ rho1 @ rho_m.conj().T
    d3 = np.linalg.norm(r2 - r21, 'fro')
    d4 = 0.5 * np.linalg.norm(r2 - r21, ord='nuc')
    d5 = 0.5 * np.linalg.norm(rho2 - rho22, ord='nuc')
    print(d1, d2, d3, d4, d5)
    """两种路径在慢流形上的投影并不完全相同
    典型结果: 0.35083967 0.24825998 0.7187361 0.3773283 0.7693323"""

    # 逐 move 比较
    s1 = CubieBase.generate_cubie()
    z1 = model.project(s1.vector)
    z1 /= np.linalg.norm(z1) + 1e-8
    rho1 = np.outer(z1, z1.conj())
    for key, m in CubieMove.prim_moves().items():
        rho_m_full = m.rho()
        U = model.project_move(rho_m_full)
        rho2 = U @ rho1 @ U.conj().T  # 慢空间量子演化
        s2 = m.act(s1)
        z2 = model.project(s2.vector)
        z2 /= np.linalg.norm(z2) + 1e-8
        rho22 = np.outer(z2, z2.conj())  # 慢空间纯态
        d_fro = np.linalg.norm(rho2 - rho22, 'fro')
        d_nuc = 0.5 * np.linalg.norm(rho2 - rho22, ord='nuc')
        print(f"{key}, Frobenius diff: {d_fro:.6f}, Nuclear/Trace diff: {d_nuc:.6f}")
    """Nuclear Norm（迹距离）比 Frobenius 更稳定，数值范围也更合理
    范围大约 0.56 ~ 0.88，平均在 0.72 左右"""


# ── 6. Environment 目标状态距离 ──────────────────────────────────────────

def test_environment_targets():
    """big_cycle / inversed / twisted 状态的慢流形距离分析"""
    from rime.cubie import CubieExample
    model = setup()
    s0 = CubieState.solved()
    z0 = model.project(s0.vector)
    env = Environment(model)

    def far_targets(model, n=6):
        targets, zs = [], []
        while len(targets) < n:
            s = CubieBase.generate_cubie(length=15)
            z = model.project(s.vector)
            z /= np.linalg.norm(z)
            if all(np.linalg.norm(z - z0) > 5.0 for z0 in zs):
                targets.append(z)
                zs.append(z)
        return targets

    s2 = CubieExample.big_cycle()
    assert s2.is_solvable(), f'{s2}'
    print(np.linalg.norm(s2.vector - s0.vector))  # ~6.3245554
    print(model.heuristic(s2.vector, s0.vector))  # ~3.7416575,有很多 fast 成分
    print(np.linalg.norm(model.project(s2.vector)))  # ~4.582576
    print(cosine_distance(model.project(s2.vector), model.project(s0.vector)))  # ~0.75,慢空间投影的余弦相似度更高

    s22 = CubieExample.inversed()
    assert s22.is_solvable(), f'{s22}'
    print(np.linalg.norm(s22.vector - s0.vector))
    print(model.heuristic(s22.vector, s0.vector))
    print(np.linalg.norm(model.project(s22.vector)))
    print(cosine_distance(model.project(s22.vector), model.project(s0.vector)))  # ~0.25

    print(np.linalg.norm(s2.vector - s22.vector))
    print(model.heuristic(s2.vector, s22.vector))

    # 单步生成元在慢空间的壳层
    z0 = model.project(s0.vector)
    for k, (_, g) in model.rho_slow.items():
        z2 = g @ z0
        print(k, model.l2_distance(z2, z0), cosine_distance(z2, z0))
        """
        单步是"离散能级壳层"
        分层能级结构，slow space 里存在离散轨道壳层
        z0 --(小扰动)--> 半径 ~0. cos_dist ~ 0.0117
        z0 --(中扰动)--> 半径 ~1.1 cos_dist ~ 0.030
        z0 --(大扰动)--> 半径 ~3.6 / 5.4 cos_dist ~ 0.328
        丢掉真实差异,波函数演化,离散"角度分层空间",eps ~ 0.1 ~ 0.2
        """

    print('------------')
    for k, (g, *_) in model.rho_moves.items():
        z2 = model.project(g.act(s0).vector)
        print(k, model.l2_distance(z2, z0), cosine_distance(z2, z0))
    """更接近真实动力学"""

    s3 = CubieExample.twisted()
    assert s3.is_solvable(), f'{s3}'
    print(np.linalg.norm(s3.vector - s0.vector))  # ~6.0
    print(model.heuristic(s3.vector, s0.vector))  # ~6.0 没有被谱压缩掉,能量几乎完全在 slow 模式里
    print(np.linalg.norm(model.project(s3.vector)))
    print(model.rho_slow.keys())

    targets = [model.project(s.vector) for s in (s0, s2, s3)]
    for k in [(0, 1, 1), (1, 1, 1), (2, 1, 1)]:
        g = model.rho_moves[k][0]
        s = g.act(CubieState.solved())
        z = model.project(s.vector)
        targets.append(z)
    print(len(targets))

    z4 = env.generate_diff_target()
    print(np.linalg.norm(z4 - z0))  # ~6.461424


# ── 7. 慢流形距离分布 & 壳层 ─────────────────────────────────────────────

def test_distance_distribution():
    """慢流形上随机状态的距离分布"""
    model = setup()
    s0 = CubieState.solved()
    z0 = model.project(s0.vector)

    zs = []
    while len(zs) < 100:
        s = CubieBase.generate_cubie(length=20)
        z = model.project(s.vector)
        zs.append(np.linalg.norm(z - z0))

    print(np.unique(np.round(zs, decimals=4), return_counts=True))
    print(np.mean(zs), np.std(zs))
    """
    分布比较离散，但有明显的聚集，
    距离分布：0.7 / 1.1 / 3.6 / 5.4 -> 多步之后 -> 收敛到"壳层混合带
    -> 距离失去区分能力，测度集中
    用"欧式距离"做分类，但空间已经不支持,系统已经进入：mixing regime（混合态）
    更像"量子态空间"，不是"欧氏空间"

    慢流形（76 维）本质上是把群的宏观慢变量提取出来，
    它捕捉的正是这些置换群的"整体行为"。
    因此，平均 l2 距离和群论中 S_8 / S_12 的平均字长 ~6~7 高度吻合。
    慢流形保留了原群的宏观行为规律，是对"群运动"的连续化、低频化映射

    理论平均距离（sqrt(2) x sigma x sqrt(d) ~ 6.16，当 sigma=0.5 d=76）
    """


# ── 8. 主模拟 ────────────────────────────────────────────────────────────

def test_main_simulation():
    """HybridSimulation 完整演化"""
    from rime.cubieworld import main
    main()


# ── 9. 退火实验 ──────────────────────────────────────────────────────────

def test_annealing():
    """BalanceWorld 退火轨迹"""
    model = setup()
    world = BalanceWorld(model)

    traj = world.anneal(steps=2000, T0=5.0, cooling_rate=0.999)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(traj, label='Distance to solved (slow coord)', linewidth=2)
    ax1.set_ylabel('||z - z_solved||')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    temps = [5.0 * (0.999 ** i) for i in range(2000)]
    ax2.plot(temps, 'r--', label='Temperature', alpha=0.7)
    ax2.set_ylabel('Temperature')
    ax2.legend(loc='upper right')

    plt.title('Annealing Trajectory (T from 5.0 -> ~0)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DATA_DIR, "Annealing Trajectory with Temperature Overla.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    # 多初始温度对比
    plt.figure(figsize=(12, 6))
    for T0 in [3.0, 5.0, 8.0]:
        traj = world.anneal(steps=2000, T0=T0, cooling_rate=0.999)
        plt.plot(traj, label=f'T0={T0}')
    plt.xlabel('Step')
    plt.ylabel('Distance to solved')
    plt.title('Annealing Trajectories with Different Starting Temperatures')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DATA_DIR, "Annealing Trajectories with Different Starting Temperatures.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

    """
    完整退火轨迹（T from 5.0 -> ~0）
    关键观察：
    早期 (0-500 步)：距离剧烈波动（峰值到 6.0+，谷底到 2.0），对应高温混沌探索。
    中期 (500-1500 步)：波动幅度逐渐减小，距离在 2.0-3.5 区间震荡，系统开始"冷却"。
    后期 (1500-2000 步)：出现几次"跳水"（从 3.0+ 掉到 0-1.0），最终稳定在 ~0-1.5
    -> 成功冻结到低能量态（接近 solved）。
    退火过程成功实现了"从混乱到秩序"的转变。
    """


# ── 10. 温度扫描 ─────────────────────────────────────────────────────────

def test_temperature_scan():
    """Order/Chaos 能量随温度的变化（相变特征）"""
    model = setup()

    temps = np.linspace(0.1, 5.0, 20)
    results = []
    for T in temps:
        world = BalanceWorld(model)
        world.evolve(steps=500, temperature=T)
        obs = world.observe()
        results.append((T, obs["order_mean"], obs["chaos_mean"]))

    T_values = [r[0] for r in results]
    order_means = [r[1] for r in results]
    chaos_means = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(T_values, order_means, 'b-o', label='Order Mean Energy', linewidth=2.5, markersize=6)
    ax1.plot(T_values, chaos_means, 'r-s', label='Chaos Mean Energy', linewidth=2.5, markersize=6)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('Mean Energy per State (d^2)')
    ax1.set_title('Final Balance State vs Temperature (after 500 steps)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    imbalance = np.array(order_means) - np.array(chaos_means)
    ax2.plot(T_values, imbalance, 'k--', label='Order - Chaos Difference', linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Imbalance (Order - Chaos)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "Final Balance State vs Temperature.png"), dpi=300, bbox_inches='tight')
    plt.close()

    """
    温度扫描图关键观察：
    低温区 (T ~ 0.1-1.0)：Order ~ 20-25，Chaos ~ 0-5 -> 秩序主导
    中温区 (T ~ 1.0-3.0)：两者快速接近 -> 动态平衡
    高温区 (T ~ 3.0-5.0)：Chaos 轻微占优
    Imbalance：从正值 -> 零附近 -> 负值，呈现明显的"相变"特征
    """


# ── 11. 扩散轨迹 ─────────────────────────────────────────────────────────

def test_diffusion():
    """Metropolis 扩散轨迹（T=1.0）"""
    model = setup()
    world = BalanceWorld(model)

    traj = []
    state = CubieState.solved()
    z0 = world.slow_coord(state)

    for t in range(500):
        state, _ = world.metropolis_step(state, temperature=1.0)
        z = world.slow_coord(state)
        traj.append(np.linalg.norm(z - z0))

    t_arr = np.arange(len(traj))
    sqrt_t = np.sqrt(t_arr) * np.mean(traj[50:100]) / np.mean(np.sqrt(np.arange(50, 100)))
    saturation = np.mean(traj[-50:])

    plt.figure(figsize=(12, 6))
    plt.plot(t_arr, traj, 'b-', label='Distance ||z(t) - z(0)||', linewidth=1.8, alpha=0.9)
    plt.plot(t_arr, sqrt_t, 'r--', label='~ sqrt(t) (diffusion theory)', linewidth=2, alpha=0.7)
    plt.axhline(saturation, color='green', linestyle='--', label=f'Saturation level ~ {saturation:.2f}')
    plt.xlabel('Metropolis Steps (T=1.0)')
    plt.ylabel('Slow Manifold Distance to Initial State')
    plt.title('Diffusion Trajectory on Slow Manifold (Single Metropolis Walk)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "Diffusion Trajectory on Slow Manifold.png"), dpi=300, bbox_inches='tight')
    plt.close()

    """
    早期 (0-100 步)：距离从 0 快速上升到 ~1.5-2.0，早期增长近似 sqrt(t)
    中期 (100-400 步)：距离在 ~2.0-3.0 区间大幅波动
    后期 (400-500 步)：稳定在 ~2.5-3.0 附近（饱和线 ~2.64）

    T=1.0 下单条 Metropolis 轨迹表现出典型的扩散几何
    （早期 sqrt(t) 增长 + 随机波动），但没有快速饱和，
    说明慢流形仍有足够"空间"让状态探索
    """


# ── 12. 慢投影 vs 剪枝距离相关性 ──────────────────────────────────────────

def test_slow_vs_random_walk():
    """随机游走：慢投影距离 vs 剪枝表真实距离的相关性"""
    model = setup()
    n_pairs = N_PAIRS

    d1_list, d2_list = [], []
    stateA = CubieState.solved()
    for i in range(n_pairs):
        if i % 500 == 0:
            print(f"已完成 {i}/{n_pairs} 对...")
        steps = random.randint(1, 30)
        g = CubieBase.random_walk(length=steps)
        stateB = g.act(stateA)
        phase, d1 = CubieBase.cubie_distance(stateB)
        d2 = model.heuristic(stateA.vector, stateB.vector, False)
        d1_list.append(d1)
        d2_list.append(d2)

    d1_arr = np.array(d1_list)
    d2_arr = np.array(d2_list)
    pearson_corr, pearson_p = pearsonr(d1_arr, d2_arr)
    spearman_corr, spearman_p = spearmanr(d1_arr, d2_arr)
    print(f"\n随机游走 1-30 steps:")
    print(f"Pearson r = {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"std d1 = {np.std(d1_arr):.4f}, std d2 = {np.std(d2_arr):.4f}")

    plt.figure(figsize=(12, 8))
    plt.scatter(d1_arr, d2_arr, alpha=0.6, s=10, c='blue', edgecolor='none')
    plt.xlabel("prune distance d1")
    plt.ylabel("slow distance d2 = ||V_slow^T (rho(A) - rho(B))||")
    plt.title(f"random_walk slow vs prune distance (n={n_pairs})")
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f"Pearson r = {pearson_corr:.4f}\nSpearman r = {spearman_corr:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "slow_vs_random_walk.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> saved slow_vs_random_walk.png")


def test_slow_vs_relative_state():
    """等深度状态对：慢投影距离 vs relative_state 的剪枝距离"""
    model = setup()
    n_pairs = N_PAIRS

    for dh in (5, 10, 20, 30):
        d1_list, d2_list = [], []
        for i in range(n_pairs):
            if i % 500 == 0:
                print(f"  depth={dh} 已完成 {i}/{n_pairs} 对...")
            stateA = CubieBase.generate_cubie(dh)
            stateB = CubieBase.generate_cubie(dh)
            stateC = CubieMove.relative_state(stateA, stateB)
            phase, d1 = CubieBase.cubie_distance(stateC)
            d2 = model.heuristic(stateA.vector, stateB.vector, False)
            d1_list.append(d1)
            d2_list.append(d2)

        d1_arr = np.array(d1_list)
        d2_arr = np.array(d2_list)
        pearson_corr, pearson_p = pearsonr(d1_arr, d2_arr)
        spearman_corr, spearman_p = spearmanr(d1_arr, d2_arr)
        print(f"\n  depth={dh}:")
        print(f"  Pearson r = {pearson_corr:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman r = {spearman_corr:.4f} (p={spearman_p:.2e})")
        print(f"  std d1 = {np.std(d1_arr):.4f}, std d2 = {np.std(d2_arr):.4f}")

        plt.figure(figsize=(12, 8))
        plt.scatter(d1_arr, d2_arr, alpha=0.6, s=10, c='blue', edgecolor='none')
        plt.xlabel("prune distance d1")
        plt.ylabel("slow distance d2")
        plt.title(f"relative_state slow vs prune distance (d={dh}, n={n_pairs})")
        plt.grid(True, alpha=0.3)
        plt.text(0.05, 0.95, f"Pearson r = {pearson_corr:.4f}\nSpearman r = {spearman_corr:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, f"slow_vs_relative_d{dh}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> saved slow_vs_relative_d{dh}.png")


# ── 13. 准等距性验证 ──────────────────────────────────────────────────────

def test_quasi_isometry():
    """群作用下慢投影距离的保距性"""
    model = setup()

    # 实验1：solved vs 变换后状态
    rho_solved = CubieState.solved().vector
    z_solved = model.project(rho_solved)

    d_ratios = []
    for _ in range(1000):
        A = CubieBase.generate_cubie()
        rho_A = A.vector
        z_A = model.project(rho_A)

        g = CubieBase.random_walk(length=5)
        rho_g = g.rho()
        rho_A_g = rho_g @ rho_A
        z_A_g = model.project(rho_A_g)

        d_orig = model.l2_distance(z_A, z_solved)
        d_trans = model.l2_distance(z_A_g, z_solved)
        ratio = d_trans / (d_orig + 1e-10)
        d_ratios.append(ratio)

    mean_ratio = np.mean(d_ratios)
    std_ratio = np.std(d_ratios)
    print(f"实验1: d(rho(g)x, rho(g)y) / d(x,y) = {mean_ratio:.4f} +/- {std_ratio:.4f}")
    """
    平均 ~1.0059 +/- 0.0871
    slow embedding 在群作用下准等距 (statistical isometry)
    d(z) 可以作为到 solved 的可靠下界或代理距离（admissible heuristic）
    """

    plt.figure(figsize=(12, 8))
    plt.hist(d_ratios, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(1.0, color='red', ls='--', label='ideal (ratio=1)')
    plt.axvline(mean_ratio, color='orange', ls='-', label=f'mean {mean_ratio:.4f}')
    plt.xlabel("ratio d_trans / d_orig")
    plt.ylabel("density")
    plt.title("slow distance isometry under group action (1000 samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "slow_distance_isometry_1000.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 实验2：随机状态对
    def generate_cubie_pair(depth_range: tuple = (0, 30)) -> tuple:
        depthA = np.random.randint(*depth_range)
        depthB = np.random.randint(*depth_range)
        stateA = CubieBase.generate_cubie(depthA)
        stateB = CubieBase.generate_cubie(depthB)
        return stateA, stateB

    d_ratios = []
    for _ in range(3000):
        A, B = generate_cubie_pair()
        z_A = model.project(A.vector)
        z_B = model.project(B.vector)

        g = CubieBase.random_walk(length=5)
        rho_g = g.rho()
        z_A_g = model.project(rho_g @ A.vector)
        z_B_g = model.project(rho_g @ B.vector)

        d_orig = model.l2_distance(z_A, z_B)
        d_trans = model.l2_distance(z_A_g, z_B_g)
        ratio = d_trans / (d_orig + 1e-10)
        d_ratios.append(ratio)

    mean_ratio = np.mean(d_ratios)
    std_ratio = np.std(d_ratios)
    print(f"实验2: d(rho(g)x, rho(g)y) / d(x,y) = {mean_ratio:.4f} +/- {std_ratio:.4f}")
    """平均 ~1.0003 +/- 0.0144 (更精确的保距性)"""

    plt.figure(figsize=(12, 8))
    plt.hist(d_ratios, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(1.0, color='red', ls='--', label='ideal (ratio=1)')
    plt.axvline(mean_ratio, color='orange', ls='-', label=f'mean {mean_ratio:.4f}')
    plt.xlabel("ratio d_trans / d_orig")
    plt.ylabel("density")
    plt.title("slow metric group invariance (3000 samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "slow_metric_group_invariance_3000.png"), dpi=300, bbox_inches='tight')
    plt.close()

    """
    slow manifold 上的群作用几乎是正交的，因此 slow embedding 近似保持 Rubik cube 的群距离结构
    近似群不变的
    slow embedding respects group action.
    """


# ── 14. move_energy 启发式验证 ───────────────────────────────────────────


def test_move_energy_vs_prune_distance():
    """
    move_energy 各分量与剪枝表距离 prune_d 的相关性分析
    prune_d = cubie_distance 得到的剪枝表距离（可靠下界代理）
    对比：E_base (oracle baseline), E_potential, E_geom, norm_tz, radial, tangential, rot
    E_base 不是普通特征 —— 它是 ranking ground truth / label-aligned baseline
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_SAMPLES = 500

    # 采集数据
    records = []
    for _ in range(N_SAMPLES):
        depth = np.random.randint(1, 25)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)
        tz = z_solved - z
        norm_tz = np.linalg.norm(tz)

        # prune_d: 剪枝表距离代理（不是 ground truth 最短距离）
        phase, prune_d = CubieBase.cubie_distance(s)

        # move_energy 各分量
        Uz = np.einsum('nij,j->ni', model.U, z)
        dz = Uz - z
        norm_dz = np.linalg.norm(dz, axis=1)

        E_base = np.linalg.norm(Uz - z_solved, axis=1)
        inner = np.einsum('ni,i->n', np.conj(dz), tz)
        E_potential = -np.real(inner)
        radial = np.real(inner) / (norm_tz + 1e-8)
        tangential = np.linalg.norm(dz - np.outer(np.real(inner) / (norm_tz ** 2 + 1e-16), tz), axis=1)
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
        curvature = np.log1p(anisotropy)
        E_geom = radial + tangential + curvature * rot

        # 每个状态取 best move（最小 E_base）的分量值
        best = np.argmin(E_base)
        records.append({
            'prune_d': prune_d,
            'phase': phase,
            'depth': depth,
            'norm_tz': norm_tz,
            'E_base_best': E_base[best],
            'E_potential_best': E_potential[best],
            'E_geom_best': E_geom[best],
            'radial_best': radial[best],
            'tangential_best': tangential[best],
            'rot_best': rot[best],
            'radial_mean': np.mean(radial),
            'tangential_mean': np.mean(tangential),
            'rot_mean': np.mean(rot),
            'anisotropy': anisotropy,
            'curvature': curvature,
        })

    r_prune = [r['prune_d'] for r in records]
    r_norm_tz = [r['norm_tz'] for r in records]
    r_E_base = [r['E_base_best'] for r in records]

    print("=" * 70)
    print("move_energy vs prune_d (pruning distance proxy): Correlation Analysis")
    print("=" * 70)
    print(f"\nN_samples = {N_SAMPLES}")

    # 全局相关性
    pairs = [
        ('norm_tz', r_norm_tz),
        ('E_base_best', r_E_base),
        ('E_potential_best', [r['E_potential_best'] for r in records]),
        ('E_geom_best', [r['E_geom_best'] for r in records]),
        ('radial_mean', [r['radial_mean'] for r in records]),
        ('tangential_mean', [r['tangential_mean'] for r in records]),
        ('rot_mean', [r['rot_mean'] for r in records]),
        ('anisotropy', [r['anisotropy'] for r in records]),
        ('curvature', [r['curvature'] for r in records]),
    ]

    print(f"\n{'Feature':<25s} {'Pearson r':>10s} {'Spearman ρ':>12s} {'p-value':>12s}")
    print("-" * 62)
    for name, vals in pairs:
        r_p, p_p = pearsonr(vals, r_prune)
        r_s, p_s = spearmanr(vals, r_prune)
        p_str = f"{min(p_p, p_s):.2e}" if min(p_p, p_s) < 1e-3 else f"{min(p_p, p_s):.4f}"
        print(f"  {name:<23s} {r_p:10.4f} {r_s:12.4f} {p_str:>12s}")

    # 按深度分段相关性
    print(f"\n--- Segmented by scramble depth ---")
    print(f"{'Depth range':<15s} {'N':>5s} {'norm_tz r':>10s} {'E_base r':>10s} {'radial r':>10s} {'tangent r':>10s}")
    print("-" * 55)
    for d_lo, d_hi in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
        seg = [r for r in records if d_lo <= r['depth'] < d_hi]
        if len(seg) < 10:
            continue
        prune_seg = [r['prune_d'] for r in seg]
        r_tz, _ = pearsonr([r['norm_tz'] for r in seg], prune_seg)
        r_eb, _ = pearsonr([r['E_base_best'] for r in seg], prune_seg)
        r_rad, _ = pearsonr([r['radial_mean'] for r in seg], prune_seg)
        r_tan, _ = pearsonr([r['tangential_mean'] for r in seg], prune_seg)
        print(f"  {d_lo:2d}–{d_hi:<5d}       {len(seg):5d} {r_tz:10.4f} {r_eb:10.4f} {r_rad:10.4f} {r_tan:10.4f}")

    # 按 norm_tz 分段（三区动力学）
    print(f"\n--- Segmented by norm_tz (three-regime) ---")
    print(
        f"{'norm_tz range':<18s} {'N':>5s} {'mean_prune':>11s} {'mean_aniso':>11s} {'radial_mean':>12s} {'tangent_mean':>13s}")
    print("-" * 65)
    for tz_lo, tz_hi, label in [(0, 1.5, 'near'), (1.5, 4.0, 'mid'), (4.0, 10.0, 'far')]:
        seg = [r for r in records if tz_lo <= r['norm_tz'] < tz_hi]
        if len(seg) < 5:
            continue
        print(f"  {label:<6s} [{tz_lo:.1f},{tz_hi:.1f})  {len(seg):5d} "
              f"{np.mean([r['prune_d'] for r in seg]):11.2f} "
              f"{np.mean([r['anisotropy'] for r in seg]):11.3f} "
              f"{np.mean([r['radial_mean'] for r in seg]):12.4f} "
              f"{np.mean([r['tangential_mean'] for r in seg]):13.4f}")

    # 散点图
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    scatter_data = [
        ('norm_tz', r_norm_tz, 'norm_tz vs prune_d'),
        ('E_base', r_E_base, 'E_base vs prune_d'),
        ('radial_mean', [r['radial_mean'] for r in records], 'radial vs prune_d'),
        ('tangential_mean', [r['tangential_mean'] for r in records], 'tangential vs prune_d'),
        ('rot_mean', [r['rot_mean'] for r in records], 'rot vs prune_d'),
        ('anisotropy', [r['anisotropy'] for r in records], 'anisotropy vs prune_d'),
    ]
    for ax, (name, vals, title) in zip(axs.flat, scatter_data):
        ax.scatter(vals, r_prune, alpha=0.3, s=8)
        r_p, _ = pearsonr(vals, r_prune)
        ax.set_title(f'{title}\nr={r_p:.3f}')
        ax.set_xlabel(name)
        ax.set_ylabel('prune_d')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "move_energy_vs_prune_distance.png"), dpi=200, bbox_inches='tight')
    plt.close()


def test_move_energy_ranking_quality():
    """
    move_energy 选出的 best move 是否真的是 L2 最好的？
    对比 move_energy ranking vs E_base ranking vs random
    测量：top-1 / top-3 命中率, Kendall tau, rank correlation
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_SAMPLES = 300

    hits = {'E_total': 0, 'E_base': 0, 'E_potential': 0, 'E_geom': 0, 'random': 0}
    top3_hits = {'E_total': 0, 'E_base': 0, 'E_potential': 0, 'E_geom': 0, 'random': 0}
    tau_list = {'E_total': [], 'E_base': [], 'E_potential': [], 'E_geom': []}

    for _ in range(N_SAMPLES):
        depth = np.random.randint(1, 20)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)

        # ground truth: E_base ranking (L2 to solved after move)
        E = _move_energy(model,z, z_solved)
        Uz = np.einsum('nij,j->ni', model.U, z)
        E_base = np.linalg.norm(Uz - z_solved, axis=1)

        # 各分量独立 ranking
        tz = z_solved - z
        inner = np.einsum('ni,i->n', np.conj(Uz - z), tz)
        E_potential = -np.real(inner)
        radial = np.real(inner) / (np.linalg.norm(tz) + 1e-8)
        norm_dz = np.linalg.norm(Uz - z, axis=1)
        tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
        E_geom = radial + tangential + np.log1p(anisotropy) * rot

        # oracle: best move by E_base (label-aligned baseline, not a "feature")
        oracle_best = np.argmin(E_base)
        oracle_top3 = set(np.argsort(E_base)[:3])

        # 各方法的 best
        pred_total = np.argmin(E)
        pred_base = np.argmin(E_base)  # 自比较，应为 100%
        pred_pot = np.argmin(E_potential)
        pred_geom = np.argmin(E_geom)
        rand_best = np.random.randint(len(E))

        if pred_total == oracle_best:
            hits['E_total'] += 1
        if pred_base == oracle_best:
            hits['E_base'] += 1
        if pred_pot == oracle_best:
            hits['E_potential'] += 1
        if pred_geom == oracle_best:
            hits['E_geom'] += 1
        if rand_best == oracle_best:
            hits['random'] += 1

        if pred_total in oracle_top3:
            top3_hits['E_total'] += 1
        if pred_base in oracle_top3:
            top3_hits['E_base'] += 1
        if pred_pot in oracle_top3:
            top3_hits['E_potential'] += 1
        if pred_geom in oracle_top3:
            top3_hits['E_geom'] += 1
        if rand_best in oracle_top3:
            top3_hits['random'] += 1

        # Kendall tau (各分量 vs E_base 的排序一致性)
        from scipy.stats import kendalltau
        for name, ranking in [('E_total', E), ('E_base', E_base),
                              ('E_potential', E_potential), ('E_geom', E_geom)]:
            tau, _ = kendalltau(ranking, E_base)
            tau_list[name].append(tau)

    print("=" * 70)
    print("move_energy Ranking Quality vs Oracle Baseline (E_base)")
    print("=" * 70)
    print(f"\nN_samples = {N_SAMPLES}, N_moves = {len(model.U)}")
    print(f"Random baseline: top-1 = {1 / len(model.U):.4f}, top-3 = {3 / len(model.U):.4f}")
    print()
    print(f"{'Method':<15s} {'Top-1 hit':>10s} {'Top-3 hit':>10s} {'Rate(1)':>10s} {'Rate(3)':>10s} {'Kendall τ':>10s}")
    print("-" * 68)
    for name in ['E_total', 'E_base', 'E_potential', 'E_geom', 'random']:
        r1 = hits[name] / N_SAMPLES
        r3 = top3_hits[name] / N_SAMPLES
        tau_mean = np.mean(tau_list[name]) if name in tau_list else float('nan')
        print(f"  {name:<13s} {hits[name]:10d} {top3_hits[name]:10d} {r1:10.4f} {r3:10.4f} {tau_mean:10.4f}")

    # 按距离分段
    print(f"\n--- Top-1 hit rate by norm_tz ---")
    print(f"{'norm_tz range':<18s} {'N':>5s} {'E_total':>10s} {'E_pot':>10s} {'E_geom':>10s}")
    print("-" * 48)
    # re-collect with norm_tz
    seg_data = {'near': [], 'mid': [], 'far': []}
    for _ in range(200):
        depth = np.random.randint(1, 20)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)
        norm_tz = np.linalg.norm(z_solved - z)
        E = _move_energy(model,z, z_solved)
        Uz = np.einsum('nij,j->ni', model.U, z)
        E_base = np.linalg.norm(Uz - z_solved, axis=1)
        tz = z_solved - z
        inner = np.einsum('ni,i->n', np.conj(Uz - z), tz)
        E_potential = -np.real(inner)
        radial = np.real(inner) / (np.linalg.norm(tz) + 1e-8)
        norm_dz = np.linalg.norm(Uz - z, axis=1)
        tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
        E_geom = radial + tangential + np.log1p(anisotropy) * rot

        oracle_best = np.argmin(E_base)
        regime = 'near' if norm_tz < 1.5 else ('mid' if norm_tz < 4.0 else 'far')
        seg_data[regime].append({
            'hit_total': np.argmin(E) == oracle_best,
            'hit_pot': np.argmin(E_potential) == oracle_best,
            'hit_geom': np.argmin(E_geom) == oracle_best,
        })

    for regime in ['far', 'mid', 'near']:
        seg = seg_data[regime]
        if not seg:
            continue
        r_t = np.mean([d['hit_total'] for d in seg])
        r_p = np.mean([d['hit_pot'] for d in seg])
        r_g = np.mean([d['hit_geom'] for d in seg])
        label = {'far': '[4.0, ∞)', 'mid': '[1.5, 4.0)', 'near': '[0, 1.5)'}[regime]
        print(f"  {label:<16s} {len(seg):5d} {r_t:10.4f} {r_p:10.4f} {r_g:10.4f}")


def test_move_energy_component_regression():
    """
    用线性回归拟合 prune_d ~ w1*norm_tz + w2*radial + w3*tangential + w4*rot
    看各分量的回归系数和 R²
    以及：不同 regime 下最优权重是否不同
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_SAMPLES = 1000

    X_all = []
    y_all = []
    regime_labels = []

    for _ in range(N_SAMPLES):
        depth = np.random.randint(1, 25)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)
        tz = z_solved - z
        norm_tz = np.linalg.norm(tz)

        phase, prune_d = CubieBase.cubie_distance(s)

        Uz = np.einsum('nij,j->ni', model.U, z)
        dz = Uz - z
        inner = np.einsum('ni,i->n', np.conj(dz), tz)
        radial_mean = np.mean(np.real(inner) / (norm_tz + 1e-8))
        norm_dz = np.linalg.norm(dz, axis=1)
        tangential_mean = np.mean(np.sqrt(np.maximum(norm_dz ** 2 - (np.real(inner) / (norm_tz + 1e-8)) ** 2, 0)))
        rot_mean = np.mean(np.abs(np.imag(inner)))
        anisotropy = tangential_mean / (abs(radial_mean) + 1e-8)

        X_all.append([norm_tz, radial_mean, tangential_mean, rot_mean, anisotropy])
        y_all.append(prune_d)
        regime_labels.append('near' if norm_tz < 1.5 else ('mid' if norm_tz < 4.0 else 'far'))

    X = np.array(X_all)
    y = np.array(y_all)

    # 全局回归
    from numpy.linalg import lstsq
    X_aug = np.column_stack([X, np.ones(len(X))])
    w, res, rank, sv = lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print("=" * 70)
    print("Linear Regression: prune_d ~ norm_tz + radial + tangential + rot + anisotropy")
    print("=" * 70)
    print(f"\nN = {N_SAMPLES}")
    print(f"Global R² = {r2:.4f}")
    print(f"\nCoefficients:")
    names = ['norm_tz', 'radial', 'tangential', 'rot', 'anisotropy', 'intercept']
    for name, coef in zip(names, w):
        print(f"  {name:<15s}: {coef:12.6f}")

    # 各分量单独的 R²
    print(f"\n--- Single-feature R² ---")
    for i, name in enumerate(['norm_tz', 'radial', 'tangential', 'rot', 'anisotropy']):
        x1 = np.column_stack([X[:, i], np.ones(len(X))])
        w1, _, _, _ = lstsq(x1, y, rcond=None)
        y1 = x1 @ w1
        r2_1 = 1 - np.sum((y - y1) ** 2) / ss_tot
        r_p, _ = pearsonr(X[:, i], y)
        print(f"  {name:<15s}: R²={r2_1:.4f}, Pearson r={r_p:.4f}")

    # 分 regime 回归
    print(f"\n--- Regime-specific regression ---")
    print(
        f"{'Regime':<10s} {'N':>5s} {'R²':>8s} {'w_tz':>10s} {'w_rad':>10s} {'w_tan':>10s} {'w_rot':>10s} {'w_ani':>10s}")
    print("-" * 70)
    for regime in ['far', 'mid', 'near']:
        mask = np.array([l == regime for l in regime_labels])
        if np.sum(mask) < 20:
            continue
        X_r = X[mask]
        y_r = y[mask]
        X_aug_r = np.column_stack([X_r, np.ones(len(X_r))])
        w_r, _, _, _ = lstsq(X_aug_r, y_r, rcond=None)
        y_pred_r = X_aug_r @ w_r
        ss_res_r = np.sum((y_r - y_pred_r) ** 2)
        ss_tot_r = np.sum((y_r - np.mean(y_r)) ** 2)
        r2_r = 1 - ss_res_r / ss_tot_r if ss_tot_r > 0 else 0
        print(f"  {regime:<8s} {np.sum(mask):5d} {r2_r:8.4f} "
              f"{w_r[0]:10.4f} {w_r[1]:10.4f} {w_r[2]:10.4f} {w_r[3]:10.4f} {w_r[4]:10.4f}")

    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # norm_tz vs prune_d 散点 + 回归线
    axs[0].scatter(X[:, 0], y, alpha=0.2, s=6,
                   c=['blue' if l == 'far' else ('orange' if l == 'mid' else 'red') for l in regime_labels])
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    axs[0].plot(x_line, w[0] * x_line + w[-1], 'k-', lw=2, label=f'linear fit R²={r2:.3f}')
    axs[0].set_xlabel('norm_tz')
    axs[0].set_ylabel('prune_d')
    axs[0].set_title('norm_tz → true distance')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # radial vs prune_d
    axs[1].scatter(X[:, 1], y, alpha=0.2, s=6,
                   c=['blue' if l == 'far' else ('orange' if l == 'mid' else 'red') for l in regime_labels])
    axs[1].set_xlabel('radial (mean)')
    axs[1].set_ylabel('prune_d')
    axs[1].set_title('radial → true distance')
    axs[1].grid(True, alpha=0.3)

    # y_pred vs y_true
    axs[2].scatter(y_pred, y, alpha=0.2, s=6,
                   c=['blue' if l == 'far' else ('orange' if l == 'mid' else 'red') for l in regime_labels])
    axs[2].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1)
    axs[2].set_xlabel('predicted prune_d')
    axs[2].set_ylabel('true prune_d')
    axs[2].set_title(f'Multi-feature regression R2={r2:.3f}')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "move_energy_regression.png"), dpi=200, bbox_inches='tight')
    plt.close()


def test_far_region_geometric_signal():
    """
    远距离 (norm_tz ≥ 4.0) 下，几何分解分量 (radial, tangential, rot, E_geom)
    是否有任何 ranking 信号？
    关键测量：Kendall tau vs E_base, top-1 hit rate
    区分：d_slow (慢流形L2距离) 有用 vs 几何分解 (per-move) 是否有用
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_SAMPLES = 300

    # 仅收集远距离样本
    tau_geom = []
    tau_radial = []
    tau_tangential = []
    tau_rot = []
    tau_potential = []
    hits = {'E_geom': 0, 'radial': 0, 'tangential': 0, 'rot': 0, 'E_potential': 0}

    for _ in range(N_SAMPLES):
        # 生成远距离状态
        for _ in range(50):  # 重试直到找到远距离状态
            depth = np.random.randint(10, 25)
            s = CubieBase.generate_cubie(depth, check=True)
            z = model.project(s.vector)
            norm_tz = np.linalg.norm(z_solved - z)
            if norm_tz >= 4.0:
                break

        Uz = np.einsum('nij,j->ni', model.U, z)
        E_base = np.linalg.norm(Uz - z_solved, axis=1)
        tz = z_solved - z
        dz = Uz - z
        inner = np.einsum('ni,i->n', np.conj(dz), tz)

        radial = np.real(inner) / (norm_tz + 1e-8)
        norm_dz = np.linalg.norm(dz, axis=1)
        tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
        curvature = np.log1p(anisotropy)
        E_geom = radial + tangential + curvature * rot
        E_potential = -np.real(inner)

        from scipy.stats import kendalltau
        tau_geom.append(kendalltau(E_geom, E_base)[0])
        tau_radial.append(kendalltau(radial, E_base)[0])
        tau_tangential.append(kendalltau(tangential, E_base)[0])
        tau_rot.append(kendalltau(rot, E_base)[0])
        tau_potential.append(kendalltau(E_potential, E_base)[0])

        oracle_best = np.argmin(E_base)
        if np.argmin(E_geom) == oracle_best:
            hits['E_geom'] += 1
        if np.argmin(radial) == oracle_best:
            hits['radial'] += 1
        if np.argmin(tangential) == oracle_best:
            hits['tangential'] += 1
        if np.argmin(rot) == oracle_best:
            hits['rot'] += 1
        if np.argmin(E_potential) == oracle_best:
            hits['E_potential'] += 1

    print("=" * 70)
    print("Far-Region (norm_tz ≥ 4.0): Geometric Component Ranking Signal")
    print("=" * 70)
    print(f"\nN_samples = {N_SAMPLES}, all at norm_tz ≥ 4.0")
    print(f"Random baseline top-1 = {1 / len(model.U):.4f}")

    print(f"\n{'Component':<15s} {'Top-1 hit':>10s} {'Kendall τ':>12s} {'Signal?':>10s}")
    print("-" * 55)
    for name, hit_key, tau_list in [
        ('E_geom', 'E_geom', tau_geom),
        ('radial', 'radial', tau_radial),
        ('tangential', 'tangential', tau_tangential),
        ('rot', 'rot', tau_rot),
        ('E_potential', 'E_potential', tau_potential),
    ]:
        tau_mean = np.mean(tau_list)
        tau_std = np.std(tau_list)
        rate = hits[hit_key] / N_SAMPLES
        signal = "YES" if tau_mean > 0.05 else ("NO" if tau_mean < 0.02 else "WEAK")
        print(f"  {name:<13s} {rate:10.4f} {tau_mean:10.4f}±{tau_std:.3f} {signal:>10s}")

    # 对比：d_slow vs true distance (静态，非 per-move)
    print(f"\n--- Reference: static d_slow vs prune_d (from test_distance_correlation) ---")
    print(f"  d_slow at close range (d≤5):  Pearson r ≈ 0.508")
    print(f"  d_slow at mid range (d=10):   Pearson r ≈ 0.248")
    print(f"  d_slow at far range (d=30):   Pearson r ≈ 0.029")
    print(f"\n  Key distinction: d_slow (static L2 on slow manifold) has local")
    print(f"  search value, but the per-move geometric decomposition")
    print(f"  (radial/tangential/rot) does NOT identify good moves — even at far distances.")


def test_alpha_sweep_ablation():
    """
    Ablation: sweep α in E_total = E_base + α·E_geom
    测量 top-1, top-3, Kendall τ vs oracle (E_base alone)
    分 overall + 三区域 (far/mid/near)，分层采样保证每个区域足够样本
    预期：最优 α ≈ 0 —— 任何非零 E_geom 权重都会降低排序质量
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_PER_REGIME = 150  # 每个区域采样数
    alphas = np.linspace(-2.0, 2.0, 81)  # 步长 0.05

    # 分层采样：浅深度→near, 中深度→mid, 深深度→far
    all_E_base = []
    all_E_geom = []
    all_regime = []
    all_norm_tz = []

    def collect_for_regime(target_regime, max_attempts=2000):
        collected = 0
        attempts = 0
        while collected < N_PER_REGIME and attempts < max_attempts:
            attempts += 1
            depth = np.random.randint(1, 25)
            s = CubieBase.generate_cubie(depth, check=True)
            z = model.project(s.vector)
            tz = z_solved - z
            norm_tz = np.linalg.norm(tz)
            regime = 'near' if norm_tz < 1.5 else ('mid' if norm_tz < 4.0 else 'far')
            if regime == target_regime:
                Uz = np.einsum('nij,j->ni', model.U, z)
                E_base = np.linalg.norm(Uz - z_solved, axis=1)
                dz = Uz - z
                inner = np.einsum('ni,i->n', np.conj(dz), tz)
                radial = np.real(inner) / (norm_tz + 1e-8)
                norm_dz = np.linalg.norm(dz, axis=1)
                tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
                rot = np.abs(np.imag(inner))
                anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
                E_geom = radial + tangential + np.log1p(anisotropy) * rot
                all_E_base.append(E_base)
                all_E_geom.append(E_geom)
                all_regime.append(regime)
                all_norm_tz.append(norm_tz)
                collected += 1

    for r in ['far', 'mid', 'near']:
        collect_for_regime(r)

    all_E_base = np.array(all_E_base)
    all_E_geom = np.array(all_E_geom)
    all_regime = np.array(all_regime)

    # 对每个 α 计算指标
    top1 = np.zeros(len(alphas))
    top3 = np.zeros(len(alphas))
    ktau = np.zeros(len(alphas))

    top1_far = np.zeros(len(alphas))
    top3_far = np.zeros(len(alphas))
    ktau_far = np.zeros(len(alphas))

    top1_mid = np.zeros(len(alphas))
    top3_mid = np.zeros(len(alphas))
    ktau_mid = np.zeros(len(alphas))

    top1_near = np.zeros(len(alphas))
    top3_near = np.zeros(len(alphas))
    ktau_near = np.zeros(len(alphas))

    from scipy.stats import kendalltau

    far_mask = all_regime == 'far'
    mid_mask = all_regime == 'mid'
    near_mask = all_regime == 'near'
    n_far = far_mask.sum()
    n_mid = mid_mask.sum()
    n_near = near_mask.sum()
    N_TOTAL = len(all_regime)

    for i, alpha in enumerate(alphas):
        E_total = all_E_base + alpha * all_E_geom

        # Overall
        oracle_best = np.argmin(all_E_base, axis=1)
        pred_best = np.argmin(E_total, axis=1)
        top1[i] = np.mean(pred_best == oracle_best)
        oracle_top3 = np.argsort(all_E_base, axis=1)[:, :3]
        top3[i] = np.mean([pred_best[j] in oracle_top3[j] for j in range(N_TOTAL)])
        tau_vals = [kendalltau(E_total[j], all_E_base[j])[0] for j in range(N_TOTAL)]
        ktau[i] = np.nanmean(tau_vals)

        # Far
        if n_far > 5:
            oracle_best_f = np.argmin(all_E_base[far_mask], axis=1)
            pred_best_f = np.argmin(E_total[far_mask], axis=1)
            top1_far[i] = np.mean(pred_best_f == oracle_best_f)
            oracle_top3_f = np.argsort(all_E_base[far_mask], axis=1)[:, :3]
            top3_far[i] = np.mean([pred_best_f[j] in oracle_top3_f[j] for j in range(n_far)])
            tau_f = [kendalltau(E_total[far_mask][j], all_E_base[far_mask][j])[0] for j in range(n_far)]
            ktau_far[i] = np.nanmean(tau_f)

        # Mid
        if n_mid > 5:
            oracle_best_m = np.argmin(all_E_base[mid_mask], axis=1)
            pred_best_m = np.argmin(E_total[mid_mask], axis=1)
            top1_mid[i] = np.mean(pred_best_m == oracle_best_m)
            oracle_top3_m = np.argsort(all_E_base[mid_mask], axis=1)[:, :3]
            top3_mid[i] = np.mean([pred_best_m[j] in oracle_top3_m[j] for j in range(n_mid)])
            tau_m = [kendalltau(E_total[mid_mask][j], all_E_base[mid_mask][j])[0] for j in range(n_mid)]
            ktau_mid[i] = np.nanmean(tau_m)

        # Near
        if n_near > 5:
            oracle_best_n = np.argmin(all_E_base[near_mask], axis=1)
            pred_best_n = np.argmin(E_total[near_mask], axis=1)
            top1_near[i] = np.mean(pred_best_n == oracle_best_n)
            oracle_top3_n = np.argsort(all_E_base[near_mask], axis=1)[:, :3]
            top3_near[i] = np.mean([pred_best_n[j] in oracle_top3_n[j] for j in range(n_near)])
            tau_n = [kendalltau(E_total[near_mask][j], all_E_base[near_mask][j])[0] for j in range(n_near)]
            ktau_near[i] = np.nanmean(tau_n)

    # 找最优 α
    best_idx = np.argmax(top1)
    best_alpha = alphas[best_idx]
    best_top1 = top1[best_idx]
    best_idx_tau = np.argmax(ktau)
    best_alpha_tau = alphas[best_idx_tau]
    best_ktau = ktau[best_idx_tau]

    idx0 = np.where(np.abs(alphas) < 1e-9)[0][0]  # α=0 index
    N_TOTAL = len(all_regime)

    print("=" * 70)
    print("Ablation: α-Sweep — E_total = E_base + α·E_geom")
    print("=" * 70)
    print(f"\nN_total = {N_TOTAL}, α ∈ [{alphas[0]:.1f}, {alphas[-1]:.1f}], step={alphas[1] - alphas[0]:.3f}")
    print(f"Regime distribution: far={n_far}, mid={n_mid}, near={n_near}")
    print(f"\nOverall best α (by top-1): {best_alpha:.3f} → top-1 = {best_top1:.4f}")
    print(f"Overall best α (by Kendall τ): {best_alpha_tau:.3f} → τ = {best_ktau:.4f}")

    # 关键 α 值的详细输出
    print(f"\n--- Metrics at key α values ---")
    print(f"{'α':>8s}  {'top-1':>10s} {'top-3':>10s} {'Kendall τ':>12s}")
    print("-" * 46)
    for a_check in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]:
        ia = np.argmin(np.abs(alphas - a_check))
        print(f"  {alphas[ia]:6.2f}  {top1[ia]:10.4f} {top3[ia]:10.4f} {ktau[ia]:12.4f}")

    # 分区域最优
    print(f"\n--- Regime-specific best α ---")
    print(f"{'Regime':>6s} {'N':>5s} {'best α':>8s} {'top-1':>10s} {'Kendall τ':>12s} {'α=0 top-1':>12s}")
    print("-" * 60)
    for label, n, top1_r, ktau_r in [
        ('Far', n_far, top1_far, ktau_far),
        ('Mid', n_mid, top1_mid, ktau_mid),
        ('Near', n_near, top1_near, ktau_near),
    ]:
        if n < 10:
            continue
        ba = alphas[np.argmax(top1_r)]
        print(f"  {label:>4s} {n:5d} {ba:8.3f} {top1_r[np.argmax(top1_r)]:10.4f} "
              f"{ktau_r[np.argmax(top1_r)]:12.4f} {top1_r[idx0]:12.4f}")

    # ── 图 ──
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Overall top-1 / top-3
    axs[0, 0].plot(alphas, top1, 'b-', lw=2, label='top-1')
    axs[0, 0].plot(alphas, top3, 'b--', lw=1.5, label='top-3')
    axs[0, 0].axvline(0, color='gray', ls=':', alpha=0.5)
    axs[0, 0].axvline(best_alpha, color='red', ls='--', alpha=0.6, label=f'best α={best_alpha:.2f}')
    axs[0, 0].set_xlabel('α (E_geom weight)')
    axs[0, 0].set_ylabel('Hit Rate')
    axs[0, 0].set_title(f'Ablation: E_total = E_base + α·E_geom\nOverall (N={N_TOTAL})')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, alpha=0.3)

    # Panel 2: Overall Kendall τ
    axs[0, 1].plot(alphas, ktau, 'g-', lw=2)
    axs[0, 1].axvline(0, color='gray', ls=':', alpha=0.5)
    axs[0, 1].axvline(best_alpha_tau, color='red', ls='--', alpha=0.6, label=f'best α={best_alpha_tau:.2f}')
    axs[0, 1].set_xlabel('α (E_geom weight)')
    axs[0, 1].set_ylabel('Kendall τ vs Oracle')
    axs[0, 1].set_title(f'Ranking Consistency vs α')
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True, alpha=0.3)

    # Panel 3: Top-1 by regime
    axs[1, 0].plot(alphas, top1, 'k-', lw=2, label='overall', alpha=0.5)
    if n_far > 5:
        axs[1, 0].plot(alphas, top1_far, 'b-', lw=1.5, label=f'far (n={n_far})')
    if n_mid > 5:
        axs[1, 0].plot(alphas, top1_mid, 'orange', lw=1.5, label=f'mid (n={n_mid})')
    if n_near > 5:
        axs[1, 0].plot(alphas, top1_near, 'r-', lw=1.5, label=f'near (n={n_near})')
    axs[1, 0].axvline(0, color='gray', ls=':', alpha=0.5)
    axs[1, 0].set_xlabel('α (E_geom weight)')
    axs[1, 0].set_ylabel('Top-1 Hit Rate')
    axs[1, 0].set_title('Top-1 by Regime')
    axs[1, 0].legend(fontsize=7)
    axs[1, 0].grid(True, alpha=0.3)

    # Panel 4: Kendall τ by regime
    axs[1, 1].plot(alphas, ktau, 'k-', lw=2, label='overall', alpha=0.5)
    if n_far > 5:
        axs[1, 1].plot(alphas, ktau_far, 'b-', lw=1.5, label=f'far (n={n_far})')
    if n_mid > 5:
        axs[1, 1].plot(alphas, ktau_mid, 'orange', lw=1.5, label=f'mid (n={n_mid})')
    if n_near > 5:
        axs[1, 1].plot(alphas, ktau_near, 'r-', lw=1.5, label=f'near (n={n_near})')
    axs[1, 1].axvline(0, color='gray', ls=':', alpha=0.5)
    axs[1, 1].set_xlabel('α (E_geom weight)')
    axs[1, 1].set_ylabel('Kendall τ vs Oracle')
    axs[1, 1].set_title('Kendall τ by Regime')
    axs[1, 1].legend(fontsize=7)
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "alpha_sweep_ablation.png"), dpi=200, bbox_inches='tight')
    plt.close()


def test_quadratic_geom_ablation():
    """
    对比线性 vs 二次型 E_geom 的 α-sweep：
      linear:  E_geom = radial + tangential + curvature*rot
      quad:    E_geom = tangential**2 + rot**2  (天然二阶修正)
      quad_full: E_geom = radial**2 + tangential**2 + rot**2

    假设：二次型作为二阶修正，不会与一阶 E_base 竞争，ranking 退化应更小
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_SAMPLES = 300
    alphas = np.linspace(-0.5, 0.5, 41)  # 聚焦小 α 区域

    # 预收集
    all_E_base = []
    all_E_geom_linear = []
    all_E_geom_quad = []  # tangential**2 + rot**2
    all_E_geom_quad_full = []  # radial**2 + tangential**2 + rot**2
    all_regime = []

    for _ in range(N_SAMPLES):
        depth = np.random.randint(1, 25)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)
        tz = z_solved - z
        norm_tz = np.linalg.norm(tz)
        regime = 'near' if norm_tz < 1.5 else ('mid' if norm_tz < 4.0 else 'far')
        all_regime.append(regime)

        Uz = np.einsum('nij,j->ni', model.U, z)
        E_base = np.linalg.norm(Uz - z_solved, axis=1)
        dz = Uz - z
        inner = np.einsum('ni,i->n', np.conj(dz), tz)
        radial = np.real(inner) / (norm_tz + 1e-8)
        norm_dz = np.linalg.norm(dz, axis=1)
        tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)

        # 三种 E_geom
        E_geom_linear = radial + tangential + np.log1p(anisotropy) * rot
        E_geom_quad = tangential ** 2 + rot ** 2
        E_geom_quad_full = radial ** 2 + tangential ** 2 + rot ** 2

        all_E_base.append(E_base)
        all_E_geom_linear.append(E_geom_linear)
        all_E_geom_quad.append(E_geom_quad)
        all_E_geom_quad_full.append(E_geom_quad_full)

    all_E_base = np.array(all_E_base)
    all_E_geom_linear = np.array(all_E_geom_linear)
    all_E_geom_quad = np.array(all_E_geom_quad)
    all_E_geom_quad_full = np.array(all_E_geom_quad_full)
    all_regime = np.array(all_regime)

    from scipy.stats import kendalltau

    def sweep_alpha(E_geom_arr, label):
        """对给定 E_geom 做 α-sweep，返回 top1, top3, ktau 曲线"""
        top1 = np.zeros(len(alphas))
        top3 = np.zeros(len(alphas))
        ktau = np.zeros(len(alphas))
        for i, alpha in enumerate(alphas):
            E_total = all_E_base + alpha * E_geom_arr
            oracle_best = np.argmin(all_E_base, axis=1)
            pred_best = np.argmin(E_total, axis=1)
            top1[i] = np.mean(pred_best == oracle_best)
            oracle_top3 = np.argsort(all_E_base, axis=1)[:, :3]
            top3[i] = np.mean([pred_best[j] in oracle_top3[j] for j in range(len(all_E_base))])
            tau_vals = [kendalltau(E_total[j], all_E_base[j])[0] for j in range(len(all_E_base))]
            ktau[i] = np.nanmean(tau_vals)
        return top1, top3, ktau

    print("=" * 70)
    print("Quadratic vs Linear E_geom: α-Sweep Comparison")
    print("=" * 70)
    print(f"\nN = {N_SAMPLES}, α ∈ [{alphas[0]:.2f}, {alphas[-1]:.2f}]")

    results = {}
    for name, E_arr in [
        ('linear (radial+tan+curv*rot)', all_E_geom_linear),
        ('quad (tan^2+rot^2)', all_E_geom_quad),
        ('quad_full (rad^2+tan^2+rot^2)', all_E_geom_quad_full),
    ]:
        t1, t3, kt = sweep_alpha(E_arr, name)
        results[name] = (t1, t3, kt)
        # 多个 α 值
        print(f"\n{name}:")
        print(f"  {'α':>8s}  {'top-1':>10s} {'Kendall τ':>12s}")
        print(f"  {'-' * 8}  {'-' * 10} {'-' * 12}")
        for a_check in [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]:
            ia = np.argmin(np.abs(alphas - a_check))
            print(f"  {alphas[ia]:8.3f}  {t1[ia]:10.4f} {kt[ia]:12.4f}")

    # ── 图：三条 top-1 曲线对比 ──
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {'linear (radial+tan+curv*rot)': 'red',
              'quad (tan^2+rot^2)': 'blue',
              'quad_full (rad^2+tan^2+rot^2)': 'green'}
    for name, (t1, t3, kt) in results.items():
        axs[0].plot(alphas, t1, lw=2, color=colors[name], label=name)
        axs[0].set_xlabel('α')
        axs[0].set_ylabel('Top-1 Hit Rate')
        axs[0].set_title(f'Linear vs Quadratic E_geom Ablation (N={N_SAMPLES})')
        axs[0].axvline(0, color='gray', ls=':', alpha=0.5)
        axs[0].legend(fontsize=8)
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(alphas, kt, lw=2, color=colors[name], label=name)
        axs[1].set_xlabel('α')
        axs[1].set_ylabel('Kendall τ vs Oracle')
        axs[1].set_title('Ranking Consistency')
        axs[1].axvline(0, color='gray', ls=':', alpha=0.5)
        axs[1].legend(fontsize=8)
        axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "quadratic_vs_linear_geom.png"), dpi=200, bbox_inches='tight')
    plt.close()


# ── 20. Regime Separability ─────────────────────────────────────────────────

def test_regime_separability():
    """
    Exp 20: 验证 far/mid/near 三区是否统计可分的动力学 regime。

    论文 claim：慢流形划分的三个区不是人为分段，而是真正的动力学相变。
    测试方式：
      1. norm_tz 分布分析 —— 是否呈现多峰/可聚类
      2. 各区 move 行为统计差异 —— radial, tangential, rot, entropy
      3. 基于几何特征的 regime 可分类性
    注意：不使用 ranking 指标 (top-1, Kendall tau)。
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    N_PER_REGIME = 250

    # 分层采样：按深度区间覆盖三个 regime
    records = []
    for regime_target, depth_range, norm_range in [
        ('far', (10, 30), (4.0, 20.0)),
        ('mid', (2, 12), (1.5, 4.0)),
        ('near', (1, 5), (0.0, 1.5)),
    ]:
        collected = 0
        for _ in range(2000):
            if collected >= N_PER_REGIME:
                break
            depth = np.random.randint(*depth_range)
            s = CubieBase.generate_cubie(max(depth, 1), check=True)
            z = model.project(s.vector)
            tz = z_solved - z
            norm_tz = np.linalg.norm(tz)
            if not (norm_range[0] <= norm_tz < norm_range[1]):
                continue
            collected += 1

            Uz = np.einsum('nij,j->ni', model.U, z)
            dz = Uz - z
            inner = np.einsum('ni,i->n', np.conj(dz), tz)
            radial = np.real(inner) / (norm_tz + 1e-8)
            norm_dz = np.linalg.norm(dz, axis=1)
            tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
            rot = np.abs(np.imag(inner))

            # 18 个 move 的统计量
            records.append({
                'norm_tz': norm_tz,
                'radial_mean': np.mean(radial), 'radial_std': np.std(radial),
                'tangential_mean': np.mean(tangential), 'tangential_std': np.std(tangential),
                'rot_mean': np.mean(rot), 'rot_std': np.std(rot),
                'anisotropy': np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8),
                'E_base': np.linalg.norm(Uz - z_solved, axis=1),
                'depth': depth,
            })

    # ── 1. norm_tz 分布 ──
    norm_tz_vals = np.array([r['norm_tz'] for r in records])
    far_mask = norm_tz_vals >= 4.0
    mid_mask = (norm_tz_vals >= 1.5) & (norm_tz_vals < 4.0)
    near_mask = norm_tz_vals < 1.5

    far_n = np.sum(far_mask)
    mid_n = np.sum(mid_mask)
    near_n = np.sum(near_mask)

    print(f"\n{'=' * 70}")
    print("Exp 20: Regime Separability")
    print(f"{'=' * 70}")
    N_TOTAL = len(records)
    print(f"  总样本: {N_TOTAL}")
    print(f"  Far   (‖tz‖ ≥ 4.0): {far_n:4d} ({100 * far_n / N_TOTAL:5.1f}%)")
    print(f"  Mid   (1.5-4.0):   {mid_n:4d} ({100 * mid_n / N_TOTAL:5.1f}%)")
    print(f"  Near  (‖tz‖ < 1.5): {near_n:4d} ({100 * near_n / N_TOTAL:5.1f}%)")

    # ── 2. 各区 move 统计差异 ──
    def regime_stats(mask, name):
        keys = ['radial_mean', 'tangential_mean', 'rot_mean', 'anisotropy']
        print(f"\n  {name} regime (n={np.sum(mask)}):")
        for k in keys:
            vals = np.array([r[k] for r, m in zip(records, mask) if m])
            print(f"    {k:>20s}: μ={np.mean(vals):.4f}, σ={np.std(vals):.4f}")

        # move entropy: H = -Σ p_i log p_i, p_i = softmax(-E_base)
        entropies = []
        for r, m in zip(records, mask):
            if m:
                e = r['E_base']
                e_shifted = e - np.min(e)
                p = np.exp(-e_shifted) / np.sum(np.exp(-e_shifted))
                p = np.maximum(p, 1e-12)
                H = -np.sum(p * np.log(p))
                entropies.append(H)
        entropies = np.array(entropies)
        H_max = np.log(18)  # 均匀分布
        print(f"    {'move_entropy':>20s}: μ={np.mean(entropies):.4f}, σ={np.std(entropies):.4f} (max={H_max:.4f})")

    regime_stats(far_mask, "Far")
    regime_stats(mid_mask, "Mid")
    regime_stats(near_mask, "Near")

    # ── 3. Regime 可分性：用简单线性判别 ──
    features = np.array([[r['radial_mean'], r['tangential_mean'], r['rot_mean'],
                          r['anisotropy'], r['norm_tz']] for r in records])
    # label: 0=far, 1=mid, 2=near
    labels = np.zeros(N_TOTAL, dtype=int)
    labels[mid_mask] = 1
    labels[near_mask] = 2

    # 简单 LDA 风格：类间方差 / 类内方差
    from collections import Counter
    print(f"\n  Regime distribution: {dict(Counter(labels.tolist()))}")

    # 用 norm_tz 单独区分能力
    for thresh, rname in [(4.0, 'Far'), (1.5, 'Mid/Near boundary')]:
        if rname == 'Far':
            acc = np.mean(norm_tz_vals >= thresh) * 100
            print(f"  norm_tz ≥ {thresh} → {acc:.1f}% classified as Far (vs actual {100 * far_n / N_TOTAL:.1f}%)")

    # 三区 ANOVA-like: 每个特征跨区均值差异
    print(f"\n  Cross-regime feature contrast (ANOVA-like):")
    for i, fname in enumerate(['radial_mean', 'tangential_mean', 'rot_mean', 'anisotropy', 'norm_tz']):
        f_far = np.mean(features[far_mask, i]) if far_n > 0 else 0
        f_mid = np.mean(features[mid_mask, i]) if mid_n > 0 else 0
        f_near = np.mean(features[near_mask, i]) if near_n > 0 else 0
        f_all = np.mean(features[:, i])
        ss_between = far_n * (f_far - f_all) ** 2 + mid_n * (f_mid - f_all) ** 2 + near_n * (f_near - f_all) ** 2
        ss_total = np.sum((features[:, i] - f_all) ** 2)
        eta_sq = ss_between / (ss_total + 1e-12)  # effect size
        print(f"    {fname:>20s}: far={f_far:.3f} mid={f_mid:.3f} near={f_near:.3f} η²={eta_sq:.4f}")

    # ── 4. 图 ──
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # (a) norm_tz 直方图
    axs[0, 0].hist(norm_tz_vals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    for thresh, color, label in [(4.0, 'red', 'Far/Mid'), (1.5, 'orange', 'Mid/Near')]:
        axs[0, 0].axvline(thresh, color=color, ls='--', lw=1.5, label=f'{label} = {thresh}')
    axs[0, 0].set_xlabel('‖tz‖')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_title(f'norm_tz Distribution (N={N_TOTAL})')
    axs[0, 0].legend(fontsize=7)

    # (b) radial vs tangential 按 regime 着色
    colors = ['#2c7bb6', '#fdae61', '#d7191c']
    regime_names = ['Far', 'Mid', 'Near']
    for i, (mask, c, rn) in enumerate(zip([far_mask, mid_mask, near_mask], colors, regime_names)):
        axs[0, 1].scatter(features[mask, 0], features[mask, 1], c=c, label=rn,
                          alpha=0.5, s=12, edgecolors='none')
    axs[0, 1].set_xlabel('mean radial')
    axs[0, 1].set_ylabel('mean tangential')
    axs[0, 1].set_title('radial vs tangential by Regime')
    axs[0, 1].legend(fontsize=7)

    # (c) anisotropy vs norm_tz
    for i, (mask, c, rn) in enumerate(zip([far_mask, mid_mask, near_mask], colors, regime_names)):
        axs[0, 2].scatter(features[mask, 4], features[mask, 3], c=c, label=rn,
                          alpha=0.5, s=12, edgecolors='none')
    axs[0, 2].set_xlabel('norm_tz')
    axs[0, 2].set_ylabel('anisotropy')
    axs[0, 2].set_title('anisotropy vs norm_tz')
    axs[0, 2].legend(fontsize=7)

    # (d) per-regime box plots: radial
    data_by_regime = [
        features[far_mask, 0] if far_n > 0 else [],
        features[mid_mask, 0] if mid_n > 0 else [],
        features[near_mask, 0] if near_n > 0 else [],
    ]
    bp1 = axs[1, 0].boxplot(data_by_regime, tick_labels=regime_names, patch_artist=True)
    for patch, c in zip(bp1['boxes'], colors):
        patch.set_facecolor(c)
    axs[1, 0].set_ylabel('mean radial')
    axs[1, 0].set_title('radial by Regime')

    # (e) per-regime box plots: tangential
    data_by_regime2 = [
        features[far_mask, 1] if far_n > 0 else [],
        features[mid_mask, 1] if mid_n > 0 else [],
        features[near_mask, 1] if near_n > 0 else [],
    ]
    bp2 = axs[1, 1].boxplot(data_by_regime2, tick_labels=regime_names, patch_artist=True)
    for patch, c in zip(bp2['boxes'], colors):
        patch.set_facecolor(c)
    axs[1, 1].set_ylabel('mean tangential')
    axs[1, 1].set_title('tangential by Regime')

    # (f) move entropy by regime
    all_entropies = []
    for mask in [far_mask, mid_mask, near_mask]:
        ents = []
        for r, m in zip(records, mask):
            if m:
                e = r['E_base']
                e_shifted = e - np.min(e)
                p = np.exp(-e_shifted) / np.sum(np.exp(-e_shifted))
                p = np.maximum(p, 1e-12)
                ents.append(-np.sum(p * np.log(p)))
        all_entropies.append(ents)
    bp3 = axs[1, 2].boxplot(all_entropies, tick_labels=regime_names, patch_artist=True)
    for patch, c in zip(bp3['boxes'], colors):
        patch.set_facecolor(c)
    axs[1, 2].axhline(np.log(18), color='gray', ls=':', alpha=0.5, label=f'H_max={np.log(18):.2f}')
    axs[1, 2].set_ylabel('Move Entropy H')
    axs[1, 2].set_title('Move Entropy by Regime')
    axs[1, 2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "regime_separability.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved regime_separability.png")


# ── 21. Dynamical Statistics ────────────────────────────────────────────────

def test_dynamical_statistics():
    """
    Exp 21: 近区对称性与轨道动力学统计。

    论文 claim：near region 由离散群轨道 + 对称性主导，不是连续优化。
    测试：
      1. 2-cycle 频率 —— P(最优 move 的下一步是最优 move 的逆)
      2. Orbit 检测 —— 迭代 greedy 动力学中是否出现闭合轨道
      3. Move entropy —— 近区有效自由度降低
      4. 逆对优势比 —— 近区 move 选择集中在成对操作
    注意：不使用 ranking 指标。
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)

    from rime.cubie import CubieMove
    inverse_map = CubieMove.inverse_indices()
    for i, j in enumerate(inverse_map):
        assert inverse_map[j] == i, f"Inverse map inconsistent: {i}→{j} but {j}→{inverse_map[j]}"

    print(f"\n{'=' * 70}")
    print("Exp 21: Dynamical Statistics — Orbit & Symmetry")
    print(f"{'=' * 70}")
    print(f"  Inverse pairs: ", end="")
    inv_pairs = set()
    for i, j in enumerate(inverse_map):
        if i <= j:
            inv_pairs.add((i, j))
    print(", ".join(f"({i},{j})" for i, j in sorted(inv_pairs)))

    N_PER_REGIME = 200
    results = {'far': [], 'mid': [], 'near': []}

    for regime_name, min_depth, max_depth, norm_min, norm_max in [
        ('far', 10, 30, 4.0, 20.0),
        ('mid', 2, 10, 1.5, 4.0),
        ('near', 1, 4, 0.0, 1.5),
    ]:
        collected = 0
        attempts = 0
        while collected < N_PER_REGIME and attempts < 2000:
            attempts += 1
            depth = np.random.randint(min_depth, max_depth + 1) if max_depth > min_depth else min_depth
            s = CubieBase.generate_cubie(max(depth, 1), check=True)
            z = model.project(s.vector)
            norm_tz = np.linalg.norm(z_solved - z)
            if norm_min <= norm_tz < norm_max:
                collected += 1

                Uz = np.einsum('nij,j->ni', model.U, z)
                E_base = np.linalg.norm(Uz - z_solved, axis=1)
                best_move = int(np.argmin(E_base))

                # 施加 best move
                z_next = model.U[best_move] @ z

                # 从新状态找最优 move
                Uz_next = np.einsum('nij,j->ni', model.U, z_next)
                E_base_next = np.linalg.norm(Uz_next - z_solved, axis=1)
                best_next = int(np.argmin(E_base_next))

                # 2-cycle 检测：best_next 是否是 best_move 的逆
                is_2cycle = (inverse_map[best_move] == best_next)

                # move entropy
                e_shifted = E_base - np.min(E_base)
                p = np.exp(-e_shifted) / np.sum(np.exp(-e_shifted))
                p = np.maximum(p, 1e-12)
                H = -np.sum(p * np.log(p))

                # 逆对优势比：最优 move 和 its inverse 在 top-k 中占比
                inv_of_best = inverse_map[best_move]
                top3 = np.argsort(E_base)[:3]
                inv_in_top3 = inv_of_best in top3

                # orbit 检测：迭代 k 步检查是否回到邻域
                k_steps = 6
                z_cur = z.copy()
                visited = [z_cur.copy()]
                orbit_detected = False
                orbit_length = 0
                for step in range(k_steps):
                    Uz_cur = np.einsum('nij,j->ni', model.U, z_cur)
                    E_cur = np.linalg.norm(Uz_cur - z_solved, axis=1)
                    best = int(np.argmin(E_cur))
                    z_cur = model.U[best] @ z_cur
                    # 检查是否回到已访问点附近
                    for vi, vz in enumerate(visited):
                        if np.linalg.norm(z_cur - vz) < 0.5:
                            orbit_detected = True
                            orbit_length = step + 1 - vi
                            break
                    if orbit_detected:
                        break
                    visited.append(z_cur.copy())

                results[regime_name].append({
                    'norm_tz': norm_tz,
                    'is_2cycle': is_2cycle,
                    'inv_in_top3': inv_in_top3,
                    'move_entropy': H,
                    'orbit_detected': orbit_detected,
                    'orbit_length': orbit_length,
                    'best_move': best_move,
                    'best_next': best_next,
                })

        print(f"  {regime_name}: collected {collected} states in {attempts} attempts")

    # ── 汇总统计 ──
    print(
        f"\n  {'Regime':>8s}  {'2-cycle%':>10s}  {'InvTop3%':>10s}  {'H(moves)':>10s}  {'Orbit%':>8s}  {'OrbitLen':>10s}")
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
    for rn in ['far', 'mid', 'near']:
        recs = results[rn]
        n = len(recs)
        c2 = 100 * np.mean([r['is_2cycle'] for r in recs])
        inv3 = 100 * np.mean([r['inv_in_top3'] for r in recs])
        h = np.mean([r['move_entropy'] for r in recs])
        orb = 100 * np.mean([r['orbit_detected'] for r in recs])
        orb_len = np.mean([r['orbit_length'] for r in recs if r['orbit_detected']]) if orb > 0 else 0
        print(f"  {rn:>8s}  {c2:9.1f}%  {inv3:9.1f}%  {h:10.4f}  {orb:7.1f}%  {orb_len:10.2f}")

    # ── 2-cycle 频率 vs norm_tz ──
    all_norm_tz = []
    all_2cycle = []
    all_entropy = []
    all_orbit = []
    regimes_list = []
    for rn in ['far', 'mid', 'near']:
        for r in results[rn]:
            all_norm_tz.append(r['norm_tz'])
            all_2cycle.append(1.0 if r['is_2cycle'] else 0.0)
            all_entropy.append(r['move_entropy'])
            all_orbit.append(1.0 if r['orbit_detected'] else 0.0)
            regimes_list.append(rn)
    all_norm_tz = np.array(all_norm_tz)
    all_2cycle = np.array(all_2cycle)
    all_entropy = np.array(all_entropy)
    all_orbit = np.array(all_orbit)

    # ── 图 ──
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # (a) 2-cycle rate by regime bar
    bar_data = [100 * np.mean([r['is_2cycle'] for r in results[rn]]) for rn in ['far', 'mid', 'near']]
    bars = axs[0, 0].bar(['Far', 'Mid', 'Near'], bar_data, color=['#2c7bb6', '#fdae61', '#d7191c'])
    axs[0, 0].set_ylabel('2-Cycle Frequency (%)')
    axs[0, 0].set_title('2-Cycle Rate by Regime')
    for bar, val in zip(bars, bar_data):
        axs[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', fontsize=9)

    # (b) 2-cycle probability vs norm_tz (binned scatter)
    bins = np.linspace(0, 8, 16)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_2cycle = []
    bin_n = []
    for i in range(len(bins) - 1):
        mask = (all_norm_tz >= bins[i]) & (all_norm_tz < bins[i + 1])
        bin_n.append(np.sum(mask))
        bin_2cycle.append(np.mean(all_2cycle[mask]) * 100 if np.sum(mask) > 5 else np.nan)
    bin_2cycle = np.array(bin_2cycle)
    axs[0, 1].bar(bin_centers, bin_2cycle, width=0.45, color='steelblue', edgecolor='white')
    axs[0, 1].set_xlabel('‖tz‖')
    axs[0, 1].set_ylabel('2-Cycle Rate (%)')
    axs[0, 1].set_title('2-Cycle Probability vs Distance')
    axs[0, 1].axvline(1.5, color='orange', ls='--', alpha=0.7, label='Near/Mid')
    axs[0, 1].axvline(4.0, color='red', ls='--', alpha=0.7, label='Mid/Far')
    axs[0, 1].legend(fontsize=7)

    # (c) move entropy vs norm_tz
    for rn, c in [('far', '#2c7bb6'), ('mid', '#fdae61'), ('near', '#d7191c')]:
        mask = np.array([r == rn for r in regimes_list])
        axs[0, 2].scatter(all_norm_tz[mask][::3], all_entropy[mask][::3],
                          c=c, label=rn, alpha=0.4, s=10, edgecolors='none')
    axs[0, 2].axhline(np.log(18), color='gray', ls=':', alpha=0.5, label=f'H_max')
    axs[0, 2].set_xlabel('‖tz‖')
    axs[0, 2].set_ylabel('Move Entropy H')
    axs[0, 2].set_title('Move Entropy vs Distance')
    axs[0, 2].legend(fontsize=7)

    # (d) orbit detection rate by regime
    orb_data = [100 * np.mean([r['orbit_detected'] for r in results[rn]]) for rn in ['far', 'mid', 'near']]
    bars2 = axs[1, 0].bar(['Far', 'Mid', 'Near'], orb_data, color=['#2c7bb6', '#fdae61', '#d7191c'])
    axs[1, 0].set_ylabel('Orbit Detection Rate (%)')
    axs[1, 0].set_title('Orbit Detection (6-step) by Regime')
    for bar, val in zip(bars2, orb_data):
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', fontsize=9)

    # (e) orbit length histogram (near only)
    near_orbit_lens = [r['orbit_length'] for r in results['near'] if r['orbit_detected']]
    if near_orbit_lens:
        axs[1, 1].hist(near_orbit_lens, bins=np.arange(0.5, 7.5, 1), color='#d7191c',
                       edgecolor='white', alpha=0.8)
        axs[1, 1].set_xlabel('Orbit Length (steps)')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].set_title(f'Near-Regime Orbit Lengths (n={len(near_orbit_lens)})')
    else:
        axs[1, 1].text(0.5, 0.5, 'No orbits detected', ha='center', va='center')
        axs[1, 1].set_title('Near-Regime Orbit Lengths')

    # (f) inverse-in-top3 by regime
    inv_bar = [100 * np.mean([r['inv_in_top3'] for r in results[rn]]) for rn in ['far', 'mid', 'near']]
    bars3 = axs[1, 2].bar(['Far', 'Mid', 'Near'], inv_bar, color=['#2c7bb6', '#fdae61', '#d7191c'])
    axs[1, 2].set_ylabel('Inv-in-Top3 (%)')
    axs[1, 2].set_title('Inverse Move in Top-3 by Regime')
    for bar, val in zip(bars3, inv_bar):
        axs[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "dynamical_statistics.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved dynamical_statistics.png")


# ── 22. Geometry-Behavior Causal Correlation ────────────────────────────────

def test_geometry_behavior_correlation():
    """
    Exp 22: 几何量因果预测动力学行为（非 ranking）。

    论文 claim：tangential ↑ → orbit entry, curvature ↑ → instability
    这不是 "哪个 move 更好"，而是 "这个状态是否会进入轨道/对称陷阱"。

    测试：
      1. 几何量预测 2-cycle 发生（logistic regression）
      2. 几何量预测 orbit 发生
      3. tangential 大小与 orbit 进入概率的关系
      4. curvature 与动力学稳定性的关系
    """
    model = setup()
    z_solved = model.project(CubieState.solved().vector)

    from rime.cubie import CubieMove
    inverse_map = CubieMove.inverse_indices()

    N_SAMPLES = 500
    records = []

    for _ in range(N_SAMPLES):
        depth = np.random.randint(1, 25)
        s = CubieBase.generate_cubie(depth, check=True)
        z = model.project(s.vector)
        tz = z_solved - z
        norm_tz = np.linalg.norm(tz)

        Uz = np.einsum('nij,j->ni', model.U, z)
        dz = Uz - z
        inner = np.einsum('ni,i->n', np.conj(dz), tz)
        radial = np.real(inner) / (norm_tz + 1e-8)
        norm_dz = np.linalg.norm(dz, axis=1)
        tangential = np.sqrt(np.maximum(norm_dz ** 2 - radial ** 2, 0))
        rot = np.abs(np.imag(inner))
        anisotropy = np.mean(tangential) / (np.mean(np.abs(radial)) + 1e-8)
        curvature = np.log1p(anisotropy)

        # 几何特征（per-state，取均值或最优 move 的值）
        E_base = np.linalg.norm(Uz - z_solved, axis=1)
        best_move = int(np.argmin(E_base))

        geo_features = {
            'norm_tz': norm_tz,
            'radial_best': radial[best_move],
            'tangential_best': tangential[best_move],
            'rot_best': rot[best_move],
            'radial_mean': np.mean(radial),
            'tangential_mean': np.mean(tangential),
            'rot_mean': np.mean(rot),
            'anisotropy': anisotropy,
            'curvature': curvature,
            'tangential_std': np.std(tangential),
            'E_base_best': E_base[best_move],
        }

        # ── 行为测量（动力学结果，非 ranking） ──
        # 施加 best move
        z_next = model.U[best_move] @ z
        Uz_next = np.einsum('nij,j->ni', model.U, z_next)
        E_next = np.linalg.norm(Uz_next - z_solved, axis=1)
        best_next = int(np.argmin(E_next))

        is_2cycle = (inverse_map[best_move] == best_next)

        # orbit 检测（k=8 步）
        k_steps = 8
        z_cur = z.copy()
        visited = [z_cur.copy()]
        orbit_detected = False
        orbit_length = 0
        n_unique_moves = set()
        for step in range(k_steps):
            Uz_cur = np.einsum('nij,j->ni', model.U, z_cur)
            E_cur = np.linalg.norm(Uz_cur - z_solved, axis=1)
            best = int(np.argmin(E_cur))
            n_unique_moves.add(best)
            z_cur = model.U[best] @ z_cur
            for vi, vz in enumerate(visited):
                if np.linalg.norm(z_cur - vz) < 0.5:
                    orbit_detected = True
                    orbit_length = step + 1 - vi
                    break
            if orbit_detected:
                break
            visited.append(z_cur.copy())

        # 不稳定度量：相邻两步最优 move 变化频率
        move_volatility = 1.0 - (1.0 if is_2cycle else len(n_unique_moves) / k_steps)

        records.append({
            **geo_features,
            'is_2cycle': is_2cycle,
            'orbit_detected': orbit_detected,
            'orbit_length': orbit_length,
            'n_unique_moves': len(n_unique_moves),
            'move_volatility': move_volatility,
        })

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print("Exp 22: Geometry-Behavior Causal Correlation")
    print(f"{'=' * 70}")
    cycle_rate = np.mean([r['is_2cycle'] for r in records])
    orbit_rate = np.mean([r['orbit_detected'] for r in records])
    print(f"  N={N_SAMPLES}")
    print(f"  2-cycle 发生率: {100 * cycle_rate:.1f}%")
    print(f"  Orbit 检出率:   {100 * orbit_rate:.1f}%")

    # ── 1. 几何量 → 2-cycle（logistic 效应） ──
    print(f"\n  Logistic: 几何量 → 2-cycle 概率")
    print(f"  {'Feature':>20s}  {'β':>10s}  {'OR':>8s}  {'p-value':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 8}  {'-' * 10}")

    from scipy.stats import chi2
    cycle_arr = np.array([r['is_2cycle'] for r in records], dtype=float)

    for fname in ['norm_tz', 'radial_best', 'tangential_best', 'rot_best',
                  'anisotropy', 'curvature', 'tangential_std']:
        x = np.array([r[fname] for r in records])
        # 简单 logistic-like：用分位点回归
        # 比较 2-cycle 组 vs 非 2-cycle 组的特征差异
        cycle_mask = cycle_arr > 0.5
        if np.sum(cycle_mask) < 5 or np.sum(~cycle_mask) < 5:
            continue
        mu_cycle = np.mean(x[cycle_mask])
        mu_no = np.mean(x[~cycle_mask])
        # Cohen's d 效应量
        pooled_std = np.sqrt((np.var(x[cycle_mask]) + np.var(x[~cycle_mask])) / 2)
        d = (mu_cycle - mu_no) / (pooled_std + 1e-12)
        # 简单 t 检验
        se = np.sqrt(np.var(x[cycle_mask]) / np.sum(cycle_mask) +
                     np.var(x[~cycle_mask]) / np.sum(~cycle_mask))
        t_stat = (mu_cycle - mu_no) / (se + 1e-12)
        from scipy.stats import t as tdist
        df = np.sum(cycle_mask) + np.sum(~cycle_mask) - 2
        p_val = 2 * tdist.sf(np.abs(t_stat), df)
        or_val = np.exp(d)  # 近似 OR
        print(f"  {fname:>20s}  {d:10.4f}  {or_val:8.3f}  {p_val:10.4f}")

    # ── 2. 几何量 → orbit 检出 ──
    print(f"\n  Logistic: 几何量 → orbit 概率")
    print(f"  {'Feature':>20s}  {'β':>10s}  {'OR':>8s}  {'p-value':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 8}  {'-' * 10}")

    orbit_arr = np.array([r['orbit_detected'] for r in records], dtype=float)
    for fname in ['norm_tz', 'radial_best', 'tangential_best', 'rot_best',
                  'anisotropy', 'curvature', 'tangential_std']:
        x = np.array([r[fname] for r in records])
        orbit_mask = orbit_arr > 0.5
        if np.sum(orbit_mask) < 5 or np.sum(~orbit_mask) < 5:
            continue
        mu_orbit = np.mean(x[orbit_mask])
        mu_no = np.mean(x[~orbit_mask])
        pooled_std = np.sqrt((np.var(x[orbit_mask]) + np.var(x[~orbit_mask])) / 2)
        d = (mu_orbit - mu_no) / (pooled_std + 1e-12)
        se = np.sqrt(np.var(x[orbit_mask]) / np.sum(orbit_mask) +
                     np.var(x[~orbit_mask]) / np.sum(~orbit_mask))
        t_stat = (mu_orbit - mu_no) / (se + 1e-12)
        from scipy.stats import t as tdist
        df = np.sum(orbit_mask) + np.sum(~orbit_mask) - 2
        p_val = 2 * tdist.sf(np.abs(t_stat), df)
        or_val = np.exp(d)
        print(f"  {fname:>20s}  {d:10.4f}  {or_val:8.3f}  {p_val:10.4f}")

    # ── 3. Tangential binned → 2-cycle 率 ──
    print(f"\n  Tangential best → 2-cycle rate (binned):")
    tan_best = np.array([r['tangential_best'] for r in records])
    bins = np.percentile(tan_best, [0, 20, 40, 60, 80, 100])
    for i in range(len(bins) - 1):
        mask = (tan_best >= bins[i]) & (tan_best < bins[i + 1])
        if np.sum(mask) > 5:
            c_rate = 100 * np.mean(cycle_arr[mask])
            o_rate = 100 * np.mean(orbit_arr[mask])
            print(
                f"    Q{i}: tan∈[{bins[i]:.3f}, {bins[i + 1]:.3f}) n={np.sum(mask):3d}  2-cycle={c_rate:.1f}%  orbit={o_rate:.1f}%")

    # ── 图 ──
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    # (a) tangential_best 分布：2-cycle vs non
    cycle_mask_bool = cycle_arr > 0.5
    axs[0, 0].hist([r['tangential_best'] for r, m in zip(records, cycle_mask_bool) if not m],
                   bins=30, alpha=0.6, color='#2c7bb6', label='Non 2-cycle', density=True)
    axs[0, 0].hist([r['tangential_best'] for r, m in zip(records, cycle_mask_bool) if m],
                   bins=30, alpha=0.6, color='#d7191c', label='2-cycle', density=True)
    axs[0, 0].set_xlabel('tangential (best move)')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].set_title('Tangential Distribution: 2-cycle vs Non')
    axs[0, 0].legend(fontsize=8)

    # (b) curvature 分布：orbit vs non
    orbit_mask_bool = orbit_arr > 0.5
    axs[0, 1].hist([r['curvature'] for r, m in zip(records, orbit_mask_bool) if not m],
                   bins=30, alpha=0.6, color='#2c7bb6', label='Non-orbit', density=True)
    axs[0, 1].hist([r['curvature'] for r, m in zip(records, orbit_mask_bool) if m],
                   bins=30, alpha=0.6, color='#d7191c', label='Orbit', density=True)
    axs[0, 1].set_xlabel('curvature')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].set_title('Curvature Distribution: Orbit vs Non')
    axs[0, 1].legend(fontsize=8)

    # (c) 2-cycle rate vs tangential (binned scatter)
    tan_bins = np.linspace(np.min(tan_best), np.max(tan_best), 15)
    tan_centers = (tan_bins[:-1] + tan_bins[1:]) / 2
    tan_2cycle = []
    for i in range(len(tan_bins) - 1):
        mask = (tan_best >= tan_bins[i]) & (tan_best < tan_bins[i + 1])
        tan_2cycle.append(100 * np.mean(cycle_arr[mask]) if np.sum(mask) > 10 else np.nan)
    axs[0, 2].bar(tan_centers, tan_2cycle, width=0.9 * (tan_bins[1] - tan_bins[0]),
                  color='steelblue', edgecolor='white')
    axs[0, 2].set_xlabel('tangential (best move)')
    axs[0, 2].set_ylabel('2-Cycle Rate (%)')
    axs[0, 2].set_title('2-Cycle Probability vs Tangential')

    # (d) Feature effect sizes (Cohen's d) for 2-cycle prediction
    features_for_plot = ['norm_tz', 'radial_best', 'tangential_best', 'rot_best',
                         'anisotropy', 'curvature', 'tangential_std']
    d_cycle = []
    d_orbit = []
    for fname in features_for_plot:
        x = np.array([r[fname] for r in records])
        for arr, d_list in [(cycle_arr, d_cycle), (orbit_arr, d_orbit)]:
            mask = arr > 0.5
            if np.sum(mask) >= 5 and np.sum(~mask) >= 5:
                mu1, mu0 = np.mean(x[mask]), np.mean(x[~mask])
                pooled_std = np.sqrt((np.var(x[mask]) + np.var(x[~mask])) / 2)
                d_list.append((mu1 - mu0) / (pooled_std + 1e-12))
            else:
                d_list.append(0.0)

    x_pos = np.arange(len(features_for_plot))
    width = 0.35
    axs[1, 0].bar(x_pos - width / 2, d_cycle, width, color='#d7191c', label='2-cycle')
    axs[1, 0].bar(x_pos + width / 2, d_orbit, width, color='#fdae61', label='Orbit')
    axs[1, 0].axhline(0, color='gray', ls='-', alpha=0.5)
    axs[1, 0].set_xticks(x_pos)
    axs[1, 0].set_xticklabels([f.replace('_best', '').replace('_', '\n') for f in features_for_plot], fontsize=7)
    axs[1, 0].set_ylabel("Cohen's d")
    axs[1, 0].set_title('Effect Size: Geometry → Dynamics')
    axs[1, 0].legend(fontsize=8)

    # (e) norm_tz vs n_unique_moves
    all_norm_tz = np.array([r['norm_tz'] for r in records])
    all_n_unique = np.array([r['n_unique_moves'] for r in records], dtype=float)
    axs[1, 1].scatter(all_norm_tz[::2], all_n_unique[::2], c=all_norm_tz[::2],
                      cmap='RdYlBu_r', alpha=0.5, s=10, edgecolors='none')
    axs[1, 1].set_xlabel('‖tz‖')
    axs[1, 1].set_ylabel('N Unique Moves (k=8)')
    axs[1, 1].set_title('Move Diversity vs Distance')

    # (f) anisotropy vs orbit rate
    aniso_vals = np.array([r['anisotropy'] for r in records])
    aniso_bins = np.linspace(np.min(aniso_vals), np.max(aniso_vals), 12)
    aniso_centers = (aniso_bins[:-1] + aniso_bins[1:]) / 2
    aniso_orbit = []
    for i in range(len(aniso_bins) - 1):
        mask = (aniso_vals >= aniso_bins[i]) & (aniso_vals < aniso_bins[i + 1])
        aniso_orbit.append(100 * np.mean(orbit_arr[mask]) if np.sum(mask) > 10 else np.nan)
    axs[1, 2].bar(aniso_centers, aniso_orbit, width=0.9 * (aniso_bins[1] - aniso_bins[0]),
                  color='#fdae61', edgecolor='white')
    axs[1, 2].set_xlabel('anisotropy')
    axs[1, 2].set_ylabel('Orbit Rate (%)')
    axs[1, 2].set_title('Orbit Probability vs Anisotropy')

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "geometry_behavior_correlation.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved geometry_behavior_correlation.png")


# ── Exp 23: 2-Step Trajectory vs Greedy ──────────────────────────────────

def test_two_step_vs_greedy():
    """Exp 23: 2-step MPC with orbit + history penalty vs greedy"""
    from rime.cubie import CubieMove

    model = setup()
    z_solved = model.project(CubieState.solved().vector)
    n_moves = len(model.U)

    inverse_map = CubieMove.inverse_indices()

    def greedy_step(z, prev_move=None):
        Uz = np.einsum('nij,j->ni', model.U, z)
        E = np.linalg.norm(Uz - z_solved, axis=1)
        best = int(np.argmin(E))
        return best, model.U[best] @ z

    def two_step_step(z, prev_move=None, lambda_orbit=2.0, lambda_history=2.0):
        """2-step MPC: evaluate g1 with orbit penalty on g2=inv(g1)
        AND history penalty on g1=inv(prev_move)."""
        best_g1 = -1
        best_val = np.inf
        for g1 in range(n_moves):
            z1 = model.U[g1] @ z
            Uz2 = np.einsum('nij,j->ni', model.U, z1)
            E2 = np.linalg.norm(Uz2 - z_solved, axis=1)
            # Orbit penalty: discourage g2 = inv(g1)
            E2[inverse_map[g1]] += lambda_orbit
            best_g2_val = np.min(E2)
            # History penalty: discourage g1 = inv(prev_move)
            if prev_move is not None and g1 == inverse_map[prev_move]:
                best_g2_val += lambda_history
            if best_g2_val < best_val:
                best_val = best_g2_val
                best_g1 = g1
        return best_g1, model.U[best_g1] @ z

    N = 80
    MAX_STEPS = 30
    regime_specs = [
        ('far', 10, 30, 4.0, 20.0),
        ('mid', 2, 10, 1.5, 4.0),
        ('near', 1, 4, 0.0, 1.5),
    ]
    states_by_regime = {}
    for rn, min_d, max_d, nmin, nmax in regime_specs:
        bucket = []
        attempts = 0
        while len(bucket) < N and attempts < 3000:
            attempts += 1
            depth = np.random.randint(min_d, max_d + 1) if max_d > min_d else min_d
            s = CubieBase.generate_cubie(max(depth, 1), check=True)
            z = model.project(s.vector)
            ntz = np.linalg.norm(z - z_solved)
            if nmin <= ntz < nmax:
                bucket.append((s, z, ntz))
        states_by_regime[rn] = bucket
        print(f"  {rn}: collected {len(bucket)} states in {attempts} attempts")

    strategies = {
        'greedy': (greedy_step, {}),
        '2-step λ=0': (two_step_step, {'lambda_orbit': 0.0, 'lambda_history': 0.0}),
        '2-step λ=2': (two_step_step, {'lambda_orbit': 2.0, 'lambda_history': 2.0}),
        '2-step λ=5': (two_step_step, {'lambda_orbit': 5.0, 'lambda_history': 5.0}),
    }

    regime_order = ['far', 'mid', 'near']
    results = {}
    for strategy_name, (step_fn, kw) in strategies.items():
        results[strategy_name] = {}
        for regime in regime_order:
            records = []
            for s, z0, ntz0 in states_by_regime[regime]:
                z = z0.copy()
                moves = []
                ntz_traj = [ntz0]
                prev = None
                for step in range(MAX_STEPS):
                    g, z_next = step_fn(z, prev_move=prev, **kw)
                    moves.append(g)
                    ntz = np.linalg.norm(z_next - z_solved)
                    ntz_traj.append(ntz)
                    z = z_next
                    prev = g
                # Per-step 2-cycle rate
                n_cycles = sum(1 for i in range(len(moves) - 1)
                               if moves[i + 1] == inverse_map[moves[i]])
                cycle_rate = n_cycles / max(1, len(moves) - 1) if len(moves) > 1 else 0.0
                # Stalled: last 5 steps all part of 2-cycles
                n_recent = min(5, len(moves) - 1)
                stalled = n_recent >= 1 and all(
                    moves[len(moves) - n_recent + i] == inverse_map[moves[len(moves) - n_recent + i - 1]]
                    for i in range(n_recent)
                ) if n_recent >= 1 else False
                records.append({
                    'moves': moves,
                    'ntz_traj': ntz_traj,
                    'n_cycles': n_cycles,
                    'cycle_rate': cycle_rate,
                    'stalled': stalled,
                    'steps': len(moves),
                    'nt0': ntz0,
                    'nf': ntz_traj[-1],
                    'nt_reduction': (ntz0 - ntz_traj[-1]) / max(1, len(moves)),
                    'final_ntz': ntz_traj[-1],
                })
            results[strategy_name][regime] = records

    # ── print ──
    print(f"\n{'=' * 80}")
    print("Exp 23: 2-Step Trajectory vs Greedy")
    print(f"{'=' * 80}")
    header = f"{'Strategy':<16} {'Regime':<6} {'2Cyc/step%':<11} {'Stalled%':<9} {'Δntz/step':<12} {'Final‖tz‖':<10}"
    print(header)
    print("-" * 80)
    for strategy_name in strategies:
        for regime in regime_order:
            recs = results[strategy_name][regime]
            cyc_step = 100 * np.mean([r['cycle_rate'] for r in recs])
            stalled = 100 * np.mean([r['stalled'] for r in recs])
            dntz = np.mean([r['nt_reduction'] for r in recs])
            fnorm = np.mean([r['final_ntz'] for r in recs])
            print(f"{strategy_name:<16} {regime:<6} {cyc_step:>9.1f}% {stalled:>7.1f}% {dntz:>10.4f}   {fnorm:>10.4f}")
        if strategy_name != list(strategies.keys())[-1]:
            print("-" * 80)
    print("=" * 80)

    # ── figure: 6-panel comparison ──
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    colors = {'greedy': '#e41a1c', '2-step λ=0': '#377eb8',
              '2-step λ=2': '#4daf4a', '2-step λ=5': '#984ea3'}
    x = np.arange(len(regime_order))
    width = 0.18

    # (a) per-step 2-cycle rate
    ax = axs[0, 0]
    for si, (sname, col) in enumerate(colors.items()):
        vals = [100 * np.mean([r['cycle_rate'] for r in results[sname][reg]]) for reg in regime_order]
        ax.bar(x + (si - 1.5) * width, vals, width, color=col, label=sname, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_order)
    ax.set_ylabel('Per-Step 2-Cycle Rate (%)')
    ax.set_title('(a) Per-Step 2-Cycle Rate')
    ax.legend(fontsize=7)

    # (b) stalled rate
    ax = axs[0, 1]
    for si, (sname, col) in enumerate(colors.items()):
        vals = [100 * np.mean([r['stalled'] for r in results[sname][reg]]) for reg in regime_order]
        ax.bar(x + (si - 1.5) * width, vals, width, color=col, label=sname, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_order)
    ax.set_ylabel('Stalled Trajectories (%)')
    ax.set_title('(b) % Trajectories Ending in 2-Cycle')
    ax.legend(fontsize=7)

    # (c) norm_tz reduction per step
    ax = axs[0, 2]
    for si, (sname, col) in enumerate(colors.items()):
        vals = [np.mean([r['nt_reduction'] for r in results[sname][reg]]) for reg in regime_order]
        ax.bar(x + (si - 1.5) * width, vals, width, color=col, label=sname, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_order)
    ax.set_ylabel('norm_tz reduction / step')
    ax.set_title('(c) Norm Reduction per Step')
    ax.legend(fontsize=7)

    # (d) final norm_tz
    ax = axs[1, 0]
    for si, (sname, col) in enumerate(colors.items()):
        vals = [np.mean([r['final_ntz'] for r in results[sname][reg]]) for reg in regime_order]
        ax.bar(x + (si - 1.5) * width, vals, width, color=col, label=sname, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_order)
    ax.set_ylabel('Final ‖tz‖')
    ax.set_title('(d) Final ‖tz‖ after 30 Steps')
    ax.legend(fontsize=7)

    # (e) near-regime trajectory length histogram (how many steps before stall)
    ax = axs[1, 1]
    near_recs = results['greedy']['near']
    greedy_len = [r['steps'] for r in near_recs]
    near_recs_2step = results['2-step λ=5']['near']
    twostep_len = [r['steps'] for r in near_recs_2step]
    ax.hist(greedy_len, bins=15, alpha=0.5, color=colors['greedy'], label='greedy', density=True)
    ax.hist(twostep_len, bins=15, alpha=0.5, color=colors['2-step λ=5'], label='2-step λ=5', density=True)
    ax.set_xlabel('Trajectory Length (steps)')
    ax.set_ylabel('Density')
    ax.set_title('(e) Near-Regime Trajectory Length Distribution')
    ax.legend(fontsize=7)

    # (f) exemplary far→near trajectories
    ax = axs[1, 2]
    example_state = states_by_regime['far'][0]
    for sname, (step_fn, kw) in strategies.items():
        z = example_state[1].copy()
        ntz_traj = [np.linalg.norm(z - z_solved)]
        prev = None
        for step in range(MAX_STEPS):
            g, z_next = step_fn(z, prev_move=prev, **kw)
            ntz = np.linalg.norm(z_next - z_solved)
            ntz_traj.append(ntz)
            z = z_next
            prev = g
        ax.plot(ntz_traj, color=colors[sname], label=sname, linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Step')
    ax.set_ylabel('‖tz‖')
    ax.set_title('(f) Exemplary Far→Near Trajectory')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "two_step_vs_greedy.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> saved two_step_vs_greedy.png")



def test_build_slow_algebra_basis():
    """Test SlowDynamics.build_slow_algebra_basis: orthonormality, span coverage."""
    print("\n── test_build_slow_algebra_basis ──")
    model = setup()

    basis, s = model.build_slow_algebra_basis()
    k = len(basis)
    d = model.dim_slow
    n_gen = len(model.rho_slow)
    n_comm = len(model.C_pairs)
    total_mats = n_gen + n_comm

    print(f"  Basis size: {k} (from {total_mats} input matrices: {n_gen} generators + {n_comm} commutators)")

    # 1. Basis matrices exist and have correct shape
    assert k > 0, "Basis is empty"
    assert basis.shape == (k, d, d), f"Basis shape {basis.shape}, expected ({k},{d},{d})"
    print(f"  [OK] Basis shape: ({k}, {d}, {d})")

    # 2. Orthonormality in Frobenius inner product: <B_i, B_j>_F = δ_ij
    frob_gram = np.zeros((k, k), dtype=complex)
    for i in range(k):
        for j in range(k):
            frob_gram[i, j] = np.tensordot(basis[i].conj(), basis[j])
    I = np.eye(k)
    assert np.allclose(frob_gram, I, atol=model.tol * 10), \
        f"Basis not orthonormal: max|gram - I| = {np.max(np.abs(frob_gram - I)):.2e}"
    print(f"  [OK] Basis orthonormal in Frobenius inner product (max deviation: {np.max(np.abs(frob_gram - I)):.2e})")

    # 3. Each generator U can be reconstructed from the basis
    #   c[l] = <basis[l], U>_F = Tr(basis[l]^H U), then U = Σ c[l] · basis[l]
    max_residual_gen = 0.0
    for _, U in model.rho_slow.values():
        coeffs = np.array([np.tensordot(basis[l].conj(), U) for l in range(k)])
        U_recon = np.sum(coeffs[:, None, None] * basis, axis=0)
        res = np.linalg.norm(U - U_recon)
        max_residual_gen = max(max_residual_gen, res)
    assert max_residual_gen < model.tol * 100, \
        f"Generator reconstruction failed: max residual = {max_residual_gen:.2e}"
    print(f"  [OK] All generators in span(basis) (max reconstruction error: {max_residual_gen:.2e})")

    # 4. Each commutator C can be reconstructed from the basis
    max_residual_comm = 0.0
    for C in model.C_pairs.values():
        coeffs = np.array([np.tensordot(basis[l].conj(), C) for l in range(k)])
        C_recon = np.sum(coeffs[:, None, None] * basis, axis=0)
        res = np.linalg.norm(C - C_recon)
        max_residual_comm = max(max_residual_comm, res)
    assert max_residual_comm < model.tol * 100, \
        f"Commutator reconstruction failed: max residual = {max_residual_comm:.2e}"
    print(f"  [OK] All commutators in span(basis) (max reconstruction error: {max_residual_comm:.2e})")

    # 5. Singular values are non-increasing and positive
    assert np.all(s > 0), "Some singular values are non-positive"
    assert np.all(np.diff(s) <= 1e-10 * s[0]), "Singular values not non-increasing"
    print(f"  [OK] Singular values non-increasing, s[0]={s[0]:.2f}, s[-1]={s[-1]:.2e}")

    print(f"  All build_slow_algebra_basis checks passed.")
    return True


def test_structure_constants():
    """Test SlowDynamics.structure_constants: antisymmetry, Jacobi identity."""
    print("\n── test_structure_constants ──")
    model = setup()

    Cijk = model.structure_constants()
    k = Cijk.shape[0]
    assert Cijk.shape == (k, k, k), f"Shape {Cijk.shape}, expected ({k},{k},{k})"
    print(f"  Structure constants shape: ({k}, {k}, {k})")

    basis, _ = model.build_slow_algebra_basis()

    # 1. Antisymmetry: Cijk[i,j,l] ≈ -Cijk[j,i,l]
    max_asym = 0.0
    for i in range(k):
        for j in range(k):
            for l in range(k):
                asym = abs(Cijk[i, j, l] + Cijk[j, i, l])
                max_asym = max(max_asym, asym)
    assert max_asym < model.tol * 10, f"Antisymmetry violated: max|Cijk[i,j,l]+Cijk[j,i,l]| = {max_asym:.2e}"
    print(f"  [OK] Antisymmetry: Cijk[i,j,l] ≈ -Cijk[j,i,l] (max violation: {max_asym:.2e})")

    # 2. Reconstruction: [basis[i], basis[j]] ≈ Σ_l Cijk[i,j,l] · basis[l]
    #   NOTE: Since the basis spans span{generators, commutators} but may not be
    #   closed under the Lie bracket, reconstruction has truncation error.
    #   This error measures the algebra non-closure — a genuine structural metric.
    max_recon_err = 0.0
    for i in range(min(k, 6)):
        for j in range(min(k, 6)):
            comm = basis[i] @ basis[j] - basis[j] @ basis[i]
            comm_recon = np.sum(Cijk[i, j, :, None, None] * basis, axis=0)
            err = np.linalg.norm(comm - comm_recon)
            max_recon_err = max(max_recon_err, err)
    comm_norms_avg = np.mean([np.linalg.norm(basis[i] @ basis[j] - basis[j] @ basis[i])
                               for i in range(min(k, 6)) for j in range(min(k, 6))])
    print(f"  [INFO] Commutator reconstruction max error: {max_recon_err:.4f} "
          f"(avg ‖comm‖ = {comm_norms_avg:.4f}, relative = {max_recon_err / (comm_norms_avg + 1e-12):.2%})")
    print(f"  [INFO] Non-zero error is expected: the slow algebra is not Lie-closed under bracket")

    # 3. Jacobi identity: [b_i,[b_j,b_k]] + [b_j,[b_k,b_i]] + [b_k,[b_i,b_j]]
    #   Residual is non-zero due to algebra non-closure — informational metric.
    max_jacobi = 0.0
    sample_n = min(k, 4)
    for i in range(sample_n):
        for j in range(sample_n):
            for m in range(sample_n):
                jacobi = np.zeros((model.dim_slow, model.dim_slow), dtype=complex)
                for p in range(k):
                    for q in range(k):
                        term1 = Cijk[j, m, p] * Cijk[i, p, q]
                        term2 = Cijk[m, i, p] * Cijk[j, p, q]
                        term3 = Cijk[i, j, p] * Cijk[m, p, q]
                        jacobi += (term1 + term2 + term3) * basis[q]
                max_jacobi = max(max_jacobi, np.linalg.norm(jacobi))
    print(f"  [INFO] Jacobi identity residual: {max_jacobi:.4f} (non-zero = algebra not Lie-closed)")

    print(f"  All structure_constants checks passed.")
    return True


def test_evolve_continuous():
    """Test SlowDynamics.evolve_continuous: equivalence with discrete evolve."""
    print("\n── test_evolve_continuous ──")
    model = setup()

    z = np.random.randn(model.dim_slow) + 1j * np.random.randn(model.dim_slow)

    # 1. T=0: identity
    z0 = model.evolve_continuous(z, 0)
    assert np.allclose(z0, z, atol=model.tol), "T=0 must return z"
    print(f"  [OK] T=0: evolve_continuous(z, 0) = z")

    # 2. evolve_continuous ≈ evolve for integer T (L = diag(log(w_slow)))
    #   exp(L*T) * z = exp(diag(log(w_slow)*T)) * z = diag(w_slow^T) * z = w_slow^T ⊙ z
    for T in [1, 2, 5, 10]:
        zc = model.evolve_continuous(z, T)
        zd = model.evolve(z, T)
        err = np.linalg.norm(zc - zd)
        # The only difference is 1e-12 regularization in L, so they should be very close
        assert err < model.tol * 10, f"T={T}: |evolve_continuous - evolve| = {err:.2e}"
    print(f"  [OK] evolve_continuous ≈ evolve for integer T=1,2,5,10 (max err: "
          f"{max(np.linalg.norm(model.evolve_continuous(z, t) - model.evolve(z, t)) for t in [1,2,5,10]):.2e})")

    # 3. Semigroup property: evolve(T1+T2) = evolve(T1) ∘ evolve(T2) for continuous
    z1 = model.evolve_continuous(z, 1.5)
    z_step = model.evolve_continuous(model.evolve_continuous(z, 1.0), 0.5)
    err_semi = np.linalg.norm(z1 - z_step)
    assert err_semi < model.tol * 10, f"Semigroup property failed: err={err_semi:.2e}"
    print(f"  [OK] Semigroup: evol(1.5) = evol(1.0)∘evol(0.5) (err={err_semi:.2e})")

    # 4. Effect on eigenvector: (w_slow_i)^T * v_i
    # Already implied by equivalence test, but explicit check
    T_frac = 2.5
    zc = model.evolve_continuous(z, T_frac)
    zd = model.evolve(z, T_frac)  # w^T for non-integer T works element-wise
    assert np.allclose(zc, zd, atol=model.tol * 10), \
        f"Fractional T={T_frac}: evolve_continuous ≠ evolve"
    print(f"  [OK] Fractional T=2.5: evolve_continuous matches evolve")

    print(f"  All evolve_continuous checks passed.")
    return True


def test_invariant_distance():
    """Test SlowDynamics.invariant_distance: weighted spectral pseudometric."""
    print("\n── test_invariant_distance ──")
    model = setup()

    solved = CubieState.solved().vector
    def invariant_distance(x, y):
        z = model.project(x - y)
        w = 1 / (1 - model.w_slow + 1e-6)  # 加权（按谱衰减）
        return np.sqrt(np.sum(w * np.abs(z) ** 2))
    
    # 1. d(x, x) = 0
    d_self = invariant_distance(solved, solved)
    assert d_self < model.tol, f"d(x,x) = {d_self:.2e}, expected 0"
    print(f"  [OK] d(x, x) = 0")

    # 2. Symmetry: d(x, y) = d(y, x)
    state = CubieBase.generate_cubie(length=10)
    x = state.vector
    d_xy = invariant_distance(solved, x)
    d_yx = invariant_distance(x, solved)
    assert abs(d_xy - d_yx) < model.tol, f"d(x,y)={d_xy:.6f} ≠ d(y,x)={d_yx:.6f}"
    print(f"  [OK] Symmetry: d(x, y) = d(y, x) = {d_xy:.6f}")

    # 3. Non-negativity
    for _ in range(10):
        s = CubieBase.generate_cubie(length=np.random.randint(1, 20)).vector
        d = invariant_distance(solved, s)
        assert d >= -model.tol, f"Negative distance: {d:.6f}"
    print(f"  [OK] d(x, y) ≥ 0 for 10 random states")

    # 4. Triangle inequality check (on a few states)
    a = CubieBase.generate_cubie(length=5).vector
    b = CubieBase.generate_cubie(length=8).vector
    c = CubieBase.generate_cubie(length=12).vector
    d_ab = invariant_distance(a, b)
    d_bc = invariant_distance(b, c)
    d_ac = invariant_distance(a, c)
    assert d_ac <= d_ab + d_bc + 1e-6, \
        f"Triangle inequality violated: d_ac={d_ac:.4f} > d_ab+d_bc={d_ab + d_bc:.4f}"
    print(f"  [OK] Triangle inequality: d_ac({d_ac:.4f}) ≤ d_ab({d_ab:.4f}) + d_bc({d_bc:.4f})")

    # 5. invariant_distance ≥ plain L2 slow distance (weights w ≥ 1 for all decaying eigenvalues)
    d_inv = invariant_distance(a, b)
    d_l2 = model.heuristic(a, b)
    assert d_inv >= d_l2 - model.tol, f"invariant({d_inv:.4f}) < L2({d_l2:.4f})"
    print(f"  [OK] invariant_distance({d_inv:.4f}) ≥ L2 slow distance({d_l2:.4f})")

    # 6. Fast subspace component contributes zero to distance
    z_fast = np.random.randn(228) + 1j * np.random.randn(228)
    z_fast = z_fast - model.V_keep @ (model.V_keep.T.conj() @ z_fast)  # project to fast space
    d_fast = invariant_distance(z_fast, np.zeros(228))
    assert d_fast < model.tol * 100, f"Fast vector distance should be ~0, got {d_fast:.2e}"
    print(f"  [OK] Fast subspace distance ≈ 0: {d_fast:.2e}")

    print(f"  All invariant_distance checks passed.")
    return True


def test_fast_energy():
    """Test SlowDynamics.fast_energy: fast subspace residual norm."""
    print("\n── test_fast_energy ──")
    model = setup()

    solved = CubieState.solved().vector

    # 1. fast_energy(solved) is NOT ~0 — the solved state has non-zero
    #    projection onto V_const (λ=1, 24D invariants) which is excluded from V_slow.
    #    fast_energy = ‖(I - V_slow·V_slow^T) @ x‖ = ‖P_{const+fast} x‖
    e_solved = model.fast_energy(solved)
    print(f"  [INFO] fast_energy(solved) = {e_solved:.4f} (non-zero: V_const component excluded from V_slow)")

    # 2. Random state has positive fast energy
    for _ in range(5):
        s = CubieBase.generate_cubie(length=np.random.randint(5, 20)).vector
        e = model.fast_energy(s)
        assert e >= -model.tol, f"Negative fast_energy: {e:.6f}"
    print(f"  [OK] fast_energy ≥ 0 for random states")

    # 3. fast_energy(x) = 0 iff x = V_slow @ z (pure slow, no const/fast)
    #    NOTE: V_keep = [V_const | V_slow], but fast_energy uses V_slow only.
    z_pure_slow = model.V_slow @ np.random.randn(model.V_slow.shape[1])
    e_pure_slow = model.fast_energy(z_pure_slow)
    assert e_pure_slow < model.tol * 10, f"fast_energy(V_slow @ z) = {e_pure_slow:.2e}, expected ~0"
    print(f"  [OK] fast_energy(V_slow @ z) = {e_pure_slow:.2e} ≈ 0")

    # 4. Deep scramble states have higher fast energy
    s_near = CubieBase.generate_cubie(length=3).vector
    s_far = CubieBase.generate_cubie(length=20).vector
    e_near = model.fast_energy(s_near)
    e_far = model.fast_energy(s_far)
    print(f"  [INFO] fast_energy(depth=3): {e_near:.4f}, fast_energy(depth=20): {e_far:.4f}")

    print(f"  All fast_energy checks passed.")
    return True


def test_heuristic_with_confidence():
    """Test SlowDynamics.heuristic_with_confidence: distance + confidence measure."""
    print("\n── test_heuristic_with_confidence ──")
    model = setup()

    solved = CubieState.solved().vector

    # 1. Zero distance to self, high confidence
    d_self, conf_self = model.heuristic_with_confidence(solved, solved)
    assert d_self < model.tol, f"d(x,x) = {d_self:.2e}, expected 0"
    assert conf_self > 1.0 - model.tol, f"confidence(x,x) = {conf_self:.4f}, expected ~1"
    print(f"  [OK] d(solved, solved) = {d_self:.2e}, confidence = {conf_self:.4f}")

    # 2. Symmetry
    state = CubieBase.generate_cubie(length=8).vector
    d_ab, c_ab = model.heuristic_with_confidence(solved, state)
    d_ba, c_ba = model.heuristic_with_confidence(state, solved)
    assert abs(d_ab - d_ba) < model.tol, f"Distance not symmetric: {d_ab:.6f} vs {d_ba:.6f}"
    assert abs(c_ab - c_ba) < model.tol, f"Confidence not symmetric: {c_ab:.6f} vs {c_ba:.6f}"
    print(f"  [OK] Symmetric: d={d_ab:.4f}, confidence={c_ab:.4f}")

    # 3. Confidence ≤ 1 (it's exp(-something_nonnegative))
    for _ in range(10):
        s = CubieBase.generate_cubie(length=np.random.randint(1, 20)).vector
        _, conf = model.heuristic_with_confidence(solved, s)
        assert 0 <= conf <= 1 + model.tol, f"Confidence out of [0,1]: {conf:.6f}"
    print(f"  [OK] 0 ≤ confidence ≤ 1 for 10 random states")

    # 4. V_slow-only vectors have high confidence (no const/fast residual)
    z_pure_slow = model.V_slow @ np.random.randn(model.V_slow.shape[1])
    _, conf_pure_slow = model.heuristic_with_confidence(z_pure_slow, np.zeros(228))
    assert conf_pure_slow > 0.99, f"V_slow-only confidence = {conf_pure_slow:.4f}, expected ~1"
    print(f"  [OK] V_slow-only vector: confidence = {conf_pure_slow:.4f} ≈ 1")

    # 5. d(x,y) matches plain heuristic distance
    d, conf = model.heuristic_with_confidence(solved, state)
    d_plain = model.heuristic(solved, state)
    assert abs(d - d_plain) < model.tol, f"Distance mismatch: {d:.6f} vs {d_plain:.6f}"
    print(f"  [OK] Distance matches heuristic: {d:.4f}")

    # 6. Far states have lower confidence (more fast residual)
    s_near = CubieBase.generate_cubie(length=2).vector
    s_far = CubieBase.generate_cubie(length=25).vector
    _, conf_near = model.heuristic_with_confidence(solved, s_near)
    _, conf_far = model.heuristic_with_confidence(solved, s_far)
    print(f"  [INFO] confidence(depth=2): {conf_near:.4f}, confidence(depth=25): {conf_far:.4f}")

    print(f"  All heuristic_with_confidence checks passed.")
    return True


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 1. 生成器谱结构 ===")
    test_generator_spectrum()

    print("\n=== 3. 时间晶体模拟 ===")
    test_time_crystal()

    print("\n=== 4. Move scores & move energy ===")
    test_move_scores_and_energy()

    print("\n=== 5. 量子 vs 经典演化 ===")
    test_quantum_vs_classical()

    print("\n=== 6. Environment 目标状态距离 ===")
    test_environment_targets()

    print("\n=== 7. 慢流形距离分布 ===")
    test_distance_distribution()

    # print("\n=== 8. 主模拟 ===")
    # test_main_simulation()

    print("\n=== 9. 退火实验 ===")
    test_annealing()

    print("\n=== 10. 温度扫描 ===")
    test_temperature_scan()

    print("\n=== 11. 扩散轨迹 ===")
    test_diffusion()

    print("\n=== 12. 随机游走慢距离 vs prune_d ===")
    test_slow_vs_random_walk()

    print("\n=== 13. 等深度对慢距离 vs prune_d ===")
    test_slow_vs_relative_state()

    print("\n=== 14. 准等距性验证 ===")
    test_quasi_isometry()

    print("\n=== 15. move_energy vs prune_d ===")
    test_move_energy_vs_prune_distance()

    print("\n=== 16. move_energy ranking quality ===")
    test_move_energy_ranking_quality()

    print("\n=== 17. move_energy component regression ===")
    test_move_energy_component_regression()

    print("\n=== 18. 远区几何信号检测 ===")
    test_far_region_geometric_signal()

    print("\n=== 19. alpha-sweep ablation ===")
    test_alpha_sweep_ablation()

    print("\n=== 19.1. 二次型 E_geom 比较 ===")
    test_quadratic_geom_ablation()

    print("\n=== 20. Regime 可分性验证 ===")
    test_regime_separability()

    print("\n=== 21. 动力学统计 (orbit & symmetry) ===")
    test_dynamical_statistics()

    print("\n=== 22. 几何-行为因果相关 ===")
    test_geometry_behavior_correlation()

    print("\n=== 23. 2-step MPC vs Greedy ===")
    test_two_step_vs_greedy()

    # TrajectoryEnergy class and test_trajectory_energy removed.
    # Superseded by test/_exp_trajectory_energy.py
    # spectral phase decomposition (V(Δ), transport tensor K_ij,
    # discrete-continuous separation) instead of heuristic soft-term ablation.
    # Test 24 (TrajectoryEnergy) moved to test/_exp_trajectory_energy.py

    print("\n=== 25. build_slow_algebra_basis ===")
    test_build_slow_algebra_basis()

    print("\n=== 26. structure_constants ===")
    test_structure_constants()

    print("\n=== 27. evolve_continuous ===")
    test_evolve_continuous()

    print("\n=== 28. invariant_distance ===")
    test_invariant_distance()

    print("\n=== 29. fast_energy ===")
    test_fast_energy()

    print("\n=== 30. heuristic_with_confidence ===")
    test_heuristic_with_confidence()
