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
  12. 慢投影 vs 剪枝距离相关性
  13. 准等距性验证

运行: python test/test_cubieworld.py
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rime.cubieworld import SlowDynamics, Environment, HybridSimulation,N_GENERATORS
from rime.cubie import CubieState, CubieMove, CubieBase
from rime.helpers import cosine_distance
from rime.cubieoperator import poly_rank
from rime.base import DATA_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

N_PAIRS = 5000


def setup():
    model = SlowDynamics(n=N_GENERATORS)
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
    V_modes = model.V[:, mode_indices]
    w_modes = model.w[mode_indices]

    state0 = CubieBase.generate_cubie(length=5)
    z0 = model.project(state0.vector)

    steps = 300
    trajectory = [z0]
    for t in range(steps):
        z_next = z0 * (w_modes ** t)
        trajectory.append(z_next)
    trajectory = np.array(trajectory)

    proj = trajectory @ V_modes

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
    for n in [18, 16, 12, 10, 9, 8, 6, 4, 3, 2]:
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


# ── 3. 交换子谱分析 ─────────────────────────────────────────────────────

def test_commutator_spectrum():
    """18 个基本 move 两两组合（含交换子）的 rho 分析"""
    prim_list = CubieMove.prim_moves.copy()
    products2 = CubieBase.generate_compose_moves(prim_list, commutator=True)
    print(f"18 两两组合后去重 + 去 identity + commutator 数量: {len(products2)}")
    """6*6*6 = 3^3 x 2^3"""

    comm_rho = {k: (m, m.rho()) for k, m in products2.items()}
    """
    comm_slow = SlowDynamics(n=len(comm_rho), rho_moves=comm_rho)
    Lambda 0.222222: multiplicity 8
    Lambda 0.296296: multiplicity 24
    Lambda 0.407407: multiplicity 24
    Lambda 0.527778: multiplicity 8
    Lambda 0.703704: multiplicity 72
    Lambda 0.722222: multiplicity 24
    Lambda 0.777778: multiplicity 36
    Lambda 0.814815: multiplicity 8
    Lambda 0.925926: multiplicity 4
    Lambda 1.000000: multiplicity 20
    Fast layer spectral radius: 0.527778, Estimated mixing time (eps=1e-6): steps -> Tf=22
    144
    """


# ── 4. Move scores & move energy ────────────────────────────────────────

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
        score = model.move_scores(z1, z0, preference)
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

        energy = model.move_energy(z2, z0)
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

    # greedy search 演化
    z = model.project(CubieBase.generate_cubie(10).vector)
    for t in range(10):
        energy = model.move_energy(z, z0)
        best = np.argmin(energy)
        z = model.U[best] @ z
        V = np.linalg.norm(z - z0) ** 2
        print(V, energy.min(), energy.max())

    print('-' * 50)
    s1 = CubieBase.generate_cubie(10)
    x2 = model.greedy_search_slow(s1, CubieState.solved(), 30, min_dist=1)
    print(x2[0], x2[1], x2[2])


# ── 5. 量子 vs 经典演化 ────────────────────────────────────────────────

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

    s2 =CubieExample.big_cycle()
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
    plt.savefig(os.path.join(DATA_DIR, "Annealing Trajectory with Temperature Overla.png"), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(DATA_DIR, "Annealing Trajectories with Different Starting Temperatures.png"), dpi=300, bbox_inches='tight')
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

def test_distance_correlation():
    """慢投影距离与剪枝表真实距离的 Pearson/Spearman 相关"""
    model = setup()
    n_pairs = N_PAIRS

    # 随机步数采样
    d1_list, d2_list = [], []
    stateA = CubieState.solved()
    for i in range(n_pairs):
        if i % 500 == 0:
            print(f"已处理 {i}/{n_pairs} 对...")
        steps = random.randint(1, 30)
        g = CubieBase.random_walk(length=steps)
        stateB = g.act(stateA)
        phase, d1 = CubieBase.cubie_distance(stateB)
        d2 = model.heuristic(stateA.vector, stateB.vector, False)
        d1_list.append(d1)
        d2_list.append(d2)

    d1_arr = np.array(d1_list)
    d2_arr = np.array(d2_list)
    pearson_corr, pearson_p = pearsonr(np.log(d1_arr + 1), d2_arr)
    spearman_corr, spearman_p = spearmanr(d1_arr, d2_arr)
    print(f"\n全深度相关系数:")
    print(f"Pearson r = {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_corr:.4f} (p={spearman_p:.2e})")
    print("std d1", np.std(d1_arr), "std d2", np.std(d2_arr))

    plt.figure(figsize=(12, 8))
    plt.scatter(np.log(d1_arr + 1), d2_arr, alpha=0.6, s=10, c='blue', edgecolor='none')
    plt.xlabel("prune heuristic d1 log")
    plt.ylabel("slow d2 = ||V_slow^T (rho(A) - rho(B))||")
    plt.title(f"slow vs real distance (n={n_pairs})")
    plt.grid(True, alpha=0.3)
    plt.text(0.05, 0.95, f"Pearson r = {pearson_corr:.4f}\nSpearman r = {spearman_corr:.4f}",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "slow_distance_vs_real_distance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 分深度采样
    for dh in (5, 10, 20, 30):
        d1_list, d2_list = [], []
        for i in range(n_pairs):
            if i % 500 == 0:
                print(f"  depth={dh} 已处理 {i}/{n_pairs} 对...")
            stateA = CubieBase.generate_cubie(dh)
            stateB = CubieBase.generate_cubie(dh)
            stateC = CubieMove.relative_state(stateA, stateB)
            phase, d1 = CubieBase.cubie_distance(stateC)
            d2 = model.heuristic(stateA.vector, stateB.vector, False)
            d1_list.append(d1)
            d2_list.append(d2)

        d1_arr = np.array(d1_list)
        d2_arr = np.array(d2_list)
        pearson_corr, pearson_p = pearsonr(np.log(d1_arr + 1), d2_arr)
        spearman_corr, spearman_p = spearmanr(d1_arr, d2_arr)
        print(f"\ndepth={dh}: Pearson r = {pearson_corr:.4f}, Spearman r = {spearman_corr:.4f}")
        print("  std d1", np.std(d1_arr), "std d2", np.std(d2_arr))

        plt.figure(figsize=(12, 8))
        plt.scatter(np.log(d1_arr + 1), d2_arr, alpha=0.6, s=10, c='blue', edgecolor='none')
        plt.xlabel("prune heuristic d1 log")
        plt.ylabel("slow d2")
        plt.title(f"slow vs real distance (d={dh}, n={n_pairs})")
        plt.grid(True, alpha=0.3)
        plt.text(0.05, 0.95, f"Pearson r = {pearson_corr:.4f}\nSpearman r = {spearman_corr:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, f"slow_distance_vs_real_d{dh}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    """
    dh 随机0-30：
    Pearson r = 0.5081, Spearman r = 0.3636
    slow manifold 捕捉到了宏观难度

    depth=10: Pearson r = 0.2480, Spearman r = 0.1743
    depth=20: Pearson r = 0.0622, Spearman r = 0.0541
    depth=30: Pearson r = 0.0291, Spearman r = 0.0270

    Rubik 群的随机游走在大约 15-20 moves 后就会接近混合状态。
    slow manifold 对"远距离状态"区分能力下降
    slow spectral embedding ~ 局部搜索结构
    10 步以内影响巨大,小深度区域：state space 非常稀疏
    """


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
    d_ratios = []
    for _ in range(3000):
        A, B = CubieBase.generate_cubie_pair()
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


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 1. 时间晶体模拟 ===")
    test_time_crystal()

    print("\n=== 2. 生成器谱结构 ===")
    test_generator_spectrum()

    print("\n=== 3. 交换子谱分析 ===")
    test_commutator_spectrum()

    print("\n=== 4. Move scores & move energy ===")
    test_move_scores_and_energy()

    print("\n=== 5. 量子 vs 经典演化 ===")
    test_quantum_vs_classical()

    print("\n=== 6. Environment 目标状态距离 ===")
    test_environment_targets()

    print("\n=== 7. 慢流形距离分布 ===")
    test_distance_distribution()

    print("\n=== 8. 主模拟 ===")
    test_main_simulation()

    print("\n=== 9. 退火实验 ===")
    test_annealing()

    print("\n=== 10. 温度扫描 ===")
    test_temperature_scan()

    print("\n=== 11. 扩散轨迹 ===")
    test_diffusion()

    print("\n=== 12. 慢投影 vs 剪枝距离相关性 ===")
    test_distance_correlation()

    print("\n=== 13. 准等距性验证 ===")
    test_quasi_isometry()
