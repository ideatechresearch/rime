"""Hybrid agent ecosystem: QuantumAgent, HybridAgent, and multi-agent simulations.

Extracted from cubieworld.py (2026-05-18). These are pre-trilogy exploratory
classes — not grounded in the A → K_αβ → κ_d theoretical framework.
Kept for future consideration, not imported by any core module.
"""
import numpy as np
import random
import os
from collections import deque

import matplotlib.pyplot as plt

from rime.base import DATA_DIR
from rime.cubie import CubieState, CubieMove, CubieBase, CubieExample, N_GENERATORS
from rime.cubieworld import SlowDynamics
from rime.helpers import (dbscan, fidelity, softmax, sigmoid, normalize_z, cosine_distance, von_neumann_entropy)


def move_energy(model, z, target, prev_dz=None):
    """几何能量函数：势能 + 几何修正（radial / tangential / curvature）
        energy 函数尚未“大一统”——远区引导强，近区被 metric + 对称性压制，2-cycle 仍频繁出现。

        E = <dz, ∇V> + <dz, G(z) dz>
        （Fredholm / 紧算子视角）

        ================================
        1. 动力学分区（Three Regimes）
        ================================

        Far Regime（远区）
            ||tz|| large
            • radial 主导，potential 强
            • move ranking 有意义（≈ L2 正确排序）
            • energy 表现稳定

        Mixing Regime（混合区）
            • radial 与 tangential 竞争
            • anisotropy 上升，curvature 开始起作用
            • 欧式距离区分能力下降（进入 shell overlap）

        Near-Target Regime（近目标区）【核心困难】
            ||tz|| → small
            • radial → 0
            • potential → flat
            • tangential / orbit motion 主导
            • metric 容易压倒 potential
            • move_scores 区分度崩塌
            • 出现大量：
                - 2-cycle（g ↔ g⁻¹）
                - dir==2 对称陷阱
                - orbit / cycle 行为
            引入“破坏细致平衡”的项

        ================================
        2. 群对称性主导现象（Level-3）
        ================================

        • 2-cycle orbit（最稳定吸引子）
            - g ↔ g⁻¹ 振荡
            - move 分布集中在成对操作
            - 原因：投影后 U ≈ U⁻¹（方向信息丢失）
            manifestation:
            - paired move dominance
            - undo/redo loops

        • dir==2 / 180° 对称陷阱
            - radial ↓, tangential ↓, rot ≈ 0
            - energy 在这些方向近似平坦
            - 状态在对称子空间振荡（UD-like subspace）

        • Involution-like behavior
            - 多个 move 在 slow manifold 上近似互逆
            - greedy 策略 → undo / redo 循环

        • 对称子流形（Symmetric Submanifold）
            - 局部近似 Abel（低交换子）
            - 有效自由度降低
            - 非交换结构被投影压平

        ================================
        3. Energy 失效模式（必须牢记）
        ================================

        • Metric 主导问题（最危险）
            - quadratic（radial² + tangential²）过强
            → 压制 potential
            → 系统变成“少动优先”，而不是“朝目标走”

        • 对称性区分失败
            - dir==2 / Z-axis 区域：
                radial / tangential / rot 同时变弱
                所有几何量同时变小
            → energy 无法区分 move

        • 近区退化
            - norm_tz → 0
            → 所有信号衰减
            → energy landscape flatten
            在 near-target：
                radial → 0
                tangential → 弱
                rot → 弱
            → 所有动作分数接近

        ================================
        4. Ground Truth（强约束原则）
        ================================

        ||Uz - target||  （L2 distance）

        全局排序最可靠（backbone）
        在任何 regime 都不完全失效

        原则：
            L2 ranking is ground truth backbone
            geometric terms only modulate, never dominate

        ================================
        5. 关键系统事实（防误判）
        ================================

        • dir==2 / 2-cycle ≠ bug
            → 是群轨道（orbit phenomenon）

        • 接近目标 ≠ 更容易
            → 实际进入 isotropic + symmetry-dominated 区域

        • 当前系统 ≠ 连续优化
            → 本质是：
                continuous embedding + discrete orbit dynamics

        • move 行为：
            → progress + orbit + involution 混合

        ================================
        6. 额外动力学补偿（经验）
        ================================

        • prev_dz（动量项）用于：
            - 打破 2-cycle
            - 提供时间方向（temporal asymmetry）

        • 无 prev_dz 时：
            → 极易进入 orbit / 绕圈

        ================================
        7. 总结
        ================================

        魔方慢流形不是连续优化问题，
        而是：
            “离散群轨道 + 连续嵌入 + 对称性主导”的动力系统

        本函数的作用：
            提供一个“近似排序信号”，而不是精确物理能量

        对称性导致有效自由度降低，流体“冻结”成了具有周期性的结构。

    E = <dz, ∇V> + <dz, G(z) dz>  （Fredholm / 紧算子视角）

    动力学三阶段（Three Regimes）:
      Far (‖tz‖ large): radial 主导, potential 强, ranking 有意义
      Mid: radial/tangential 竞争, anisotropy 上升
      Near (‖tz‖→0): radial→0, potential flat, 2-cycle rate ~98%

    群对称性主导现象: 2-cycle orbit (g↔g⁻¹), dir==2 对称陷阱,
    对称子流形 (局部近似 Abel, 非交换结构被投影压平).

    Energy 失效模式: Metric 过强压制 potential, 近区所有信号衰减,
    energy landscape flatten → 所有动作分数接近.

    Ground Truth: ‖Uz - target‖ (L2) 是 backbone, 几何项只 modulate.
    本质: "离散群轨道 + 连续嵌入 + 对称性主导" 的动力系统.

    使用 move_energy(model, z, target). Use SlowDynamics.transport_score()
    for K_αβ-based theory-grounded scoring.
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


class QuantumAgent:
    """
      Minimal quantum agent:
    - density state
    - entropy / purity
    """

    def __init__(self, state, model: SlowDynamics):
        z = model.project(state.vector)
        z = z / (np.linalg.norm(z) + 1e-12)

        self.rho = np.outer(z, z.conj())  # density 密度矩阵（Hermitian, trace=1）
        self.energy = self.entropy()
        self.pos = np.random.randn(2) * 5
        self.age = 0

    def purity(self):
        return np.real(np.trace(self.rho @ self.rho))

    def entropy(self):
        """
        von_neumann_entropy S = -Tr(ρ ln ρ)
        """
        w = np.linalg.eigvalsh(self.rho)
        w = w[w > 1e-12]  # np.abs(agent.z)**2
        return -np.sum(w * np.log(w)) if len(w) > 0 else 0.0

    def rho_divergence(self, rho, mode: str = "trace"):
        if mode == "relative":  # 相对纯度差异
            return abs(np.real(np.trace(self.rho @ self.rho)) - np.real(np.trace(rho @ rho)))
        elif mode == "trace":  # 迹距离（Hilbert-Schmidt） 核范数
            return 0.5 * np.linalg.norm(self.rho - rho, ord='nuc')
        elif mode == 'fro':  # Frobenius 范数距离（最常用、最稳定）
            return np.linalg.norm(self.rho - rho, 'fro')

        fid = fidelity(self.rho, rho)
        return np.sqrt(2.0 - 2.0 * fid)  # Bures距离

    def expected_z(self):
        # 主方向（最大特征向量）
        w, V = np.linalg.eigh(self.rho)
        return V[:, np.argmax(w)]

    def species(self):
        """主特征向量签名,物种/基因/信仰"""
        v = self.expected_z()
        return tuple(np.sign(v.real[:6]))  # 简单hash

    def move(self):
        self.pos += 0.3 * np.random.randn(2)


class QuantumSimulation:
    """
    群结构（严格约束）
    慢流形（低维动力）
    量子概率（密度矩阵）
    统计力学（entropy/purity）
    进化（reproduce/mutate）
    密度矩阵版：连续、可微、必然均匀化（heat death）
    """

    def __init__(self, model: SlowDynamics, n_agents=20):
        self.model = model
        self.moves = [rho_s for _, rho_s in model.rho_slow.values()]
        self.agents = [
            QuantumAgent(CubieBase.generate_cubie(10), model)
            for _ in range(n_agents)
        ]

        self.history = []  # 记录纯度、熵、交换子强度

    def chaos_density(self, rho, samples: int = 20, commutator=True):
        """用交换子 [U, ρ] 量化量子非对易性/相干性
        简化版 C(t) ≈ || UW^t UV UW^{-t} ρ UW^t UV† UW^{-t} - ρ ||
        """
        total = 0
        for _ in range(samples):
            if commutator:
                Uc = random.choice(self.model.commutators.values())  # commutator 链 216
                rho2 = Uc @ rho @ Uc.conj().T
                total += np.linalg.norm(rho2 - rho, 'fro')  # || Uc ρ Uc† - ρ || 非交换性
            else:
                C_gh = random.choice(self.model.C_pairs.values())
                rho_diff = C_gh @ rho @ C_gh.conj().T
                total += np.linalg.norm(rho_diff, 'fro')

        return total / samples

    def evolve_density(self, agent, probs, alpha=0.3):
        """演化 ρ → Σ p_i U_i ρ U_i†"""
        rho_next = np.zeros_like(agent.rho, dtype=complex)
        for U, p in zip(self.moves, probs):  # (76×76) or 二阶结构（commutator algebra）作为动力
            rho_next += p * (U @ agent.rho @ U.conj().T)

        rho_next = (1 - alpha) * agent.rho + alpha * rho_next  # identity 保持结构,避免热寂
        agent.rho = rho_next / np.trace(rho_next)  # 数值稳定（归一化）

    def choose_probs(self, agent):
        """决策 Policy->后续 density / slow space 做策略评估"""
        scores = []
        for U in self.moves:
            rho2 = U @ agent.rho @ U.T
            purity_score = -agent.purity()  # 倾向保持纯度 purity 越纯越好（秩序）
            chaos_score = np.linalg.norm(rho2 - agent.rho, 'fro')  # 倾向保持非对易
            scores.append(purity_score + 0.3 * chaos_score)

        chaos = self.chaos_density(agent.rho)

        T = 0.2 + chaos  # chaos 控制随机性

        scores = np.array(scores)
        # p = np.zeros_like(scores)
        p = np.exp(-np.array(scores) / T)
        p /= p.sum() + 1e-12  # softmax(-np.array(scores) / T) 太平滑 → 导致平均化
        return p  # 会热寂

    @staticmethod
    def predator_prey(a, b):
        #  非对易性 = 捕食能力
        comm = np.linalg.norm(a.rho @ b.rho - b.rho @ a.rho)  # dominance
        if comm > 0.2:  # 非对易性强 → 耦合
            if a.purity() > b.purity():
                b.rho = 0.7 * b.rho + 0.3 * a.rho  # 高非交换 → 吞噬 a 吞 b
            else:
                a.rho = 0.7 * a.rho + 0.3 * b.rho

            a.rho /= np.trace(a.rho)
            b.rho /= np.trace(b.rho)

    def mutate(self, rho):
        """Mutation（结构保留）用 commutator chain 扰动 量子遗传（变异）"""
        # 随机选 2–4 个 U 做链
        U_chain = np.eye(rho.shape[0], dtype=complex)
        for _ in range(random.randint(2, 4)):
            U = random.choice(self.moves)
            U_chain = U @ U_chain  # @ U.conj().T
        rho2 = U_chain @ rho @ U_chain.conj().T
        return rho2 / (np.trace(rho2) + 1e-12)

    def destroy(self, rho):
        noise = np.random.randn(*rho.shape)
        rho2 = rho + 0.1 * (noise @ noise.T)
        tr = np.trace(rho2)
        return rho2 / (tr + 1e-8)

    # def repair(self, rho):
    #     for _ in range(5):
    #         p = choose_probs(...)
    #         rho = self.evolve_density(rho, p)
    #     return rho

    def death(self):
        """死亡：高熵/高年龄/高混沌/低纯度 → 淘汰"""
        self.agents = [
            a for a in self.agents
            if a.purity() > 0.1 and a.energy < 5.0 and a.age < 500
        ]

    def step(self):
        for agent in self.agents:
            p = self.choose_probs(agent)
            self.evolve_density(agent, p)
            agent.energy = agent.entropy()
            agent.age += 1

    def interact(self):
        """量子交互"""
        for i, a in enumerate(self.agents):
            for b in self.agents:
                if a is b: continue
                dist = np.linalg.norm(a.rho - b.rho, 'fro')
                if dist < 0.3:  # 相似 → 同步
                    U = random.choice(self.moves)
                    a.rho = U @ a.rho @ U.conj().T
                    b.rho = U @ b.rho @ U.conj().T

                d = np.linalg.norm(a.pos - b.pos)  # 局部才交互
                if d < 1.5:
                    self.predator_prey(a, b)

    def reproduce(self):
        """没有“遗传差异”，系统会塌缩成单一族群,鼓励多样性"""
        new_agents = []
        for agent in self.agents:
            if agent.purity() > 0.6 and random.random() < 0.1:
                child = QuantumAgent(CubieBase.generate_cubie(3), self.model)
                child.rho = self.mutate(agent.rho)
                child.pos = agent.pos + 0.5 * np.random.randn(2)
                new_agents.append(child)

        self.agents.extend(new_agents)

    def observe(self):
        ps = [a.purity() for a in self.agents]
        es = [a.entropy() for a in self.agents]
        cs = [self.chaos_density(a.rho, samples=10) for a in self.agents]
        species = [a.species() for a in self.agents]
        obs = {
            "purity_mean": np.mean(ps),  # chaos = mean(commutator)
            "entropy_mean": np.mean(es),
            "chaos_mean": np.mean(cs),
            "n_agents": len(self.agents),
            "species": len(set(species)),
        }

        self.history.append(obs)
        return obs

    def evolve(self, steps=100):
        for t in range(steps):
            self.step()
            self.interact()
            self.reproduce()
            self.death()

            obs = self.observe()
            if t % 10 == 0:
                print(
                    f"[{t}] purity={obs['purity_mean']:.3f} "
                    f"entropy={obs['entropy_mean']:.3f} "
                    f"chaos={obs['chaos_mean']:.3f} "
                    f"species={obs['species']} "
                    f"N={obs['n_agents']}"
                )

    def plot(self):
        if not self.history:
            return
        steps = np.arange(len(self.history))
        purity = [h['purity_mean'] for h in self.history]
        entropy = [h['entropy_mean'] for h in self.history]
        comm = [h['chaos_mean'] for h in self.history]
        n = [h['n_agents'] for h in self.history]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0, 0].plot(steps, purity, 'b-', label='Purity')
        axs[0, 0].set_title('Purity Evolution')
        axs[0, 0].grid(True)

        axs[0, 1].plot(steps, entropy, 'r-', label='von Neumann Entropy')
        axs[0, 1].set_title('Quantum Entropy')
        axs[0, 1].grid(True)

        axs[1, 0].plot(steps, comm, 'g-', label='Commutator Chaos')
        axs[1, 0].set_title('Exchange Chaos')
        axs[1, 0].grid(True)

        axs[1, 1].plot(steps, n, 'm-', label='Population')
        axs[1, 1].set_title('Population Size')
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_world(self):
        plt.figure(figsize=(6, 6))

        for a in self.agents:
            s = a.species()
            color = hash(s) % 10

            plt.scatter(a.pos[0], a.pos[1],
                        c=f"C{color}",
                        s=50 * a.purity(),
                        alpha=0.7)

        plt.title(f"N={len(self.agents)}")
        plt.grid(True, alpha=0.3)
        plt.show()


class Environment:
    """
    宇宙接口，迁移世界,切换模型，添加目标，调整权重，世界规则变化，
    环境驱动的认知进化
    """

    def __init__(self, model: SlowDynamics, targets: list = None):
        self.model = model

        # --- 基础 ---
        solved = CubieState.solved()
        self.z_solved = model.project(solved.vector)
        # --- target ---
        self.targets = targets if targets is not None else [self.z_solved]
        self.target_id = 0
        self.targets_weight = [1.0] * len(self.targets)

        # --- move pairs ---
        self.move_pairs = list(model.rho_slow.values())

        # --- 调度 ---
        self.scheduler = None

        self.context = {}

        self.id_counter = 0  # ID计数器

    def add_target(self, z):
        self.targets.append(z)
        self.targets_weight.append(1.0)
        size = len(self.targets)
        self.context["n_targets"] = size
        return size - 1

    def get_target(self, idx: int = 0):
        if 0 <= idx < len(self.targets):
            return self.targets[idx]
        return self.z_solved

    def get_levels(self, z):
        """返回每个目标的距离"""
        return [self.model.shell_level(z, t) for t in self.targets]

    def get_preference(self, z, target_id: int = 0, eta=0.2):
        """ 生态位"""
        # dirs = [t - z for t in self.targets]
        # weights = softmax(-[norm(d) for d in dirs])
        target_dir = self.targets[target_id] - z  # 指向目标的向量
        target_dir /= np.linalg.norm(target_dir) + 1e-8
        noise = np.random.randn(len(z))  # 个体噪声
        noise /= np.linalg.norm(noise) + 1e-8
        pref = (1 - eta) * np.real(target_dir) + eta * noise
        pref /= np.linalg.norm(pref) + 1e-8
        return pref

    def update_context(self, **kwargs):
        """环境上下文更新，可能影响 agent 的感知/决策"""
        self.context.update(kwargs)

    def adjust_weights(self, k: float = 2.0):
        """k>=1,新元素权重是每个旧元素的 k 倍"""
        n = len(self.targets)
        if n == 0:
            return
        if n == 1:
            self.targets_weight = [1.0]
            return
        w_old = 1.0 / (n - 1 + k)
        w_new = k * w_old
        self.targets_weight = [w_old] * (n - 1) + [w_new]

    def set_targets(self):
        """不要重新赋值"""
        if len(self.targets) > 1:
            return
        self.targets[0] = self.z_solved
        self.targets += self.base_targets(type=0)
        self.targets_weight = [1.0] * len(self.targets)

    def base_targets(self, type=0) -> list:
        if type == 1:
            return [self.model.project(s.vector) for s in
                    (CubieExample.twisted(), CubieExample.inversed(), CubieExample.big_cycle(),
                     CubieExample.checkerboard(), CubieExample.superflip())]

        targets = []
        s0 = CubieState.solved()
        for k in [(0, 1, 1), (1, 1, 1), (2, 1, 1)]:
            g = self.model.rho_moves[k][0]
            s1 = g.act(s0)
            z = self.model.project(s1.vector)
            targets.append(z)
        return targets

    def generate_far_target(self, threshold=7.0):
        while True:
            s = CubieBase.generate_cubie(length=15)
            z = self.model.project(s.vector)
            if all(np.linalg.norm(z - z0) > threshold for z0 in self.targets):
                return z

    def generate_diff_target(self, threshold=0.4):
        """正交目标"""
        current_threshold = threshold
        attempt = 0
        while True:
            s = CubieBase.generate_cubie(length=20)
            z = self.model.project(s.vector)
            max_dot = max(abs(np.dot(z, z0)) for z0 in self.targets)
            if max_dot < current_threshold:
                return z
            if attempt % 10 == 0:
                current_threshold += 0.01
            attempt += 1

    @staticmethod
    def generate_target_from_agents(zs):
        """agent.set_target"""
        # zs = np.stack([a.z for a in self.agents])
        center = zs.mean(axis=0)
        idx = np.argmax(np.linalg.norm(zs - center, axis=1))  # 找一个“最远点”
        return zs[idx]


class HybridAgent:
    """
    双态系统,用量子态做观测 / 决策，用群作用做真实演化
    多 agent 生态：resource 竞争 + 4 种互动（imitate / predator / reproduce / emulate）。
    Phase 决策（Exploit / Revert / Explore / Natura）驱动行为。
    目标：从离散群 + 慢流形中涌现集体智能、世界模型、记忆与时间箭头。
    | 层级 | 实现             |
    | -- | ------------------ |
    | 微观 | 群作用                |
    | 表示 | slow manifold      |
    | 认知 | density matrix     |
    | 决策 | softmax + chaos    |
    | 资源 | resource field     |
    | 社会 | imitate + predator |
    | 演化 | reproduction       |
    """

    # __slots__ = ('state', 'z', 'rho',  'pos')

    def __init__(self, state: CubieState, env: Environment, mutate_length=0):
        self.env = env
        self.model = env.model
        self.move_pairs = env.move_pairs

        # 粒子态（真实）保持离散
        self.state = state
        self.memory = deque(maxlen=10)
        self.memory.append(state)
        self.gm = CubieMove.identity()

        # 波函数（慢空间）用于几何
        self.z = env.model.project(state.vector)
        self.prev_dz = None

        # 多场耦合系统
        z_hat = self.z / (np.linalg.norm(self.z) + 1e-8)  # 物理状态 z 的模长 = 物理信息（不能丢）,方向 = 认知信息（可以变）
        self.rho = np.outer(z_hat, z_hat.conj())  # 纯态密度矩阵,感知层,不参与真实演化（观测更新），马尔可夫式的当下存在，自在的存在
        if mutate_length > 0:  # 量子遗传（变异）
            self.rho, g = self.model.mutate(self.rho, length=mutate_length)

        self.rho_plan = self.rho.copy()  # future 快层 规划态/工作记忆 planning / imagination（不被观测污染），投向各种可能性，自为的存在
        self.rho_mem = self.rho.copy()  # past 慢层 记忆场,长期记忆锚点,海马体长期记忆库,自我连续性，防止完全虚无化导致崩溃

        self.chaos_val = self.chaos_density(mode='fast')  # z
        self.curvature_val = self.model.lie_curvature(self.z)  # z,路径稳定性
        self.move_entropy_val = 0.0

        self.plan_entropy_val = von_neumann_entropy(self.rho_plan)  # plan_entropy，初始 0
        self.purity_val = 1.0  # plan 规划态纯度
        # 相对指标
        self.fidelity_val = 1.0  # mem/plan 规划↔记忆锚点 相对相似度
        self.consistency_val = 1.0  # a/mem，初始 1 vs 记忆 的连续性
        self.divergence_val = 0.0  # a/plan，初始 0  规划态漂移 vs 信念相对差异

        self.target_id = 0
        self.level, dist = self.model.shell_level(self.z, env.get_target(self.target_id))
        self.energy = dist ** 2

        self.preference = env.get_preference(self.z, self.target_id)  # 生态位偏好向量
        self.alignment = 1.0
        self.energy_delta_cum = 0.0  # 总能量变化绝对值（经典运动量）
        self.entropy_delta_cum = 0.0  # 总熵变化绝对值（持续活跃度）

        self.confidence = 0.0
        # 生态特征
        self.age = 0
        self.resource = 1.0  # 资源（生存能力）
        self.stats = {'phase': -1, 'T': 0.0, 'T_phase': 0.0,
                      'move_id': -1, 'move_probs_mean': 0.0, 'move_probs_std': 0.0,
                      'diversity': 2.0 * np.pi}
        self.pos = np.random.randn(2) * 5  # 空间位置（局部交互）
        self.species_id = tuple(np.sign(self.z.real[:6]))  # 主特征向量签名,物种/基因/信仰/行为结构 cluster

        # self.move_probs = np.zeros(len(self.move_pairs))  # 当前动作概率分布
        self.phase_count = np.zeros(4, dtype=int)  # exploit/rollback/explore/mutate
        self.interact_count = np.zeros(6, dtype=int)
        self.get_energy()
        self.children: list['HybridAgent'] = []  # 子代对象列表
        env.id_counter += 1

    def quantum_evolve(self, k=3, alpha=1 / 3, beta=1 / 9, temperature=1 / 3):
        """保持相干性,量子大脑轻微混合 局部混合 = 有方向的思考
        把 agent 的决策拆成 4 种相位（phase)
        探索-利用-修复-突变 四态系统
        存在先于本质,虚无是自由的条件
        beta 观测强度 β=1 → 现在的（强坍缩）β<1, beta 越高 → 塌缩越彻底（越不可逆）
        """
        self.divergence_val = 0.5 * np.linalg.norm(self.rho_plan - self.rho, ord='nuc')  # 0.4~1.2 规划空间的混乱程度与纯态信念的偏离程度
        beta_eff = beta + 1.0 / 9 * (self.divergence_val - self.fidelity_val)
        self.rho_plan = (1 - beta_eff) * self.rho_plan + beta_eff * self.rho  # 适度拉回当前现实观测
        self.rho_plan /= np.trace(self.rho_plan)
        old_entropy = self.plan_entropy_val
        self.plan_entropy_val = von_neumann_entropy(self.rho_plan)  # 规划空间混乱度0~3 纯 rho_b: 0.05 /0.7 /1.5 /1.2
        self.entropy_delta_cum += abs(self.plan_entropy_val - old_entropy)
        self.purity_val = np.real(np.trace(self.rho_plan @ self.rho_plan))  # 0~1，越纯越好
        self.fidelity_val = fidelity(self.rho_mem, self.rho_plan)  # 0~1,规划空间与当前记忆锚点的相似度,比 divergence 更稳定,但不敏感于纯度变化

        phase, p, idx = self.choose_phase(k=k, temperature=temperature)

        z_new = self.z.copy()
        best_id = -1
        move_cost = 0
        if phase == 0:  # 锁位 资源充足且适合 → 执行规划,试图成为某种固定本质:A Exploit
            best_id = idx[-1]
            g, U_opt = self.move_pairs[best_id]
            z_new = U_opt @ self.z
            rho_next = U_opt @ self.rho @ U_opt.conj().T
            move_cost = max(0.04, 0.21 - 0.07 * self.confidence)  # 范围约 0.03 ~ 0.18
            # exploit 允许“认知跃迁”
        elif phase == 1:  # 资源充足但不适合 →回退rollback:AB Revert
            s_past = random.choice(list(self.memory))  # 回归点原修复，容许人犯错，仅限瞬息间
            g = CubieMove.build(self.state, s_past)
            z_past = self.model.project(s_past.vector)
            z_past /= np.linalg.norm(z_past) + 1e-8
            rho_a = np.outer(z_past, z_past.conj())
            U = self.model.project_move(g.rho())
            z_new = U @ self.z  # z_past
            rho_b = U @ self.rho_plan @ U.conj().T
            rho_next = 0.5 * rho_a + 0.5 * rho_b  # 这功能有点逆天，打破真实与虚妄，虚实结合，现实 + 记忆 量子干涉
            move_cost = max(0.05, 0.32 - 0.08 * self.confidence)
        elif phase == 2:  # 更相信自身信念，采样+ 精细打分，大胆投向可能性（真实性）:B Explore
            raw_scores = self.choose_scores(idx=idx)  # shape (18, 4)
            top_component = raw_scores[idx]  # shape (k, 4)
            top_component = (top_component - top_component.mean(axis=0)) / (top_component.std(axis=0) + 1e-6)
            p_k = p[idx]
            p_k /= p_k.sum()
            w = np.array([0.4, 0.3, 0.2, 0.1])  # chaos,-purity, -divergence, mem_align
            comp_score = np.tanh(np.dot(top_component, w))
            move_scores = 1.0 / 3 * p_k + 2.0 / 3 * comp_score  # + top_p
            # best_id = np.random.choice(idx, p=p_k)
            best_id = idx[np.argmax(move_scores)]
            g, U_opt = self.move_pairs[best_id]
            z_new = U_opt @ self.z
            zt = z_new.copy()
            zt /= np.linalg.norm(zt) + 1e-8
            rho_next = np.outer(zt, zt.conj())  # 纯态更新 U_opt @ self.rho_plan @ U_opt.conj().T
            move_cost = max(0.04, 0.22 - 0.1 * self.confidence)
        else:  # 道法自然,多步演化 np.mean(p[idx]) < 1.0 / len(self.move_pairs) + np.std(p[idx]):
            zt = self.model.evolve(self.z, T=k)  # 先让当前认知在慢流形上演化一段时间，得到一个“预期位置”
            zt /= np.linalg.norm(zt) + 1e-8
            rho_a = np.outer(zt, zt.conj())
            rho_next, g = self.model.mutate(rho_a, length=random.randint(2, k), p=p)  # 探索性变异+random_walk
            U = self.model.project_move(g.rho())
            z_new = U @ self.z
            # move_cost = -0.05 + 0.1 * random.random()

        self.prev_dz = z_new - self.z

        self.phase_count[phase] += 1
        self.stats['phase'] = phase
        self.stats['move_id'] = best_id
        self.stats['move_cost'] = move_cost
        # 从 rho 出发 explore（少量混合） + 保留经典身份
        # S_norm = self.plan_entropy_val / (np.log(self.rho_plan.shape[0]) + 1e-8)
        alpha_eff = alpha + 1.0 / 9 * self.plan_entropy_val  # 用规划熵控制激进程度,混乱时更激进规划
        # alpha_eff = np.clip(alpha_eff, 0.05, 0.5)
        self.rho_plan = (1 - alpha_eff) * self.rho_plan + alpha_eff * rho_next
        self.rho_plan /= np.trace(self.rho_plan)

        # sig_loss = rho_sigreg(rho=self.rho_plan, lambda_reg=0.09)
        # self.rho_plan = self.rho_plan - sig_loss * (self.rho_plan - 0.5 * np.eye(self.rho_plan.shape[0]))

        return g, move_cost

    def quantum_update(self, g: CubieMove, samples: int = 10):
        """从真实 state 更新“认知”,观测坍缩,同步认知"""
        # 身体演化（离散、确定性） O(20) 操作（perm + ori） 严格群结构（真实世界）
        self.state = g.act(self.state)
        self.gm = self.gm.compose(g)
        self.memory.append(self.state)

        self.z = self.model.project(self.state.vector)
        self.chaos_val = self.chaos_density(samples=samples, mode='fast')  # 真实动力学的非交换性表现 'show'/'commutator'
        self.curvature_val = self.model.lie_curvature(self.z, k=6)

        z_hat = self.z / (np.linalg.norm(self.z) + 1e-8)
        self.rho = np.outer(z_hat, z_hat.conj())  # rho_obs

    def quantum_dream(self, step=5, gamma=1 / 3):
        """
        Dream 模式：使用 rho_mem 进行离线想象/模拟
        不改变真实 state，只更新 rho_plan 和 rho_mem
        混乱时更需要重组记忆
        """
        if self.age % step == 0:  # 长期记忆缓慢巩固
            rho_mem_fresh = self.memory_field(limit=10, decay_tau=4.0, symmetry=True)  # 过去+未来的稳定结构
            zt = self.env.targets[self.target_id]
            zt /= np.linalg.norm(zt) + 1e-8
            target_bias = np.outer(zt, zt.conj())  # 生成未来策略的源头
            beta_eff = 1.0 / 9 * (1 - self.alignment)
            rho_mem_fresh = (1 - beta_eff) * rho_mem_fresh + beta_eff * target_bias
            self.rho_mem = (1 - gamma) * self.rho_mem + gamma * rho_mem_fresh
        else:  # 轻微 EMA 更新（吸收当前 rho_a）
            gamma_eff = gamma / step + 1.0 / 9 * von_neumann_entropy(self.rho_mem)  # 混乱时更依赖观测
            self.rho_mem = (1 - gamma_eff) * self.rho_mem + gamma_eff * self.rho  # 当前 belief 用现实修正记忆

        self.rho_mem /= np.trace(self.rho_mem)
        self.consistency_val = np.real(np.trace(self.rho @ self.rho_mem.conj().T))  # 惯性,一致性,包含方向信息 0.22-0.7

        if self.age > 1 and self.plan_entropy_val < 0.3 and self.purity_val > 0.90 and self.divergence_val < 0.2:
            self.interact_count[4] += 1  # Boltzmann brain 状态持续,低熵涨落如何被第二定律摧毁 bb_lifetime
        if self.plan_entropy_val > 2.0 and self.fidelity_val < 0.1 and self.consistency_val < 0.1:  # 认知崩溃检测，强锚定回当前信念,混乱时回忆过去的认知
            self.rho_plan = 5.0 / 9 * self.rho + 1.0 / 3 * self.rho_mem + 1.0 / 9 * self.rho_plan
            self.rho_plan /= np.trace(self.rho_plan)
            print(f"[Crisis Anchor] entropy={self.plan_entropy_val:.3f} fidelity={self.fidelity_val:.3f} "
                  f"div={self.divergence_val:.3f} purity={self.purity_val:.3f}")  # 虚假记忆如何快速被现实修正

    def memory_field(self, limit=5, decay_tau=3.0, symmetry=False):
        """
        支持时间对称的长期记忆场, 从历史得到认知+ 时间衰减权重 + 从当前z反向演化出的虚拟记忆
        海马体式，提供认知锚点
        """
        if not self.memory:
            return self.rho.copy()
        # 取过去几个状态
        past_states = list(self.memory)[-limit:]
        rho_mem = np.zeros_like(self.rho, dtype=complex)
        for i, s in enumerate(reversed(past_states)):
            z_past = self.model.project(s.vector)
            z_past /= np.linalg.norm(z_past) + 1e-8
            weight = np.exp(-i / decay_tau) if decay_tau > 0 else 1.0
            rho_mem += weight * np.outer(z_past, z_past.conj())

        if symmetry:  # 加入预测的“未来”记忆，使其近似时间平移不变:观察 bb_lifetime 是否显著增加
            z_future = self.model.evolve(self.z, T=min(len(past_states), limit // 2))  # 用谱演化预测未来 T 步（对称于过去）
            z_future /= np.linalg.norm(z_future) + 1e-8
            rho_future = np.outer(z_future, z_future.conj())
            rho_mem = 0.5 * rho_mem + 0.5 * rho_future  # 对称融合（过去和未来等权重）

        rho_mem /= np.trace(rho_mem) + 1e-12
        return rho_mem

    def resource_field(self, pos):
        """简单径向 + 多峰资源场（可扩展成真实环境地图）"""
        if pos is None:
            pos = self.pos
        r = np.linalg.norm(pos)
        # 主峰在原点 + 4个随机小峰（模拟资源斑块）
        field = 0.6 * np.exp(-r ** 2 / 25)
        for i in range(4):
            peak = np.array([np.sin(i * 1.7) * 12, np.cos(i * 1.7) * 12])  # centers
            field += 0.15 * np.exp(-np.linalg.norm(pos - peak) ** 2 / 12)
        return field

    def get_energy(self, step=10, momentum=0.2):
        """经典能量（slow distance²）"""
        dists = self.env.get_levels(self.z)
        best_target_id = np.argmin([d for _, d in dists])  # np.argsort(dists)[:k]
        if self.age % step == 0 or self.level == -1 or self.target_id == -1:  # target 不要每步贪心切换
            self.target_id = best_target_id
            eta_eff = 0.2 * (2 - self.consistency_val + self.chaos_val)
            new_pref = self.env.get_preference(self.z, self.target_id, eta=eta_eff)
            self.preference = (1.0 - momentum) * self.preference + momentum * new_pref
            self.preference /= np.linalg.norm(self.preference) + 1e-8

        target_dir = self.env.targets[self.target_id] - self.z
        target_dir /= np.linalg.norm(target_dir) + 1e-8
        self.alignment = np.dot(target_dir.real, self.preference)  # cos θ [-1, 1] 0.3,0.7
        self.level, dist = dists[self.target_id]
        if self.level <= 3.0:  # 靠近目标额外奖励
            reward = 1.0 * (2 * np.pi - dist) / (2 * np.pi)
            if self.level == 0:
                reward += 1.0  # 额外奖励 self.state == CubieState.solved()
                self.interact_count[5] += 1
                print(
                    f"Reached target {self.target_id} at age {self.age} with resource {self.resource:.2f},stats: {self.stats}")
                self.target_id = np.argmax([d for _, d in dists])
            self.resource += reward

        old_energy = self.energy
        self.energy = dist ** 2
        self.energy_delta_cum += abs(self.energy - old_energy)
        self.resource += 0.1 * (1.0 / (1.0 + self.energy))
        return self.energy

    def chaos_density(self, samples=10, mode='fast'):
        """对易性观测量:表示层非对易性/真实动力学扰动"""
        if mode == 'fast':  # slow
            sigs = self.model.chaos_signature(self.z, samples)
            return sigs.mean()

        total = 0.0
        for _ in range(samples):
            (g, Ug), (h, Uh) = random.sample(self.move_pairs, 2)
            if mode == 'quantum':  # rho_diff ≈ C_gh @ rho @ C_gh.conj().T
                rho_gh = Ug @ (Uh @ self.rho @ Uh.conj().T) @ Ug.conj().T
                rho_hg = Uh @ (Ug @ self.rho @ Ug.conj().T) @ Uh.conj().T
                total += np.linalg.norm(rho_gh - rho_hg, 'fro')  # 0.07-0.4
            elif mode == 'show':  # 非交换性在慢空间的表现，完全在慢流形上,非对易性被压平了
                z_gh = Ug @ (Uh @ self.z)  # g 后 h
                z_hg = Uh @ (Ug @ self.z)  # h 后 g
                total += np.linalg.norm(z_gh - z_hg)  # 0.1-0.35 chaos_signature np.linalg.norm(C_gh @ z)
            else:  # Rubik group 本来就是强非交换群,chaos 太强 → 系统发散 or 震荡
                s_gh = (g @ h).act(self.state)  # g.act(h.act(self.state))
                s_hg = (h @ g).act(self.state)
                z_gh = self.model.project(s_gh.vector)
                z_hg = self.model.project(s_hg.vector)
                total += np.linalg.norm(z_gh - z_hg)  # 0.5-3

        if mode == 'quantum' or mode == 'show':
            return total / samples

        z_norm = np.linalg.norm(self.z) + 1e-8
        return (total / samples) / z_norm

    def choose_scores(self, idx=None):
        """
        只对这 k 个动作做精细的 4 维 component score 计算，节省计算量
        倾向：高纯度（秩序）+ 适度混沌（非交换性）
        对指定的 idx（top-k）计算 4 维 component score，其余保持 
        move_scores → p → idx = top-k
        """
        if idx is None:
            idx = range(len(self.move_pairs))

        # rho_next = np.zeros_like(self.rho_plan, dtype=complex)
        scores = np.zeros((len(self.move_pairs), 4), dtype=np.float64)
        for i in idx:
            _, U = self.move_pairs[i]
            sim_rho_real = U @ self.rho @ U.conj().T
            chaos = np.linalg.norm(sim_rho_real - self.rho, 'fro')  # 非对易倾向 0.2-1.3
            purity_gain = np.real(np.trace(sim_rho_real @ sim_rho_real)) - self.purity_val  # 自然系统里不是：变得更纯
            divergence = 0.5 * np.linalg.norm(sim_rho_real - self.rho_plan, ord='nuc') - self.divergence_val  # 与规划态的匹配度
            mem_align = np.linalg.norm(sim_rho_real - self.rho_mem, 'fro')  # 保持轨道连续性 -self.consistency_val
            # sim_rho_plan = U @ self.rho_plan @ U.conj().T
            # plan_consistency = np.real(np.trace(sim_rho_plan @ self.rho_plan))

            scores[i, 0] = chaos  # 探索性
            scores[i, 1] = -purity_gain  # 稳定性,适度混合,纯度太高:探索性丢失
            scores[i, 2] = -divergence  # 规划匹配度 np.real(np.trace(sim_rho @ self.rho_plan))
            scores[i, 3] = -mem_align  # 记忆连续性
            # rho_next += p[i] * sim_rho

        return scores

    def choose_phase(self, k=3, temperature=1 / 9):
        """phase 选择逻辑：整合所有变量，输出 phase,用 4 个 phase 的分数决定行为模式
        焦虑是自由的证明:phase 切换"""
        target = self.env.get_target(self.target_id)
        energy = move_energy(self.model, self.z, target, self.prev_dz)
        fitness = - energy  # 越高越好
        f = normalize_z(fitness)
        p0 = softmax(f)  # softmax 太平滑 → 导致平均,无T
        self.move_entropy_val = -np.sum(p0 * np.log(p0 + 1e-8))
        entropy = self.move_entropy_val / np.log(len(p0))
        scale = np.percentile(np.abs(fitness), 80) + 1e-6
        # scores = np.clip(fitness,-15, 15)  # 限制范围，避免溢出
        T = temperature * (1 + 3.0 * self.curvature_val) + 2.0 * self.chaos_val  # 动作层,越高越随机
        # p = np.exp(-scores / T)  # 量子决策 math.exp(-dE / temperature)
        f = fitness / (T * scale + 1e-8)
        f = f - np.max(f)
        p = np.exp(f)
        p /= p.sum() + 1e-12

        idx = np.argsort(p)[-k:]
        best_id = idx[-1]
        p_max = p[best_id] / p[idx].sum()  # 在候选中的相对优势

        sharpness = np.sum(p ** 2)  # Gini-like ,1/n,1
        suitability = p_max + 2.0 / 3 * sharpness + 1.0 / 3 * (1.0 - entropy)  # 执行阈值
        instability = self.chaos_val * (1 + 2.0 * self.curvature_val) + 1.0 / 3 * entropy
        # print(f"Suitability: {suitability:.3f} /{instability:.3f} (p={p[best_id]:.3f}, purity={self.purity_val:.3f}, plan_entropy={self.plan_entropy_val:.3f},"
        #       f"entropy={entropy:.3f},sharpness={sharpness:.3f},p_max={p_max:.3f}")

        suitability = np.clip(suitability, 0.0, 1.0)
        instability = np.clip(instability, 0.0, 1.0)
        self.confidence = (1.0 - instability + suitability) / 2.0

        phase_scores = np.array([
            suitability + 1.0 / 3 * (1.0 - self.divergence_val + self.fidelity_val + self.purity_val),
            # exploit: 高信心 + 高规划可靠
            (1.0 - suitability) * self.consistency_val * self.resource / np.sqrt(self.age + 1),
            # revert:  低信心 + 规划失效,认知崩了才回退
            instability + self.divergence_val + 1.0 / 9 * self.plan_entropy_val,  # (1.0 - self.consistency_val)
            # explore: 高混乱 + 高曲率
            self.consistency_val * (1.0 - abs(self.purity_val - 0.5) - abs(self.fidelity_val - 0.5)) +
            self.alignment * (1 + 2.0 * self.curvature_val)
            # natura:  中性、保守，结构沉淀阶段，鞅偏差
        ])  # phase_vector

        # conflict_exploit = np.clip((self.plan_entropy_val - 0.8) / 0.8, 0.0, 1.0)
        # conflict_explore = np.clip((0.5 - self.fidelity_val) / 0.5, 0.0, 1.0)

        # phase_scores[0] *= (1.0 - 0.6 * conflict_exploit)
        # phase_scores[0] -= 0.2 * conflict_explore
        # phase_scores[2] *= (1.0 + 0.5 * conflict_explore)
        # phase_scores[2] += 0.3 * conflict_exploit
        ps = normalize_z(phase_scores)
        ps = np.tanh(ps)
        T_phase = temperature * (1 + 1.0 / 3 * self.chaos_val)  # 策略层
        x_scores = ps / (T_phase + 1e-6)
        phase_p = softmax(x_scores)  # 限制范围，避免溢出, 提高数值稳定性

        try:
            phase = np.random.choice(4, p=phase_p)  # 会抖,or last_phase
        except ValueError:
            phase = np.argmax(phase_p) or self.stats['phase']  # 如果概率分布有问题，选择概率最高的阶段
            # if np.any(np.isnan(phase_p)) or np.sum(phase_p) == 0:
            print(
                f"Phase scores: {phase_scores},scores:{fitness} Phase probabilities: {phase_p}, T_phase: {T_phase:.3f}")

            print("energy range:", energy.min(), energy.max())

        self.stats['T'] = T
        self.stats['T_phase'] = T_phase
        self.stats['move_probs_mean'] = p.mean()
        self.stats['move_probs_std'] = p.std()  # 0.05
        self.stats['suitability'] = suitability
        self.stats['instability'] = instability
        # self.stats['move_cv']  = np.real(np.std(energy) / (np.mean(energy) + 1e-8))  #  概率分布集中度

        return int(phase), p, idx

    def step(self, T=1 / 9, L: int = 1, S: int = 10, neighbors=None):
        """核心双态步进：量子决策 → 经典执行 → 大脑更新 量子大脑混合"""
        for _ in range(L):
            # 量子大脑轻微演化 未来模拟,融合同步记忆 k = int(3 + 6 * self.entropy())
            g, move_cost = self.quantum_evolve(k=3, alpha=1 / 3, beta=1 / 9, temperature=T)
            if move_cost > 0:
                self.resource -= move_cost
                # print(f"Agent executes move {self.move_id} with cost {move_cost:.3f} and suitability {p[best_idx]:.3f}")
            # 观测更新
            self.quantum_update(g)

        # 弱坍缩,现实校正
        self.quantum_dream(step=5, gamma=1 / 3)

        self.get_energy(step=S)
        # 资源动态
        self.gather_resource(neighbors)
        self.age += 1

        self.pos += 0.25 * np.random.randn(2)  # 随机扩散
        grad_x = self.resource_field(self.pos + np.array([0.1, 0])) - self.resource_field(self.pos)
        grad_y = self.resource_field(self.pos + np.array([0, 0.1])) - self.resource_field(self.pos)
        self.pos += 0.8 * np.array([grad_x, grad_y])

    def emulate(self, g: CubieMove):
        self.quantum_update(g)  # 同步认知
        self.get_energy()
        self.interact_count[3] += 1

    def imitate(self, other_agent: 'HybridAgent', data: dict = None):
        """
        社会学习：模仿经典状态，量子自动同步,弱者向强者学习
        imitation = phase 同步
        """
        if not other_agent.memory:
            return
        if self.purity_val > other_agent.purity_val:
            return
        relative_strength = self.purity_val * self.fidelity_val - other_agent.purity_val * other_agent.fidelity_val
        if abs(relative_strength) < 0.1:  # 实力相近不学（避免过度模仿导致系统崩溃）
            return

        directed_d = data.get("directed_d", np.dot(other_agent.z - self.z, self.preference).real)  # 沿自身偏好方向投影
        if directed_d < 0.1:  # 方向性判断
            return  # 太近不模仿（防止塌缩,抹平结构）collapse,只有当对方在自己偏好方向上更接近目标时才模仿

        cultural_d = np.linalg.norm(self.rho_plan - other_agent.rho_plan, 'fro')  # 规划认知差异
        if cultural_d > 1.0:  # 规划认知差异过大不模仿（防止认知冲突）
            return
        score = 0.5 * sigmoid(directed_d) + 0.4 * (1 - cultural_d / 2.0) + 0.1 * self.resource
        trigger_prob = np.clip(score, 0.05, 1.0)
        if random.random() > trigger_prob:
            return

        # comm = np.linalg.norm(self.rho_mem @ other_agent.rho_mem - other_agent.rho_mem@ self.rho_mem)
        # if comm < 0.2:
        #     return

        # 模仿对方的规划认知或者参考对方的记忆场
        if self.divergence_val > 0.8 and self.resource > 3:  # 当自己的规划和现实差异过大时，更倾向于参考对方的记忆场（更稳定）
            teacher_rho = other_agent.rho_mem.copy()
            transfer = min(0.5, 0.1 * self.resource)  # 法不轻传
            self.resource -= transfer
            other_agent.resource += transfer
        else:
            teacher_rho = other_agent.rho_plan.copy()

        # 融合对方的规划认知，只学习对自己有用的部分
        mix_ratio = 1 / 9 * min(1.0, directed_d * self.resource * self.plan_entropy_val)
        delta = teacher_rho - self.rho_plan
        align = np.real(np.trace(delta @ np.outer(self.preference, self.preference)))

        self.rho_plan += mix_ratio * align * delta  # 记忆融合，文化吸收
        self.rho_plan /= np.trace(self.rho_plan)  # 暂时不影响 rho_a

        self.interact_count[0] += 1  # 模仿计数

    def predator(self, other_agent: 'HybridAgent', data: dict = None):
        """
        捕食：改变 state 和 资源+ 领地推进
        predator = phase 扰动
        """
        relative_strength = self.purity_val * self.resource - other_agent.purity_val * other_agent.resource
        if abs(relative_strength) < 0.3:  # 实力相近不打（避免过度竞争导致系统崩溃）
            return

        # 强者判定
        if relative_strength > 0:
            predator, prey = self, other_agent
        else:
            predator, prey = other_agent, self
            return  # 单向,只让 stronger 调用 weaker

        pos_d = data.get("pos_d", np.linalg.norm(self.pos - other_agent.pos))
        local_avg = 0.5 * (self.resource + other_agent.resource)
        relative_strength /= (local_avg + 1e-6)
        transfer = max(0, 0.1 * prey.resource * sigmoid(relative_strength))
        predator.resource += 0.95 * transfer
        prey.resource -= transfer

        prob = np.exp(-pos_d ** 2 / 5.0) * self.plan_entropy_val
        # 量子捕食（位置近才触发）
        if random.random() < prob:  # 边界扰动,捕食成功后，弱者可能发生认知变异（逃避/适应）
            g = self.model.random_walk(length=2)
            prey.emulate(g)

        direction = predator.pos - prey.pos
        prey.pos -= 0.3 * direction / (pos_d + 1e-6)

        self.interact_count[1] += 1  # 捕食计数

    def interact_kernel(self, agents, values: list, D=0.03, S=0.08, T=0.2, R=0.1):
        """
        连续场近似：kernel-based PDE update
        统一加权更新，状态转移
        diffusion（模仿） + selection（竞争） + transport（捕食） + entropy（抗塌缩）
        """

        rho_i = self.rho_plan
        x_i = self.z

        # local_diversity = np.mean([np.linalg.norm(self.z - n.z) for n in neighbors])

        diffusion_term = np.zeros_like(rho_i, dtype=complex)
        transport_term = np.zeros_like(x_i)

        entropy = self.plan_entropy_val
        curvature_i = self.curvature_val
        r_i = self.resource
        f_i = self.purity_val * max(1e-6, r_i)
        selection_term = 0.0
        fitness_accum = 0.0
        resource_flow = 0.0
        weight_sum = 1e-8

        for b in agents:
            if b is self:
                continue

            rho_j = b.rho_plan
            x_j = b.z
            # --- 1️⃣ kernel ---
            dist = np.linalg.norm(x_i - x_j)
            if dist > 2 * np.pi:  # 距离过远直接跳过
                continue
            same_species = (self.species_id == b.species_id)
            same_target = (self.target_id == b.target_id)
            species_gate = 0.2 + 0.8 * (1.0 if same_species else 0.0)
            target_gate = 0.5 + 0.5 * (1.0 if same_target else 0.0)

            dir_align = np.dot(x_j - x_i, self.preference).real  # imitate的核心
            # --- kernel ---
            K = np.exp(-dist ** 2 / 2.0)
            K *= species_gate
            K *= target_gate

            learn_gate = sigmoid(dir_align) * (1 - entropy)  # imitate
            cultural_d = np.linalg.norm(rho_i - rho_j, 'fro')
            cultural_gate = np.exp(-cultural_d)

            # --- predator gating（predator_prob）---
            f_j = b.purity_val * max(1e-6, b.resource)
            strength_diff = (f_j - f_i) / (abs(f_i) + abs(f_j) + 1e-6)
            flow_gate = sigmoid(strength_diff)
            pred_gate = flow_gate * np.exp(-dist)
            pred_gate *= (1 + 2.0 * curvature_i)

            # --- 2️⃣ diffusion ---
            diffusion_term += K * learn_gate * cultural_gate * (rho_j - rho_i)  # 模仿

            # === 2️⃣ selection（replicator）===
            selection_term += K * (f_j - f_i)

            # --- resource flux ---
            resource_flow += K * flow_gate * (b.resource - r_i)

            # === 3️⃣ transport（捕食 → 空间推进）===
            direction = (x_j - x_i) / (dist + 1e-6)
            transport_term += K * pred_gate * direction

            # --- 3️⃣ fitness（selection准备） ---
            f_j = b.purity_val * max(0.0, b.resource)
            fitness_accum += K * f_j

            weight_sum += K

        # === diffusion ===
        self.rho_plan += D * diffusion_term / weight_sum  # 0.02~0.05 模仿融合
        # === selection（replicator）===
        self.rho_plan += S * (selection_term / weight_sum) * rho_i  # 0.05~0.1

        # self.z += T * transport_term / weight_sum

        self.resource += R * resource_flow / weight_sum
        self.resource = max(1e-6, self.resource)
        # --- entropy（抗塌缩）---
        H = -np.real(np.trace(rho_i @ np.log(rho_i + 1e-8)))
        entropy_force = 0.02 * (1.0 - H)
        # === anti-collapse normalize ===
        self.rho_plan /= np.trace(self.rho_plan) + 1e-12
        # === mutation（底噪）===
        instability = (1 - entropy) + np.abs(selection_term / weight_sum)

        mutation_prob = sigmoid(instability)
        if random.random() < 0.02 * mutation_prob:
            g = self.model.random_walk(length=1)
            self.emulate(g)
            # teacher = random.choice(list(other_agent.memory))
            # g = CubieMove.build(self.state, teacher)  # 带着求经问道之心
            # U = self.model.project_move(g.rho())
            # teacher_rho = U @ self.rho_plan @ U.conj().T

    def gather_resource(self, neighbors):
        """获取资源：生态位匹配 + 局部空间场"""
        # 1. 慢空间生态位匹配
        gain = max(0.0, float(self.alignment)) * 0.15
        if neighbors:  # 局部竞争
            competition = sum(n.resource for n in neighbors) / len(neighbors)
            gain *= np.exp(-0.3 * competition)

        season = 0.3 * np.sin(2 * np.pi * self.age / 50)  # 季节波动
        gain *= (1.0 + season)
        self.resource += gain
        # 局部资源场（空间结构）
        self.resource += self.resource_field(self.pos)
        # 资源代谢衰减
        self.resource *= 0.97
        # self.resource += 0.05  # baseline inflow

    def reproduce(self, max_length=4):
        """繁殖：只有高纯度 + 低能量时才产生后代"""
        # 经典小变异
        length = random.randint(2, max_length)
        child_state = self.state.clone()
        if random.random() < 0.3:
            g = self.model.random_walk(length=length)
            child_state = g.act(child_state)
        else:  # 直接用交换子矩阵,群结构的“内禀曲率”. 3.7-3.9
            for _ in range(length):
                (g, _), (h, _) = random.sample(self.move_pairs, 2)
                m = g @ h @ g.inverse() @ h.inverse()  # ghg⁻¹h⁻¹
                child_state = m.act(child_state)

        child = HybridAgent(child_state, self.env, mutate_length=length)
        child.species_id = self.species_id  # 遗传物种标签
        child.pos = self.pos + 0.4 * np.random.randn(2)

        child.resource = 0.5 * self.resource
        self.resource *= 0.5
        # 部分记忆继承
        child.memory = deque(list(self.memory)[:6], maxlen=10)
        child.memory.append(child_state)
        self.children.append(child)
        self.interact_count[2] += 1  # 繁殖计数
        return child

    def vector(self):
        "behavior:行为向量,策略签名"
        energy_rate = self.energy_delta_cum / max(1, self.age)
        entropy_rate = self.entropy_delta_cum / max(1, self.age)
        resource_rate = self.resource / max(1, self.age)
        mem_drift = 1.0 - self.consistency_val  # 轨道偏移量
        return np.array([
            # --- 状态 ---
            self.plan_entropy_val,
            self.purity_val,
            self.fidelity_val,
            self.divergence_val,

            self.move_entropy_val,
            self.curvature_val,
            self.chaos_val,
            self.alignment,
            # --- 行为 ---
            energy_rate,  # 运动强度/ 动力学速度
            entropy_rate,  # 认知活跃度/ 探索程度
            mem_drift,  # 轨道稳定性/行为一致性
            resource_rate,
            self.confidence,
            # self.stats['move_probs_mean'],  # 行为模式
            # self.stats['move_probs_std'],
        ])


class HybridSimulation:
    """
    Hybrid 双态人工生命系统（平衡秩序-混沌）
    微观	群作用（Cubie）	真实动力
    中观	slow manifold	结构压缩
    宏观	density / entropy	统计行为
    agent:
        rho → rho_plan
        rho_mem → 调参数
        ↓
    interact_kernel → 局部传播

    simulation:
        agents → rho_collective（全局抽象）

    rho_collective:
        → dream（生成新结构）
        → 弱反馈 → rho_mem

    最终：
        memory → policy → action
    """

    def __init__(self, env: Environment, n_agents=25):
        self.env = env
        self.env.set_targets()

        self.agents = [
            HybridAgent(CubieBase.generate_cubie(length=random.randint(3, 12)), env)
            for _ in range(n_agents)
        ]
        self.rho_collective = None  # 全局共享记忆场,群体共同的“潜意识”
        self.history = []
        self.population_history = []
        self.stats = {'n_agents': len(self.agents), 'n_targets': len(env.targets), 'n_species': 0,
                      'birth_cum': 0, 'death_cum': 0, "resource_mean": 1.0, 'step_time': 0.0, 'P': 0, 'T': 1.0}
        # self.env.context = self.stats  # 环境上下文共享统计数据
        self.move_counter = np.zeros(len(self.env.move_pairs), dtype=int)
        self.record()

    def record(self):
        if not self.agents:
            return
        zs = np.stack([a.z for a in self.agents])
        self.population_history.append(zs)
        self.env.update_context(**self.stats)

    def step(self, T=1 / 9, L=1, S: int = 10, K=300, k_neighbors=10):
        # 1. 所有 agent 演化
        import time
        t0 = time.perf_counter()
        for i, a in enumerate(self.agents):
            neighbors = [b for b in self.agents if a is not b]
            neighbors = random.sample(neighbors, min(k_neighbors, len(neighbors))) if neighbors else None
            a.step(T, L=L, S=S, neighbors=neighbors)

        self.assign_species(eps=0.5, min_samples=3)  # 每步重新分配物种 ID，基于当前的 z 向量聚类

        self.interaction(k_neighbors=k_neighbors)  # 2. 邻居交互（模仿 + 捕食）

        # 3. 繁殖
        self.reproduce(K)

        # 4. 死亡
        self.death(K)

        t1 = time.perf_counter()
        self.stats['step_time'] = t1 - t0

        # 5. 记录
        self.record()

    def behavior_dbscan(self, eps=0.3, min_samples=4, sig_samples=6):
        """DBSCAN 的前提是：存在 density cluster（密度簇）"""
        n = len(self.agents)
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        # 预计算 signature
        sigs = [self.env.model.chaos_signature(a.z, sig_samples) for a in self.agents]

        def region_query(i):
            return [j for j in range(n) if np.linalg.norm(sigs[i] - sigs[j]) < eps]

        def expand(i, neighbors):
            nonlocal cluster_id
            labels[i] = cluster_id
            k = 0
            while k < len(neighbors):
                j = neighbors[k]
                if not visited[j]:
                    visited[j] = True
                    nj = region_query(j)
                    if len(nj) >= min_samples:
                        neighbors.extend(nj)
                if labels[j] == -1:
                    labels[j] = cluster_id
                k += 1

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neigh = region_query(i)
            if len(neigh) < min_samples:
                labels[i] = -1
            else:
                expand(i, neigh)
                cluster_id += 1

        return labels

    def assign_species(self, eps=0.5, min_samples=4):
        """
        cluster 不是连续分布,而是离散轨道 ,
        慢流形上的几何已经塌成一层壳，导致 DBSCAN 无法形成 species,direction 有信息:cosine distance
        """

        # from sklearn.cluster import DBSCAN
        # Z = np.stack([a.z.real for a in self.agents])
        # Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        # dist_matrix = 1 - np.dot(Z, Z.T)
        # # 先按壳层分组
        # levels = np.array([a.level for a in self.agents]) #, a.preference
        # species = dbscan(Z, eps=0.8, min_samples=min_samples)

        # species = self.behavior_dbscan(eps=eps, min_samples=min_samples)
        def phase_signature(agent):
            """基于 phase behavior 分布的 signature,反映行为模式的相似性"""
            total = agent.phase_count.sum()
            return np.array(agent.phase_count) / (total + 1e-8) if total > 0 else np.zeros(4)

        # X = [a.rho_b.flatten().real for a in self.agents]
        # X = np.stack([a.preference for a in agents])
        X = np.stack([np.concatenate([a.vector(), phase_signature(a)]) for a in self.agents])
        # X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        # X[:, :7] = (X[:, :7] - X[:, :7].mean(0)) / (X[:, :7].std(0) + 1e-8)
        species = dbscan(X, eps=eps, min_samples=min_samples)

        for a, s in zip(self.agents, species):
            a.species_id = s

        self.stats['n_species'] = len(set(s for s in species if s >= 0))
        return species

    def update_collective(self, agents):
        """每一步更新集体大脑"""
        if len(agents) < 2:
            return
        rhos = [a.rho_mem for a in agents]
        weights = []
        for a in agents:
            w = (
                    0.5 * a.fidelity_val +
                    0.3 * (1 - a.plan_entropy_val) +
                    0.2 * a.purity_val
            )
            weights.append(w)

        weights = np.array(weights)
        weights /= weights.sum() + 1e-8

        # 加权平均,收集当前所有 agent 的 rho_mem 贡献
        contrib = sum(w * r for w, r in zip(weights, rhos))
        contrib /= np.trace(contrib) + 1e-12

        # --- 2️⃣ 初始化 ---
        if self.rho_collective is None:
            self.rho_collective = contrib.copy()
            return

        # --- 3️⃣ 动态融合率（关键）---
        avg_entropy = np.mean([a.plan_entropy_val for a in agents])
        diversity = np.mean([
            np.linalg.norm(a.z - b.z)
            for a in agents for b in agents if a is not b
        ])

        alpha = 0.08 + 0.12 * avg_entropy + 0.05 * np.tanh(diversity)

        # --- 4️⃣ EMA 更新,弱融合进集体记忆 ---
        self.rho_collective = (1 - alpha) * self.rho_collective + alpha * contrib

        # --- 5️⃣ anti-collapse（关键！）---
        dim = self.rho_collective.shape[0]
        I = np.eye(dim) / dim

        mix = 0.02 + 0.05 * (1 - avg_entropy)
        self.rho_collective = (1 - mix) * self.rho_collective + mix * I

        # --- 6️⃣ normalize ---
        self.rho_collective /= np.trace(self.rho_collective) + 1e-12

        # --- 7️⃣ 记录 entropy ---
        self.stats['collective_entropy'] = von_neumann_entropy(self.rho_collective)

    def collective_dream(self, steps=3, T=0.9):
        """ 触发集体 dream,越“不稳定的人”越受群体影响"""
        if self.rho_collective is None:
            return None

        rho = self.rho_collective.copy()

        for _ in range(steps):
            # ---  policy:在集体记忆空间进行轻微演化 ---
            f = self.model.move_scores_from_rho(rho)
            p = softmax(f / T)

            idx = np.random.choice(len(p), p=p)
            _, U = self.env.move_pairs[idx]

            rho = U @ rho @ U.conj().T
            rho /= np.trace(rho) + 1e-12

        # 梦境回馈给所有 agent（弱影响）
        for a in self.agents:  # 需要把 agents 传进来或全局持有
            # align = np.real(np.trace(a.rho_plan @ rho))
            strength = 0.05 * (1 - a.fidelity_val)
            a.rho_mem = (1 - strength) * a.rho_mem + strength * rho
            a.rho_mem /= np.trace(a.rho_mem) + 1e-12

        return rho

    def interaction(self, k_neighbors=8, slow_radius=5.0, pos_radius=4.0):
        """
        Top-K 邻居交互,所有 agent 状态更新后再进行邻居交互
        k: 每个 agent 最多交互的邻居数量
        slow_radius: 慢空间语义距离阈值
        pos_radius: 物理位置距离阈值
        """
        if len(self.agents) < 2:
            return

        n_species = self.stats['n_species']
        for a in self.agents:
            candidates = []
            for b in self.agents:
                if a is b:
                    continue
                slow_d = np.linalg.norm(a.z - b.z)
                pos_d = np.linalg.norm(a.pos - b.pos)
                # 只考虑在慢空间或物理空间上接近的
                if slow_d < slow_radius or pos_d < pos_radius:
                    score = -slow_d * 0.6 - pos_d * 0.4  # 综合分数：慢空间距离更重要
                    candidates.append((score, b, slow_d, pos_d))

            # 选 Top-K
            candidates.sort(key=lambda x: x[0], reverse=True)  # 分数越高越优先
            selected = candidates[:k_neighbors]
            neighbors = [b for _, b, *_ in selected]

            if not neighbors:
                continue

            values = [{"slow_d": item[2], "pos_d": item[3]} for item in selected]
            diversity = np.mean([item[2] for item in selected]) if selected else 2.0 * np.pi
            a.stats['diversity'] = diversity
            # 执行交互
            if n_species <= 3:
                # 只负责和给定的邻居列表交互（慢空间模仿 + 量子捕食）
                for b, d in zip(neighbors, values or [{}] * len(neighbors)):
                    if a is b: continue
                    directed_d = np.dot(b.z - a.z, a.preference).real  # 方向性判断
                    d['directed_d'] = directed_d
                    slow_d = d.get("slow_d", np.linalg.norm(a.z - b.z))
                    pos_d = d.get("pos_d", np.linalg.norm(a.pos - b.pos))
                    if a.species_id == b.species_id:  # 同族：学习
                        # 慢空间吸引/排斥
                        interaction_strength = 1.0 - (slow_d / (diversity + 1e-8))
                        if slow_d < 2 * np.pi and random.random() < interaction_strength:
                            a.imitate(b, data=d)
                    elif a.target_id == b.target_id:  # 异族：竞争
                        predator_prob = np.exp(-len(neighbors) / 10.0) * 1 / (1 + n_species)  # species 越少越小,越拥挤越少打
                        if pos_d < 5 and random.random() < predator_prob:
                            a.predator(b, data=d)
                    elif a.level < b.level:
                        pass
            else:
                a.interact_kernel(neighbors, values)

    def reproduce(self, K=300):
        new_agents = []
        N = len(self.agents)
        density = N / K
        fitness = [0.5 * a.resource + 0.4 * a.purity_val + 0.3 * np.tanh(
            a.energy / 30) + 0.2 * a.alignment + 0.1 * np.exp(-((a.age - 50) ** 2) / 3600) for a in self.agents]
        threshold = np.percentile(fitness, 80)
        for a, s in zip(self.agents, fitness):
            birth_prob = (0.05 + 0.1 * s) * np.exp(- density)
            if a.resource > 1.0 and s > threshold and random.random() < birth_prob:
                child = a.reproduce()
                if child:
                    new_agents.append(child)
        self.agents.extend(new_agents)
        born = len(new_agents)
        self.stats["n_agents"] = len(self.agents)
        self.stats['birth_cum'] += born
        return born

    def death(self, K=300):
        N = len(self.agents)
        if N > K:
            resource_mean = self.stats.get("resource_mean", 1.0)
            life_activity_mean = self.stats.get("life_activity_mean", 1.0)
            base_prob = np.clip(0.2 + 0.8 * np.tanh((resource_mean + life_activity_mean) / 5), 0, 1)
            density = N / K  # 高密度惩罚
            survival_prob = base_prob * np.exp(-max(0, density - 1))
            self.agents = [a for a in self.agents
                           if a.purity_val > 0.15 and a.consistency_val > 0.05 and a.divergence_val < 1.0
                           and a.resource > -4.0 and a.energy < 50.0 and a.age < 500
                           and random.random() < survival_prob]
        else:
            self.agents = [a for a in self.agents
                           if a.purity_val > 0.1 and a.consistency_val > 0.05 and a.resource > -5.0 and a.age < 500]
        self.stats["n_agents"] = len(self.agents)
        died = N - self.stats["n_agents"]
        self.stats['death_cum'] += died
        return died

    def observe(self):
        """全部掉到“高能壳层”上了,慢空间的几何结构已经塌成一层壳了,所以只能统计行为特征了"""
        if not self.agents:
            obs = {
                "purity_mean": 0.0,
                "energy_mean": 0.0,
                "chaos_mean": 0.0,
                "n_agents": 0,
            }
            self.history.append(obs)
            return obs

        ps = [a.purity_val for a in self.agents]
        es = [a.energy for a in self.agents]
        cs = [a.chaos_val for a in self.agents]
        rs = [a.resource for a in self.agents]

        ts = [a.stats['T'] for a in self.agents]
        ts2 = [a.stats['T_phase'] for a in self.agents]
        fs = [a.fidelity_val for a in self.agents]
        ds = [a.divergence_val for a in self.agents]

        curvature = [a.curvature_val for a in self.agents]
        confidence = [a.confidence for a in self.agents]
        consistency = [a.consistency_val for a in self.agents]
        entropy = [a.plan_entropy_val for a in self.agents]
        move_entropy = [a.move_entropy_val for a in self.agents]
        entropy_deltas = [a.entropy_delta_cum for a in self.agents]
        energy_deltas = [a.energy_delta_cum / (a.age + 1) for a in self.agents]
        age = [a.age for a in self.agents]

        alignment = [a.alignment for a in self.agents]
        prefs = np.std(np.stack([a.preference for a in self.agents]), axis=0)

        phase_counts = np.sum([a.phase_count for a in self.agents], axis=0)
        phase_ratios = phase_counts / (phase_counts.sum() + 1e-8)
        interact_counts = np.array([a.interact_count / max(1, a.age) for a in self.agents])
        interact_ratios = interact_counts.mean(axis=0)
        hit_counts = [a.interact_count[5] for a in self.agents]

        diversity = [a.stats.get('diversity', 0.0) for a in self.agents]
        move_probs_std = [a.stats.get('move_probs_std', 0.0) for a in self.agents]
        bb_like = [a.purity_val > 0.9 and a.plan_entropy_val < 0.3 for a in self.agents]

        move_ids = [a.stats['move_id'] for a in self.agents if a.stats['move_id'] != -1]
        ids, counts = np.unique(move_ids, return_counts=True)
        self.move_counter[ids] += counts
        obs = {
            "purity_mean": np.mean(ps),
            "fidelity_mean": np.mean(fs),
            "consistency_mean": np.mean(consistency),
            "divergence_mean": np.mean(ds),

            "plan_entropy_mean": np.mean(entropy),
            "move_entropy_mean": np.mean(move_entropy),
            "energy_mean": np.mean(es),
            "chaos_mean": np.mean(cs),  # 0.275
            "curvature_mean": np.mean(curvature),

            "resource_mean": np.mean(rs),
            "age_mean": np.mean(age),  # 23
            "temperature_mean": np.mean(ts),
            "temperature_phase_mean": np.mean(ts2),
            "alignment_mean": np.mean(alignment),
            "confidence_mean": np.mean(confidence),
            'diversity_mean': np.mean(diversity),  # 邻居差异性

            "resource_std": np.std(rs),
            "resource_max": np.max(rs),
            "age_max": np.max(age),
            "energy_std": np.std(es),
            "energy_min": np.min(es),
            "entropy_max": np.max(entropy),
            "curvature_max": np.max(curvature),
            "purity_min": np.min(ps),
            "fidelity_min": np.min(fs),
            'consistency_min': np.min(consistency),
            "divergence_max": np.max(ds),
            'confidence_max': np.max(confidence),
            "temperature_max": np.max(ts),
            "pref_diversity": np.mean(prefs),  # preference 向量多样性

            'exploit_ratio': phase_ratios[0],
            'revert_ratio': phase_ratios[1],
            'explore_ratio': phase_ratios[2],
            'natura_ratio': phase_ratios[3],

            'imitate_avg': interact_ratios[0],
            'predator_avg': interact_ratios[1],
            'reproduce_avg': interact_ratios[2],
            'emulate_avg': interact_ratios[3],
            'hit_count': np.sum(hit_counts),
            'bb_lifetime_avg': interact_ratios[4],  # 0.06~0.314 虚假记忆快速被现实修正
            "bb_fraction": float(np.mean(bb_like)),  # boltzmann_brain
            "bb_count": np.sum(bb_like),

            'move_probs_std_mean': float(np.mean(move_probs_std)),
            'life_activity_mean': np.mean(entropy_deltas),  # 生命长度/活跃度(认知层面的变化)
            'cumulative_energy_delta_mean': np.mean(energy_deltas),
        }
        self.stats["resource_mean"] = obs["resource_mean"]
        self.stats["age_mean"] = obs["age_mean"]
        self.stats["life_activity_mean"] = obs["life_activity_mean"]
        obs.update(self.stats)  # 添加统计
        self.history.append(obs)
        return obs

    def save(self, filename=None):
        import pickle
        if filename is None:
            filename = os.path.join(DATA_DIR, "simulation_data.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)

    def evolve(self, steps=300, max_k=300, print_every=10, base_temp=1.0 / 9.0):
        K = len(self.agents)
        P = 0
        targets = self.env.base_targets(type=1)
        prev_short_osc = 0.0

        for t in range(steps):
            a, b = divmod(t, 10)
            if b == 0 and K < max_k:
                K += 8 * (a + 1)
                self.stats['K'] = K

            T = base_temp
            if t < 3:  # 初期强探索
                T = 4.0 * base_temp
                P = 1
            elif t < self.env.model.Tf:
                T = 3.0 * base_temp
                P = 2
            elif t < 3 * self.env.model.Tf:
                P = 3
            else:  # 周期性温度波动
                short_osc = 2.0 / 3. * np.sin(2 * np.pi * t / 32)
                long_osc = 1.0 / 3 * np.sin(2 * np.pi * t / 100)
                temp_osc = short_osc + long_osc
                T = 2.0 * base_temp * (1.0 + temp_osc * (1.0 + 3 * long_osc))  # 双频扰动 0.25~3
                if (prev_short_osc < 0) and (short_osc >= 0):  # 接近零点/ 接近波峰
                    print(
                        f"[t={t}] 温度峰值扰动！新阶段 Phase {P},T={T:.3f} (short_osc={short_osc:.3f}, long_osc={long_osc:.3f})")
                    P += 1
                prev_short_osc = short_osc

            S = 10
            if self.stats.get('P', 0) != P:
                if P <= 3:
                    new_targets = targets.pop()
                    self.env.add_target(new_targets)
                elif len(self.env.targets) < 30:
                    new_targets = self.env.generate_diff_target()  # 每次温度峰值时生成新目标
                    self.env.add_target(new_targets)  # 添加新目标，增加环境复杂度

                self.stats['P'] = P
                self.stats['n_targets'] = self.env.context['n_targets']
                S = 1

            self.stats['T'] = T

            self.step(T=T, S=S, K=K)
            obs = self.observe()

            if obs["n_agents"] == 0:
                print(f"[t={t}] 种群灭绝！自动重生 10 个新个体...")
                self.agents = [
                    HybridAgent(CubieBase.generate_cubie(length=random.randint(0, 8)), self.env)
                    for _ in range(10)
                ]
                self.record()
                obs = self.observe()

            if t % print_every == 0:
                obs_display = {k: f"{float(v):.3f}" for k, v in obs.items()}
                phases = [a.stats['phase'] for a in self.agents]
                move_ids = [a.stats['move_id'] for a in self.agents if a.stats['move_id'] != -1]
                levels = [a.level for a in self.agents]
                target_ids = [a.target_id for a in self.agents]
                obs_distribution = {
                    "phase": {int(k): int(v) for k, v in zip(*np.unique(phases, return_counts=True)) if k >= 0},
                    "level": {int(k): int(v) for k, v in zip(*np.unique(levels, return_counts=True))},
                    "target": {int(k): int(v) for k, v in zip(*np.unique(target_ids, return_counts=True))},
                    "move": {int(k): int(v) for k, v in zip(*np.unique(move_ids, return_counts=True))}
                }
                print(f"[{t:3d}] obs={obs_display} \ndistribution={obs_distribution}")
                print(self.move_counter)

    def compute_flow(self):
        flows = []
        for a in self.agents:
            z1 = a.z.copy()
            initial_state = a.gm.re_act(a.state)
            z0 = self.env.model.project(initial_state.vector)
            flows.append((z1, z1 - z0))

        return flows

    def plot_flow(self):
        flows = self.compute_flow()

        Z = np.array([f[0].real for f in flows])  # 当前位置
        V = np.array([f[1].real for f in flows])

        R = np.random.randn(Z.shape[1], 2)
        R /= np.linalg.norm(R, axis=0) + 1e-8
        Z2 = Z @ R
        V2 = V @ R

        plt.figure(figsize=(12, 8))
        plt.quiver(Z2[:, 0], Z2[:, 1], V2[:, 0], V2[:, 1],
                   angles='xy', scale_units='xy', scale=1, alpha=0.5)

        plt.title("Population Flow Field,From Birth to Current")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_population(self, n_last=60):
        if len(self.population_history) < n_last:
            return
        recent = self.population_history[-n_last:]
        plt.figure(figsize=(12, 8))
        for i, Z in enumerate(recent):
            Z_plot = self.env.model.evolve(Z, T=6)  # diffusion scaling
            # R = np.random.randn(Z_plot.shape[1], 2)
            # Z_plot = Z_plot @ R
            alpha = 0.1 + 0.9 * (i / (n_last - 1))
            plt.scatter(Z_plot[:, 0].real, Z_plot[:, 1].real, s=8, alpha=alpha, c='purple')
        plt.xlabel('Slow PC1')
        plt.ylabel('Slow PC2')
        plt.title('Hybrid Population on Slow Manifold (last 60 gens)')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_metrics(self):
        steps = np.arange(len(self.history))
        purity = [h['purity_mean'] for h in self.history]
        energy = [h['energy_mean'] for h in self.history]
        chaos = [h['chaos_mean'] for h in self.history]

        fidelity = [h['fidelity_mean'] for h in self.history]
        plan_entropy = [h['plan_entropy_mean'] for h in self.history]
        consistency = [h['consistency_mean'] for h in self.history]

        divergence = [h['divergence_mean'] for h in self.history]
        resource = [h['resource_mean'] for h in self.history]

        n = [h['n_agents'] for h in self.history]

        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        axs[0, 0].plot(steps, purity, 'b-', label='Purity')
        axs[0, 1].plot(steps, energy, 'r-', label='Classical Energy')
        axs[0, 2].plot(steps, chaos, 'g-', label='Commutator Chaos')

        axs[1, 0].plot(steps, fidelity, 'c-', label='Fidelity')
        axs[1, 1].plot(steps, plan_entropy, 'brown', label='Plan Entropy')
        axs[1, 2].plot(steps, consistency, 'y-', label='Consistency Mean')

        axs[2, 0].plot(steps, divergence, 'k-', label='Divergence')
        axs[2, 1].plot(steps, resource, 'b-', label='Resource')
        axs[2, 2].plot(steps, n, 'm-', label='Population')
        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_additional_metrics(self):
        """其他 9 个指标的演化图"""
        steps = np.arange(len(self.history))

        alignment = [h.get('alignment_mean', 0) for h in self.history]
        temperature = [h.get('temperature_mean', 0) for h in self.history]
        diversity = [h.get('diversity_mean', 0) for h in self.history]

        curvature = [h.get('curvature_mean', 0) for h in self.history]
        confidence = [h['confidence_mean'] for h in self.history]
        move_probs_std = [h.get('move_probs_std_mean', 0) for h in self.history]

        bb_fraction = [h.get('bb_fraction', 0) for h in self.history]
        species = [h.get('n_species', 0) for h in self.history]
        pref_diversity = [h.get('pref_diversity', 0) for h in self.history]

        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        axs[0, 0].plot(steps, alignment, 'm-', label='Alignment')
        axs[0, 1].plot(steps, temperature, 'orange', label='Temperature Mean')
        axs[0, 2].plot(steps, diversity, 'y-', label='Diversity Mean')

        axs[1, 0].plot(steps, curvature, 'purple', label='Curvature')
        axs[1, 1].plot(steps, confidence, 'navy', label='Confidence')
        axs[1, 2].plot(steps, move_probs_std, 'teal', label='Move Probabilities Std')

        axs[2, 0].plot(steps, bb_fraction, 'k-', label='Boltzmann Brain Fraction')
        axs[2, 1].plot(steps, species, 'gold', label='Species Count')
        axs[2, 2].plot(steps, pref_diversity, 'brown', label='Preference Diversity')

        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.suptitle('Additional Metrics Evolution', fontsize=16, y=1.0)
        plt.tight_layout()
        plt.show()

    def plot_extra_metrics(self):
        """绘制额外9个指标的演化图"""
        steps = np.arange(len(self.history))

        resource_std = [h['resource_std'] for h in self.history]
        curvature_max = [h.get('curvature_max', 0) for h in self.history]
        entropy_max = [h.get('entropy_max', 0) for h in self.history]

        life_activity = [h.get('life_activity_mean', 0) for h in self.history]  # entropy_deltas
        cumulative_energy_delta = [h.get('cumulative_energy_delta_mean', 0) for h in self.history]  # energy_deltas
        age = [h.get('age_mean', 0) for h in self.history]

        death = [h.get('death_cum', 0) for h in self.history]
        birth = [h.get('birth_cum', 0) for h in self.history]

        hit_count = [h.get('hit_count', 0) for h in self.history]
        bb_count = [h.get('bb_count', 0) for h in self.history]

        fig, axs = plt.subplots(3, 3, figsize=(18, 12))

        axs[0, 0].plot(steps, curvature_max, 'purple', label='Curvature Max')
        axs[0, 1].plot(steps, entropy_max, 'darkgreen', label='Plan Entropy Max')
        axs[0, 2].plot(steps, resource_std, 'teal', label='Resource Std')

        axs[1, 0].plot(steps, life_activity, 'magenta', label='Life Activity (Entropy Deltas)')
        axs[1, 1].plot(steps, cumulative_energy_delta, 'orange', label='Cumulative Energy Delta')
        axs[1, 2].plot(steps, age, 'cyan', label='Age Mean')

        axs[2, 0].plot(steps, death, 'brown', label='Death Cumulative')
        axs[2, 0].plot(steps, birth, 'navy', label='Birth Cumulative')
        axs[2, 1].plot(steps, hit_count, 'red', label='Hit Target Count')
        axs[2, 2].plot(steps, bb_count, 'darkblue', label='Boltzmann Brain Count')

        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.suptitle('Extra Metrics Evolution (New Variables)', fontsize=16, y=1.0)
        plt.tight_layout()
        plt.show()

    def plot_phase(self):
        """绘制四个 phase 的比例演化图（堆叠面积图）"""

        steps = np.arange(len(self.history))

        exploit = [h.get('exploit_ratio', 0) for h in self.history]
        revert = [h.get('revert_ratio', 0) for h in self.history]
        explore = [h.get('explore_ratio', 0) for h in self.history]
        natura = [h.get('natura_ratio', 0) for h in self.history]

        plt.figure(figsize=(12, 7))

        plt.stackplot(steps,
                      exploit, revert, explore, natura,
                      labels=['Exploit (利用)', 'Revert (回退)',
                              'Explore (探索)', 'Natura (自然/变异)'],
                      colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                      alpha=0.85)

        plt.xlabel('Evolution Steps')
        plt.ylabel('Phase Ratio')
        plt.title('Phase Distribution Over Time (Stacked Area)')
        plt.legend(loc='upper left', frameon=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_interaction(self, n_last=100):
        """画交互行为平均次数的变化曲线"""

        recent = self.history[-n_last:]
        steps = list(range(len(recent)))

        # 提取各项
        imitate = [h.get('imitate_avg', 0) for h in recent]
        predator = [h.get('predator_avg', 0) for h in recent]
        reproduce = [h.get('reproduce_avg', 0) for h in recent]
        emulate = [h.get('emulate_avg', 0) for h in recent]
        # crisis = [h.get('crisis_avg', 0) for h in recent]
        bb_lifetime = [h.get('bb_lifetime_avg', 0) for h in recent]

        plt.figure(figsize=(12, 8))

        plt.plot(steps, imitate, label='Imitate', linewidth=2.5, color='tab:blue')
        plt.plot(steps, predator, label='Predator', linewidth=2.5, color='tab:red')
        plt.plot(steps, reproduce, label='Reproduce', linewidth=2.5, color='tab:green')
        plt.plot(steps, emulate, label='Emulate', linewidth=2.5, color='tab:orange')
        # plt.plot(steps, crisis, label='Crisis', linewidth=2.5, color='tab:purple')
        plt.plot(steps, bb_lifetime, label='BB Lifetime', linewidth=2.5, color='tab:cyan', linestyle='--')

        plt.title("Average Interaction Counts per Agent over Time")
        plt.xlabel("Steps")
        plt.ylabel("Average Interactions per Step")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_agent_rho(self, rho_type='rho_plan'):
        """
        找出 age 最大的 Agent，并画出它的 rho 热力图
        rho_type 可选: 'rho', 'rho_plan', 'rho_mem'
        """
        import seaborn as sns
        # 找出 age 最大的 Agent
        oldest_agent = max(self.agents, key=lambda a: a.age)
        # 额外打印一些信息
        print(f"Agent (age={oldest_agent.age})")
        print(f"  Purity (rho_plan): {oldest_agent.purity_val:.4f}")
        print(f"  Fidelity: {oldest_agent.fidelity_val:.4f}")
        print(f"  Divergence: {oldest_agent.divergence_val:.4f}")
        print(f"  Entropy: {oldest_agent.plan_entropy_val:.4f}")

        # 获取对应的 rho
        if rho_type == 'rho_plan':
            rho = oldest_agent.rho_plan
            rho_name = 'rho_plan (规划态)'
        elif rho_type == 'rho_mem':
            rho = oldest_agent.rho_mem
            rho_name = 'rho_mem (长期记忆)'
        else:
            rho = oldest_agent.rho
            rho_name = 'rho'

        # 转为实部矩阵（热力图通常看实部或绝对值）
        rho_real = np.real(rho)
        mask = np.triu(np.ones_like(rho_real, dtype=bool), k=1)
        plt.figure(figsize=(10, 8))
        # sns.heatmap(np.abs(rho), cmap='viridis', annot=False)
        sns.heatmap(rho_real, mask=mask,
                    cmap='RdBu_r',
                    center=0,
                    annot=False,
                    cbar_kws={'label': 'Real Part'})

        plt.title(f"Agent rho Age = {oldest_agent.age} | {rho_name} | Purity = {oldest_agent.purity_val:.4f}")
        plt.xlabel("Dimension Index")
        plt.ylabel("Dimension Index")
        plt.tight_layout()
        plt.show()


def main():
    env = Environment(model=SlowDynamics(n=N_GENERATORS))
    sim = HybridSimulation(env=env, n_agents=30)
    sim.evolve(steps=123, print_every=10, max_k=400)
    sim.plot_population()
    sim.plot_metrics()
    sim.plot_additional_metrics()
    sim.plot_extra_metrics()
    sim.plot_phase()
    sim.plot_interaction()
    sim.plot_agent_rho()
    sim.plot_flow()
    sim.save()


if __name__ == '__main__':
    main()
