import numpy as np
import random

from rime.base import class_status
from rime.cubie import CubieMove, N_GENERATORS
from rime.cubieoperator import CubieSpectralOperator
from rime.helpers import cosine_distance


class SlowDynamics(CubieSpectralOperator):
    """Slow-manifold dynamics on the Rubik's Cube group representation (228-dim).

    Inherits from CubieSpectralOperator (the canonical spectral engine). Provides
    slow-manifold projection, generator action in the slow subspace, commutator
    infrastructure, and Lie curvature diagnostics.

    Post-ρ-fix canonical data (see paper_data.md):
      6 layers: V₁(20), V₈/₉(2), V₇/₉(39), V₂/₃(26), V₅/₉(106), V₁/₃(35)
      Slow (λ ≥ 2/3): V₁ + V₈/₉ + V₇/₉ + V₂/₃ = 87-dim
      Hub (λ = 5/9): V₅/₉ = 106-dim — transport hub, separately accessible
      Fast (λ < 5/9): V₁/₃ = 35-dim
      A_18 = (12 QT_all + 6 HT_all) / 18, ‖[QT⁰,QT¹]‖ = 2.92 (ep=2.74, co=0.61, eo=0.79)
      Transport: star topology centered on V₅/₉ hub; V₁ isolated; all slow phases
      interconnect through V₅/₉. Cross-block T7 pairs require composition.

    Three dynamical regimes (empirically validated):
      Far (‖Δz‖ ≥ 4.0): gradient-dominant, near-continuous, 2-cycle rate ~1.5%
      Mid (1.5 ≤ ‖Δz‖ < 4.0): radial-tangential competition, anisotropy rises
      Near (‖Δz‖ < 1.5): discrete orbit dominance, 2-cycle rate ~98%, iso-distance shell

    Theory connection (see Paper II/III):
      - K_αβ transport tensor governs single-step sector transitions
      - κ_d hierarchy governs multi-step accessibility (κ₀=direct, κ₁=curvature, T7=composition)
      - The near-regime 2-cycle is a symptom of Lie-algebraic freezing (Paper III Theorem 2)

    Usage:
        model = SlowDynamics()
        z = model.project(state.vector)      # → 87-dim slow subspace
        h = model.project_hub(x)             # → 106-dim V₅/₉ hub (transport router)
        plan = model.phase_path_plan(z, goal)  # routes through V₅/₉ hub
    """

    def __init__(self, n: int = N_GENERATORS, threshold: float = 2 / 3, tol=1e-6, eps=1e-6,
                 generators=None):
        """
        Parameters:
        - n: generator count
        - threshold: slow subspace threshold (default 2/3)
        - tol: numerical tolerance
        """
        super().__init__(n=n, generators=generators, tol=tol)

        # Algebra dimension via SVD
        rho_gen = [rho for _, rho, *_ in self.rho_moves.values()]
        _, s, _ = np.linalg.svd(np.stack([A.reshape(-1) for A in rho_gen]), full_matrices=False)
        self.dim_algebra = np.sum(s > tol)

        # --- invariants ---
        mask_const = np.abs(self.w - 1.0) < tol  # 提取守恒子空间,最大特征值必然是 1
        self.V_const = self.V[:, mask_const]

        # --- slow modes ---
        self.lambda_slow = self.w[~mask_const].max()  # 7 / 9  (λ₂, 次大特征值)
        mask_slow = (self.w >= threshold - tol) & (~mask_const)
        if not np.any(mask_slow):
            mask_slow = np.abs(self.w - self.lambda_slow) < tol
        self.V_slow = self.V[:, mask_slow]
        self.w_slow = self.w[mask_slow]
        self.dim_slow = len(self.w_slow)  # override parent: exclude const
        self.scale_l2 = np.sqrt(2) * 0.5 * np.sqrt(self.dim_slow)
        self.V_keep = np.concatenate([self.V_const, self.V_slow], axis=1)

        # --- V₅/₉ hub (separate from slow subspace, used for transport routing) ---
        mask_hub = np.abs(self.w - 5/9) < tol
        self.V_hub = self.V[:, mask_hub] if np.any(mask_hub) else None
        self.dim_hub = int(np.sum(mask_hub))

        # Alias for backward compat
        self.A_micro = self.A

        # Pre-cache slow-space compressed operators 预缓存慢空间表示
        self.rho_slow = {k: (mv, self.V_slow.T @ rho @ self.V_slow)
                         for k, (mv, rho, *_) in self.rho_moves.items()}  # (100,100) 约化矩阵

        slow_moves = list(self.rho_slow.values())
        self.U = np.stack([Ug for _, Ug in slow_moves])  # (n, d, d) Uz = U_tensor @ z
        I = np.eye(self.V_slow.shape[1])
        self.D = self.U - I  # (n, d, d) D_ops:Ug - I 代替：Ug @ z - z
        self.L = np.diag(np.log(self.w_slow + 1e-12))

        self.C_pairs = {}  # [Ug, Uh]
        self.commutators = {}  # Uc group commutator（非线性）
        for i, (_, Ug) in enumerate(slow_moves):
            for j in range(i + 1, len(slow_moves)):
                _, Uh = slow_moves[j]
                self.C_pairs[(i, j)] = Ug @ Uh - Uh @ Ug
                self.commutators[(i, j)] = Ug @ Uh @ Ug.conj().T @ Uh.conj().T

        M = np.stack([Ug.reshape(-1) for _, Ug in slow_moves], axis=1)
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        self.dim_algebra_slow = np.sum(s > tol)

        A_block = self.V.T.conj() @ self.A_micro @ self.V
        assert np.isclose(np.trace(self.A_micro), np.trace(A_block), atol=1e-6), "trace conservation failed"

        mask_fast = self.w < threshold
        if np.any(mask_fast):
            proj_keep = self.V_keep @ self.V_keep.T.conj()
            proj_fast = np.eye(self.V.shape[0]) - proj_keep
            A_fast = proj_fast @ self.A_micro @ proj_fast
            eigvals_fast = np.linalg.eigvals(A_fast)
            self.rho_f = min(np.max(np.abs(eigvals_fast)), 1 - 1e-8)
            t_mix = np.log(1 / eps) / (-np.log(self.rho_f))  # 23.5043
            self.Tf = int(np.ceil(t_mix))  # 混合时间步数
            print(f"Fast layer spectral radius: {self.rho_f:.6f}, mixing time (eps=1e-6): steps -> Tf={self.Tf}")

    def mutate(self, rho, length: int = 4, p=None):
        """量子变异（commutator chain，保留结构）"""
        gen = list(self.rho_slow.values())
        indices = np.random.choice(len(gen), size=length, p=p)
        U_chain = np.eye(rho.shape[0], dtype=complex)
        g = CubieMove.identity()
        for i in indices:
            m, U = gen[i]
            U_chain = U @ U_chain  # 左乘
            g = g.compose(m)
        rho2 = U_chain @ rho @ U_chain.conj().T
        return rho2 / (np.trace(rho2) + 1e-12), g

    def exact(self, x, T) -> np.ndarray:
        """xT_exact 真实 T 步 vec= A_micro @ vec"""
        return np.linalg.matrix_power(self.A_micro, T) @ x  # (228,)

    def project(self, x):
        """投影到慢子空间 (87-dim, λ ≥ 2/3)."""
        return self.V_slow.T @ x

    def project_hub(self, x):
        """投影到 V₅/₉ hub (106-dim) — 传输路由层，独立于慢子空间."""
        if self.V_hub is None:
            return np.array([])
        return self.V_hub.T @ x

    def lift_hub(self, h):
        """从 V₅/₉ hub 坐标还原到全空间."""
        if self.V_hub is None:
            return np.zeros(self.V.shape[0])
        return self.V_hub @ h

    def project_move(self, rho_m):
        """慢空间表示,慢层压缩降维,投影变换,真实作用的投影作为参考
        比直接用 已经投影过的 U 更忠实于群作用的原始几何
        一个近似，无法完全忠实还原真实群作用在慢空间的效应
        project(s.vector)用真实作用后再投影,物理上最忠实
        """
        return self.V_slow.T @ rho_m @ self.V_slow  # (d,d) unitary/约化表示

    @staticmethod
    def relative_transform(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
        """
        计算慢空间中从 z0 到 z1 的最佳线性相对变换矩阵 M (z1 ≈ M @ z0)
        返回 M (76×76 矩阵),只能是近似，不是群同态
        """
        z0 = np.asarray(z0, dtype=complex).reshape(-1, 1)  # 初始慢空间坐标
        z1 = np.asarray(z1, dtype=complex).reshape(-1, 1)

        # 最小二乘求解 M z0 ≈ z1
        M = z1 @ np.linalg.pinv(z0)  # M = z1 @ z0⁺  (Moore-Penrose 伪逆)
        # z_pred = M @ z0
        # error = np.linalg.norm(z_pred.flatten() - z1)# 重建误差
        return M

    def lift(self, z):
        """xT_approx 从慢坐标还原高维状态：x ≈ V_slow z"""
        return self.V_slow @ z  # (228,) 还原回原空间,高维状态

    def evolve(self, z, T):
        """
        谱动力学层,保持慢模演化正确性,zT 微时间指数预测,在慢子空间做  A_micro^T x(t) = x_corner(t) + x_edge(t)
        """
        return (self.w_slow ** T) * z  # (d,) np.multiply(z, self.w_slow ** T) 在慢空间演化 预测 T 步 指数衰减

    def evolve_continuous(self, z, T):
        return np.diag(np.exp(self.L * T)) * z

    def group_action(self, m: CubieMove, z):
        """群元素 ρ(m) 在慢空间的作用,自动继承群乘法：ρ(gh) = ρ(g) ρ(h)"""
        rho_slow = self.project_move(m.rho())  # (d,d) 约化表示
        return rho_slow @ z  # (d,)

    def apply_move(self, key: tuple, z):
        """group_action key move 缓存"""
        _, rho_s = self.rho_slow[key]
        return rho_s @ z  # (d,)

    @staticmethod
    def l2_distance(z0, z1):
        """
        L2 是 backbone
        几何项只能 modulation
        """
        return np.linalg.norm(z0 - z1)

    @staticmethod
    def shell_level(z0, z1):
        """classify 分层能级结构,谱量子化的相空间，slow space 里存在离散轨道壳层,幅度量子化
        用 L2 距离对 slow manifold 做一个粗粒化分区
        np.sqrt(2) * 0.5 * np.sqrt(self.dim_slow)
        注意：
            • 不是严格能级（非量子化）,壳层（shell index）
            • 是 slow manifold 上的"壳层带（shell bands）"
            • 本质来源：group word length 在连续嵌入中的模糊投影

        Distance clusters correspond to projected group word-length:
        ~0        : solved / near identity
        ~1        : small perturbation
        ~3–4      : intermediate
        ~5–6      : mixing region
        """
        d = SlowDynamics.l2_distance(z0, z1)
        # cos_sim = 1 - cosine_distance(z0, z1)
        if d < 1.0:
            return 0, d
        elif d < 2.0:
            return 1, d
        elif d < 4.0:
            return 2, d
        elif d < 6.0:  # self.scale_l2
            return 3, d
        else:  # 高混沌 2*pi
            return 4, d

    def fast_energy(self, x):
        z_slow = self.project(x)
        x_rec = self.lift(z_slow)
        return np.linalg.norm(x - x_rec)

    def heuristic_with_confidence(self, x, y):
        z_delta = self.project(x - y)
        d = np.linalg.norm(z_delta)
        fast_residual = np.linalg.norm((x - y) - self.lift(z_delta))  # 可信度
        confidence = np.exp(-fast_residual)
        return d, confidence

    def heuristic(self, x, y, norm_l2=True):
        """
        计算 Representation-Aware 距离, spectral_diffusion_distance d(x,y) = || V_slow^T (x-y) ||
        Representation Discovery using Harmonic Analysis 系列明确提出用慢特征向量做 heuristic，忽略快模式，提升搜索效率
        input state.vector
        """
        delta = x - y
        z_delta = self.project(delta)  # (100,)
        if norm_l2:
            return np.linalg.norm(z_delta)  # 慢坐标欧氏距离,谱空间 diffusion distance
        return np.sum(np.abs(z_delta))  # np.sqrt(np.sum((np.abs(z_delta) ** 2)))

    def predict(self, state_vector, T: int = 1, const=False):
        """ diffusion map dynamics
        z = V_slowᵀ x
        z_t = λ^t z
        x ≈ V_slow z
        """
        z0 = self.project(state_vector)  # (100,) 投影到慢子空间
        zT = self.evolve(z0, T)  # 微时间演化
        xT_approx = self.lift(zT)
        if const:
            xT_approx += self.V_const @ (self.V_const.T @ state_vector)  # 保持守恒量，保证角棱 parity / sum 等
        return xT_approx.real  # (228,) 返回近似状态向量,np.real(zT)

    def predict_path(self, state_vector, moves: list[tuple] = None, micro_steps=0):
        """路径预测 慢坐标,混合动力学"""
        z = self.project(state_vector)
        if moves is not None:  # 离散群作用,应用 moves 序列
            for m in moves:
                z = self.apply_move(m, z)  # group_action
        if micro_steps > 0:  # 微时间指数预测,长演化
            z = self.evolve(z, micro_steps)
        return

    def behavior_distance(self, z0, z1, samples=5):
        """
        对同一组操作的响应是否相似，衡量慢子空间的行为区分能力,slow manifold 变成了一个"薄壳"
        ≈ 在一个高维球面上随机分布, 方向驱动的动力系统,距离 ≠ 差异,角度 = 差异
        """
        diffs = []
        for _, U in random.sample(list(self.rho_slow.values()), samples):
            za = U @ z0
            zb = U @ z1
            diffs.append(cosine_distance(za, zb))
            # angle_diff = cosine_distance(z_gh, z_hg)
        return np.mean(diffs)

    def build_slow_algebra_basis(self):
        """slow algebra ≈ span{ρ(g), [ρ(g),ρ(h)]}"""
        mats = [U for _, U in self.rho_slow.values()]
        mats += [C for C in self.C_pairs.values()]

        M = np.stack([A.reshape(-1) for A in mats], axis=1)
        U, s, _ = np.linalg.svd(M, full_matrices=False)

        rank = np.sum(s > self.tol)
        # U[:, :rank] is (d², rank); transpose so each row maps to one matrix
        return U[:, :rank].T.reshape(rank, self.dim_slow, self.dim_slow), s[:rank]

    def structure_constants(self):
        """Compute approximate Lie algebra structure constants"""
        basis, _ = self.build_slow_algebra_basis()
        k = len(basis)

        Cijk = np.zeros((k, k, k), dtype=complex)

        for i in range(k):
            for j in range(k):
                comm = basis[i] @ basis[j] - basis[j] @ basis[i]

                # 投影到基
                for l in range(k):
                    Cijk[i, j, l] = np.tensordot(comm.conj(), basis[l])

        return Cijk

    def curvature_ij(self, i, j, z, lie=True):
        """
        几何不稳定性
        分母过强,[Ug, Uh] 在 slow 空间被削弱,slow manifold 把李代数压平了
        C @ z = Ug @ (Uh @ z) - Uh @ (Ug @ z)
        D[i] @ z = Ug @ z - z
        num = || [Ug, Uh] z || = || C_gh z ||
        den = || (Ug - I) z || + || (Uh - I) z ||
        """
        if not lie:  # holonomy curvature
            Uc = self.commutators[(i, j)] if i < j else self.commutators[(j, i)].conj().T
            dz = Uc @ z - z  # 群交换子作用,几何塌缩
            return np.linalg.norm(dz) / (np.linalg.norm(z) + 1e-8)

        C = self.C_pairs[(i, j)] if i < j else -self.C_pairs[(j, i)]  # 反对称

        num = np.linalg.norm(C @ z)  # 局部线性量"非交换结构矩阵"

        di = self.D[i] @ z
        dj = self.D[j] @ z

        den = np.linalg.norm(di) + np.linalg.norm(dj) + 1e-8
        return num / den

    def lie_curvature(self, z, k: int = 6, sample=True) -> float:
        """单位扰动下产生的非交换程度,规范化曲率,几何性质≈ 群结构常数
        curvature(z) = ||[Ug,Uh]z|| / (||Ug z - z|| + ||Uh z - z||)
        用随机抽样，否则全量统计会完全平滑掉局部信号 0.013
        """
        if sample:
            idx = np.random.choice(len(self.rho_slow), size=(k, 2), replace=False)
            curvs = [self.curvature_ij(i, j, z, True) for i, j in idx]
        else:
            Uz = np.einsum('nij,j->ni', self.U, z)  # (n,d)
            dz = Uz - z
            norm_dz = np.linalg.norm(dz, axis=1)
            idx = np.argsort(norm_dz)[-k:]
            # w = norm_dz[idx]
            # w = w / (np.sum(w) + 1e-8)
            curvs = [self.curvature_ij(i, j, z, True) for a, i in enumerate(idx)
                     for b, j in enumerate(idx) if a < b]

        return np.mean(curvs)  # np.percentile(curvatures, 75)

    def chaos_signature(self, z, samples=6):
        """
        动力扰动,交换子带来的"幅度差",非交换性的动力学 fingerprint, 通过随机选取两个 move 的交换子作用在 z 上，测量结果的差异来量化非交换性
        测z 所在区域的"动力学弯曲程度"，局部李代数非交换性的离散采样估计||UgUh(z) - UhUg(z)||
        chaos ≈ curvature
        """
        n = len(self.rho_slow)
        sig = []
        for _ in range(samples):
            i, j = random.sample(range(n), 2)
            if i < j:
                C = self.C_pairs[(i, j)]
            else:
                C = -self.C_pairs[(j, i)]
            l2_diff = np.linalg.norm(C @ z)  # chaos 强度: Ug @ (Uh @ z) - Uh @ (Ug @ z)
            sig.append(l2_diff)
        return np.array(sig)  # mean/std

    def move_energy_kg(self, z, target):
        """
        三阶稀疏张量（千亿级三元组） 谱投影 embedding
        (state1, move, state2) 知识图谱嵌入空间
        dz = U @ z - z
        平移 + 旋转
        """
        Uz = np.einsum('nij,j->ni', self.U, z)  # 所有可能的下一状态
        dz = Uz - z

        # TransE-style 平移 + RotatE-style 旋转
        translation_loss = np.linalg.norm(Uz - target, axis=1)  # 平移
        rotation_loss = np.abs(np.imag(np.einsum('ni,i->n', dz, target - z)))  # 旋转分量（虚部）

        energy = (
                1.0 * translation_loss
                + 0.5 * rotation_loss
                + 0.2 * np.linalg.norm(dz, axis=1)  # chaos 捕捉非对称
        )
        return energy

    # ── Theory-driven methods (Paper II: K_αβ, Paper III: κ_d) ──

    def phase_profile(self, z: np.ndarray) -> dict:
        """Spectral phase profile of a slow-space vector.

        Returns {lam: norm} — how much of |z> falls in each spectral layer.
        Layers with negligible projection (< 1e-8) are omitted.
        """
        x = self.lift(z)
        profile = {}
        for lam in self.layer_keys:
            P_lam = self.layer_projector(lam)
            nrm = np.linalg.norm(P_lam @ x)
            if nrm > 1e-8:
                profile[lam] = nrm
        return profile

    def dominant_phase(self, z: np.ndarray) -> float:
        """The spectral layer with largest projection of |z>.
        诊断当前谱相位
        """
        x = self.lift(z)
        best_lam, best_nrm = None, -1
        for lam in self.layer_keys:
            P_lam = self.layer_projector(lam)
            nrm = np.linalg.norm(P_lam @ x)
            if nrm > best_nrm:
                best_nrm, best_lam = nrm, lam
        return best_lam
    
    def phase_crossing_moves(self, lam_src, lam_dst):
        """Generator keys with non-zero transport λ_src → λ_dst."""
        P_src = self.layer_projector(lam_src)
        P_dst = self.layer_projector(lam_dst)
        result = []
        for key, (_, rho) in self.rho_moves.items():
            coupling = np.linalg.norm(P_dst @ rho @ P_src, 'fro')
            if coupling > self.tol * 10:
                result.append(key)
        return result

    def transport_score(self, key: tuple, z: np.ndarray) -> float:
        """Single-step transport score via K_αβ: max over layers of ‖P_α ρ(g) P_β z‖.

        This replaces ad-hoc geometric move scoring with the transport tensor
        from Paper II. A move g is "good" if it transports amplitude from the
        gap vector's dominant layer to a layer closer to the target.
        基于真实算子耦合打分（不是几何分解）
        """
        x = self.lift(z)
        rho = self.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        score = 0.0
        for lam_a in self.layer_keys:
            P_a = self.layer_projector(lam_a)
            x_a = P_a @ x
            if np.linalg.norm(x_a) < 1e-10:
                continue
            for lam_b in self.layer_keys:
                if lam_a == lam_b:
                    continue
                P_b = self.layer_projector(lam_b)
                coupling = np.linalg.norm(P_b @ rho @ x_a)
                score = max(score, coupling)
        return score

    def transport_scores(self, z: np.ndarray) -> dict:
        """Transport scores for all 18 generators (slow-space vector)."""
        scores = {}
        for key in self.rho_moves:
            scores[key] = self.transport_score(key, z)
        return scores

    # ── Full-space scoring (228-dim, no information loss) ──

    def phase_profile_at(self, x: np.ndarray) -> dict:
        """Phase profile of a full 228-dim vector (no projection loss).

        Unlike phase_profile(z) which lifts from slow space (losing hub/fast),
        this sees all 6 phases including V₅/₉ hub and V₁/₃ fast.
        Return {λ: ‖P_λ x‖} for all layers.
        """
        profile = {}
        for lam in self.layer_keys:
            P_lam = self.layer_projector(lam)
            nrm = np.linalg.norm(P_lam @ x)
            if nrm > 1e-8:
                profile[lam] = nrm
        return profile

    def dominant_phase_at(self, x: np.ndarray) -> float:
        """Dominant spectral phase of a full 228-dim vector.
        Return the layer λ with maximum ‖P_λ x‖.
        """
        best_lam, best_nrm = None, -1
        for lam in self.layer_keys:
            nrm = np.linalg.norm(self.layer_projector(lam) @ x)
            if nrm > best_nrm:
                best_nrm, best_lam = nrm, lam
        return best_lam

    def score_full(self, key: tuple, x: np.ndarray) -> float:
        """Transport score for a full 228-dim vector — sees all 6 phases.

        Uses ρ(g) directly on the full vector. The hub (V₅/₉) is visible,
        so inter-phase coupling through the star topology is fully captured.
        """
        rho = self.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        score = 0.0
        for lam_a in self.layer_keys:
            P_a = self.layer_projector(lam_a)
            x_a = P_a @ x
            if np.linalg.norm(x_a) < 1e-10:
                continue
            for lam_b in self.layer_keys:
                if lam_a == lam_b:
                    continue
                P_b = self.layer_projector(lam_b)
                coupling = np.linalg.norm(P_b @ rho @ x_a)
                score = max(score, coupling)
        return score

    def scores_full(self, x: np.ndarray) -> dict:
        """Transport scores for all 18 generators (full 228-dim vector)."""
        scores = {}
        for key in self.rho_moves:
            scores[key] = self.score_full(key, x)
        return scores

    def move_distance(self, key: tuple, x: np.ndarray, x_goal: np.ndarray) -> float:
        """Ground-truth L2 distance after applying ρ(g) to full vector."""
        rho = self.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        x_next = rho @ x
        return float(np.linalg.norm(x_next - x_goal))

    def transport_graph_slow(self) -> dict:
        """Transport graph restricted to slow subspace (λ ≥ 2/3).

        V₅/₉ hub is NOT included — it is separately accessible
        via V_hub / project_hub(). The slow phases (V₈/₉, V₇/₉,
        V₂/₃) appear mutually isolated here; actual transport
        routes through the V₅/₉ hub.
        """
        T = self.transport_tensor()
        graph = {}
        slow_lam = set(np.round(self.w_slow, 6))
        layers = [lam for lam in self.layer_keys if round(lam, 6) in slow_lam]
        for (lam_i, lam_j), info in T.items():
            if lam_i in layers and lam_j in layers:
                graph[(lam_i, lam_j)] = info['max']
        return graph

    # ── Theory-driven spectral search ──

    def _phase_graph(self) -> dict:
        """Build phase-level adjacency from transport tensor.

        Returns {phase: {neighbor_phase: K_max}} for all 6 layers.
        A phase pair (α,β) has a direct edge if K_αβ > tol.
        构建 6 层相位邻接图
        """
        T = self.transport_tensor()
        layers = sorted(self._layers, reverse=True)
        graph = {lam: {} for lam in layers}
        for lam_i in layers:
            for lam_j in layers:
                if lam_i == lam_j:
                    continue
                K = T[(lam_i, lam_j)]['max']
                if K > self.tol * 10:
                    graph[lam_i][lam_j] = K
        return graph

    def phase_path_plan(self, z_start, z_goal) -> list:
        """BFS on FULL phase graph to find shortest phase path.

        The phase graph includes all 6 layers (both slow and fast).
        In the slow subspace, intermediate fast phases (V₅/₉, V₁/₃)
        are invisible — they act as hidden mediators. The returned
        path is the full path; callers should skip fast phases when
        operating in the slow subspace.

        Returns list of phases [α₀, α₁, ..., αₙ].
        """
        gap_start = z_goal - z_start
        src = self.dominant_phase(gap_start)
        goal_profile = self.phase_profile(z_goal)
        if goal_profile:
            dst = max(goal_profile, key=goal_profile.get)
        else:
            dst = src

        if abs(src - dst) < self.tol:
            return [src]

        pg = self._phase_graph()
        from collections import deque
        queue = deque([[src]])
        visited = {src}
        while queue:
            path = queue.popleft()
            current = path[-1]
            if abs(current - dst) < self.tol:
                return path
            for neighbor in pg.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def _slow_phase_plan(self, z_start, z_goal) -> list:
        """Phase plan with V₅/₉ hub as explicit routing node.

        V₅/₉ (λ=5/9) is not in the slow subspace but is the transport
        hub connecting all slow phases. Include it in the plan so the
        composition search can route through it.
        """
        full_plan = self.phase_path_plan(z_start, z_goal)
        if full_plan is None:
            return None
        visible = {round(lam, 6) for lam in self.w_slow}
        visible.add(round(5/9, 6))  # V₅/₉ hub
        return [p for p in full_plan if round(p, 6) in visible]

    def _score_move_to_phase(self, key, gap, target_phase):
        """Score a move by how much it projects gap into target_phase."""
        x = self.lift(gap)
        rho = self.rho_moves[key][1]
        if hasattr(rho, 'toarray'):
            rho = rho.toarray()
        P_target = self.layer_projector(target_phase)
        return float(np.linalg.norm(P_target @ rho @ x))

    def spectral_guided_search(self, z_start, z_goal, max_depth=40):
        """
        基于 K_αβ 传输张量的相位引导搜索（单步贪心）。

        返回：(path, final_dist, depth, phase_trace)
        """
        path = []
        phase_trace = []
        z_curr = z_start.copy()

        for depth in range(max_depth):
            dist = self.l2_distance(z_curr, z_goal)
            gap = z_goal - z_curr
            current_phase = self.dominant_phase(gap)
            phase_trace.append((depth, round(current_phase, 6), round(dist, 4)))

            if dist < 1e-4:
                return path, dist, depth, phase_trace

            best_key, best_score = None, -1.0
            for key in self.rho_slow.keys():
                if len(path) > 0 and CubieMove.is_redundant(path[-1], key):
                    continue
                score = self.transport_score(key, gap)
                if score > best_score:
                    best_score, best_key = score, key

            if best_key is None:
                break

            z_curr = self.apply_move(best_key, z_curr)
            path.append(best_key)

        return path, self.l2_distance(z_curr, z_goal), max_depth, phase_trace

    def spectral_composition_search(self, z_start, z_goal, max_depth=40,
                                     stuck_threshold=4):
        """Phase-guided search with two-step composition lookahead.

        Uses the slow-phase plan (fast phases filtered out). When
        greedy single-step gets stuck (distance stagnation for
        stuck_threshold steps), evaluates 2-step move pairs to
        force a phase jump toward the next planned phase.

        This demonstrates T7 from Paper III: when direct single-step
        transport is blocked, composition (multi-step) paths are needed.

        Returns (path, final_dist, depth, phase_trace, composition_events).
        相位规划 + 卡住时触发两步组合跳跃
        星型拓扑 + hub 在慢空间外，从根本上不适合做搜索：

        慢空间内部无边，所有路由经过不可见的 V₅/₉
        相位层面没有"路径"可规划——每步都是 A→hub→B 的一跳
        transport_score 已经隐式捕获了这个（它盲打所有跨相位耦合）
        组合搜索试图瞄准特定相位，但瞄准本身就反模式
        这种拓扑不适合搜索
        """
        path = []
        phase_trace = []
        composition_events = []
        z_curr = z_start.copy()

        # Phase plan filtered to slow-visible phases
        phase_plan = self._slow_phase_plan(z_start, z_goal)
        # plan_idx: index of the phase we're currently trying to reach
        # Start at 0 (first target after source)
        plan_idx = 0 if phase_plan and len(phase_plan) > 1 else -1

        stuck_count = 0
        best_dist = float('inf')

        for depth in range(max_depth):
            dist = self.l2_distance(z_curr, z_goal)
            gap = z_goal - z_curr
            current_phase = self.dominant_phase(gap)
            phase_trace.append((depth, round(current_phase, 6), round(dist, 4)))

            if dist < 1e-4:
                return path, dist, depth, phase_trace, composition_events

            # Advance plan_idx when we've reached the target phase
            if plan_idx >= 0 and plan_idx < len(phase_plan):
                target_phase = phase_plan[plan_idx]
                if abs(current_phase - target_phase) < self.tol:
                    plan_idx += 1

            # Stuck detection: distance not improving
            if dist >= best_dist - 1e-4:
                stuck_count += 1
            else:
                stuck_count = 0
                best_dist = dist

            # Current target phase
            if plan_idx >= 0 and plan_idx < len(phase_plan):
                target_phase = phase_plan[plan_idx]
            else:
                target_phase = None

            best_key, best_score = None, -1.0
            used_composition = False

            if stuck_count >= stuck_threshold and target_phase is not None:
                # Composition mode: 2-step lookahead toward target phase
                move_keys = list(self.rho_slow.keys())
                best_pair = None
                for i, k1 in enumerate(move_keys):
                    if len(path) > 0 and CubieMove.is_redundant(path[-1], k1):
                        continue
                    z1 = self.apply_move(k1, z_curr)
                    gap1 = z_goal - z1
                    for k2 in move_keys:
                        if CubieMove.is_redundant(k1, k2):
                            continue
                        score = self._score_move_to_phase(k2, gap1, target_phase)
                        if score > best_score:
                            best_score = score
                            best_pair = (k1, k2)
                            used_composition = True

                if best_pair is not None:
                    composition_events.append({
                        'depth': depth,
                        'from_phase': round(current_phase, 4),
                        'to_phase': round(target_phase, 4),
                        'pair': best_pair,
                        'score': round(best_score, 4),
                    })
                    for k in best_pair:
                        z_curr = self.apply_move(k, z_curr)
                        path.append(k)
                    stuck_count = 0
                    best_dist = self.l2_distance(z_curr, z_goal)
                    continue

            # Single-step mode
            if not used_composition:
                for key in self.rho_slow.keys():
                    if len(path) > 0 and CubieMove.is_redundant(path[-1], key):
                        continue
                    if target_phase is not None:
                        score = self._score_move_to_phase(key, gap, target_phase)
                    else:
                        score = self.transport_score(key, gap)
                    if score > best_score:
                        best_score, best_key = score, key

                if best_key is None:
                    break

                z_curr = self.apply_move(best_key, z_curr)
                path.append(best_key)

        return path, self.l2_distance(z_curr, z_goal), max_depth, phase_trace, composition_events
