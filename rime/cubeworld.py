from dataclasses import dataclass
import numpy as np
from rime.cube import CubeBase, StickerCube
from rime.cubie import CubieState, CubieMove, StickerMove, ActionToken, CubieBase, Phase15Coord
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace
import random


class CubeEnv(CubieBase):
    """
    [ 贴纸世界 / 连续 / 感知 ]
        ↓ observables
    [ 中观物理量 / 势能 ]
            ↓ 学习
    [ 群论世界 / 离散 / 搜索 ]
    什么动作让世界更有序
    可解释、可验证的世界模型最小实例,模型是否开始稳定地偏好某些中观结构
    (obs_t, potential_t) --a--> (obs_{t+1}, potential_{t+1})
    引入两种时间尺度
    微时间：单步旋转（执行层）
    宏时间：结构调整周期（认知层）
    例如：5–10 步视为一次“雕刻尝试” 观测：张力是否下降
    引入关键概念：势能 / 张力（Potential / Tension）
    修复成本势
    纠缠传播势
    修正难度势
    """

    def __init__(self, n: int = 3):
        super().__init__(n)
        self.sticker = StickerCube(n=n)
        self.cubie = CubieState.solved()

    def apply(self, move):
        self.sticker.apply(move)
        self.cubie = CubieMove.apply(self.cubie, move)

    def critic(self):
        pass


@dataclass
class RankSample:
    obs: np.ndarray  # observables(s), shape (O,)
    act_pos: np.ndarray  # action_embedding(a+), shape (A,)
    act_neg: np.ndarray  # action_embedding(a-), shape (A,)


class RankingCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class RankingDataset(Dataset):
    def __init__(self, num_samples=1000, obs_dim=6, act_dim=8):
        """
        模拟生成数据
        obs_dim: 状态 observables 维度
        act_dim: 动作 embedding 维度
        """
        self.samples = []
        for _ in range(num_samples):
            # 随机状态向量
            obs = np.random.rand(obs_dim).astype(np.float32)

            # 随机生成动作 embedding，确保 pos < neg (模拟 heuristic)
            act_pos = np.random.rand(act_dim).astype(np.float32)
            act_neg = np.random.rand(act_dim).astype(np.float32)
            if random.random() < 0.5:  # 确保 pos 更优
                act_pos, act_neg = np.minimum(act_pos, act_neg), np.maximum(act_pos, act_neg)

            self.samples.append(RankSample(obs=obs, act_pos=act_pos, act_neg=act_neg))

    def apply(self, cube_env: CubeEnv, num_samples=1000):
        """
        cube_env: CubeEnv，包含 cubie、sticker 等
        num_samples: 样本数量
        Phase-1.5 是关键的「中层边排序 + 剩余角 coset」，动作选择有很多可能，经验启发式不足
        用 critic / ranking NN 来做动作排序
        """
        self.samples = []
        PHASE15_MOVES = CubieMove.phase15_moves()
        actions = list(PHASE15_MOVES.values())
        cubie_phase1 = cube_env.generate_phase1_cubie()
        for _ in range(num_samples):
            # 1. 随机打乱 Phase-1 状态
            cubie = cube_env.generate_phase15_cubie(cubie_phase1)
            state = cubie.to_stickers(cube_env.n)
            obs = cube_env.observables(state)  # 当前 CubieState 投影到向量表示/causal_observables

            # 3. 选择正负动作
            # 正动作 → heuristic/critic 越小越好
            scored = []
            for a in actions:
                next_cubie, next_coord = a.act(cubie)
                # dp = cube_env.delta_potential(state, a.token(cube_env.n)) #ActionToken.from_cubie_move
                # scores.append((dp, a))
                # 使用 heuristic 或 critic 评分
                score = next_coord.heuristic()  # 或 cube_env.critic(next_cubie)
                scored.append((score, a))

            # 排序，越小越好
            scored.sort(key=lambda x: x[0])
            act_pos = scored[0][1]  # 最优动作
            act_neg = scored[-1][1]  # 最差动作

            # 4. 转为 embedding
            act_pos_emb = torch.tensor(act_pos.embedding(cube_env.sticker.n), dtype=torch.float32)
            act_neg_emb = torch.tensor(act_neg.embedding(cube_env.sticker.n), dtype=torch.float32)

            self.samples.append(SimpleNamespace(
                obs=torch.tensor(obs, dtype=torch.float32),
                act_pos=act_pos_emb,
                act_neg=act_neg_emb
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s.obs), torch.tensor(s.act_pos), torch.tensor(s.act_neg)


def ranking_loss(model, batch):
    obs = batch.obs
    act_pos = batch.act_pos
    act_neg = batch.act_neg

    score_pos = model(obs, act_pos)
    score_neg = model(obs, act_neg)

    # y = torch.ones_like(score_pos)
    # loss = criterion(score_pos, score_neg, y)
    # L = -log σ(score_pos - score_neg)
    loss = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8)
    return loss.mean()


def train_ranking_critic(model, dataset, epochs=10, batch_size=128):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # criterion = nn.MarginRankingLoss(margin=1.0)  # pairwise ranking loss

    for ep in range(epochs):
        for batch in loader:
            obs = batch[0].float()
            act_pos = batch[1].float()
            act_neg = batch[2].float()
            batch_data = SimpleNamespace(obs=obs, act_pos=act_pos, act_neg=act_neg)
            loss = ranking_loss(model, batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[epoch {ep}] loss={loss.item():.4f}")
    return model


def order_moves_by_critic(cube, state: np.ndarray, model, actions: list[ActionToken]):
    """ranking critic critic(obs, act_emb) → score"""
    # state = cubie.to_stickers(cube.n)
    obs = torch.tensor(cube.observables(state), dtype=torch.float32)

    scored = []
    for a in actions:
        act_emb = torch.tensor(a.embedding(cube.n), dtype=torch.float32)
        # score = model(obs.unsqueeze(0), act_emb.unsqueeze(0)).item()
        with torch.no_grad():
            score = model(obs, act_emb).item()
        scored.append((score, a))

    scored.sort(reverse=True)  # policy(obs) scored.sort(key=lambda x: x[0])
    return [a for _, a in scored]


class Phase15Dataset(Dataset):
    def __init__(self, num_samples=1000):
        self.samples = []
        for _ in range(num_samples):
            # 随机生成 Phase15Coord
            slice_perm = random.randint(0, 23)
            corner_coset = random.randint(0, 69)
            parity = random.randint(0, 1)
            coord = Phase15Coord(slice_perm, corner_coset, parity)
            label = coord.heuristic()
            self.samples.append((coord.embedding(), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class Phase15Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.head(h).squeeze(-1)


def train_ranking_critic_15(num_epochs=10, batch_size=32, lr=1e-3):
    dataset = Phase15Dataset(num_samples=2000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(dataset[0][0])
    critic = Phase15Critic(input_dim)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            pred = critic(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset):.4f}")

    return critic


class StructuredMoveLayer(nn.Module):
    """
    Kociemba + 神经结构化群表示 Neural Group Theory
    低秩线性变换, 共享基变换,不同 move 共享结构
    s_next = s + A_m s
    A_m = W1 @ diag(e_m) @ W2
    (m1, m2, m3) 的组合等价于某个 coset shift
    在同一组“基向量”上开关不同通道
    约束:逆元一致性/组合一致性/共轭一致性
    state_dim = D
    rank = R   (32/64)

    ρ(g) ≈ I + W_right (W_left(state) ⊙ e_g)
    """

    def __init__(self, state_dim, num_moves, rank=64):
        super().__init__()
        self.W_left = nn.Linear(state_dim, rank, bias=False)
        self.W_right = nn.Linear(rank, state_dim, bias=False)
        self.move_emb = nn.Embedding(num_moves, rank)  # Move embedding e_m ∈ R^R

    def forward(self, state, move_id):
        """
        state: (B, D)
        move_id: (B,)
        """
        z = self.W_left(state)  # (B, R)
        e = self.move_emb(move_id)  # (B, R)
        z = z * e  # elementwise scaling
        delta = self.W_right(z)
        return state + delta


class GroupAwareMove(nn.Module):
    """
    结构感知 Move 层 Group Representation Layer
    把“群作用”变成可微算子,学一个连续的群表示空间
    连续群表示 + 离散投影
    在连续 move 表示空间里找 e,优化 e 是可导的
    +群结构一致性 loss （Structure Loss）
    ρ(g) = exp(∑ e_g,i A_i)
    适合 Phase 1.5 坐标低维
    """

    def __init__(self, state_dim, num_moves, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, state_dim, state_dim))
        self.move_emb = nn.Embedding(num_moves, rank)

    def forward(self, state, move_id):
        e = self.move_emb(move_id)  # (B, R)
        A_comb = torch.einsum("br,rdd->bdd", e, self.A)
        delta = torch.bmm(A_comb, state.unsqueeze(-1)).squeeze(-1)
        return state + delta


class LieMoveRepresentation(nn.Module):
    def __init__(self, num_moves, state_dim, lie_rank):
        super().__init__()

        self.num_moves = num_moves
        self.state_dim = state_dim
        self.lie_rank = lie_rank

        # 1️⃣ move embedding (α_i)
        self.move_emb = nn.Embedding(num_moves, lie_rank)
        # self.face_emb = nn.Embedding(6, lie_rank)  # 只学习 6 个面

        # 2️⃣ Lie algebra basis (A_k)
        self.A = nn.Parameter(
            torch.randn(lie_rank, state_dim, state_dim) * 0.01
        )

    def skew(self, M):
        return M - M.transpose(-1, -2)

    # def lie_element_face(self, move_id):
    #     '''
    #     0-2: U, U2, U'
    #     3-5: D, D2, D'
    #     ...'''
    #     face_id = move_id // 3
    #     power = move_id % 3 + 1  # 1,2,3
    #
    #     alpha = self.face_emb(face_id)          # (batch, r)
    #
    #     X = torch.einsum("br,rdd->bdd", alpha, self.A)
    #     X = self.skew(X)
    #
    #     # power scaling
    #     X = X * power.unsqueeze(-1).unsqueeze(-1)
    #     return X

    def lie_element(self, move_id):
        """
        返回 X = sum_k α_k A_k
        shape: (batch, d, d)
        """
        alpha = self.move_emb(move_id)  # (batch, r)

        # einsum: α_k A_k
        X = torch.einsum("br,rdd->bdd", alpha, self.A)
        # X = self.skew(X) 反对称矩阵
        return X

    def forward(self, state, move_id):
        """
        state: (batch, d)
        move_id: (batch,)
        """
        X = self.lie_element(move_id)
        rho = torch.matrix_exp(X)  # (batch, d, d)
        return torch.bmm(rho, state.unsqueeze(-1)).squeeze(-1)


def conjugation_loss(rho_a, rho_b, rho_aba_inv):
    """
    rho_a, rho_b, rho_aba_inv: (B, D, D) 矩阵表示
    """
    rho_a_inv = torch.inverse(rho_a)  # 或 torch.linalg.pinv 如果不可逆
    right = rho_a @ rho_b @ rho_a_inv
    left = rho_aba_inv

    # Frobenius norm 差异
    diff = left - right
    loss = torch.norm(diff, p='fro', dim=(-2, -1)).pow(2).mean()  # batch mean

    return loss


if __name__ == "__main__":
    critic_model = train_ranking_critic_15()
    print(critic_model)

    model = RankingCritic(obs_dim=6, act_dim=8)
    dataset = RankingDataset(num_samples=2000, obs_dim=6, act_dim=8)
    critic_model = train_ranking_critic(model, dataset)
    print(critic_model)
