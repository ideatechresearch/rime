from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import os
from rime.cube import CubeBase, StickerCube
from rime.base import DATA_DIR
from rime.cubie import CubieState, CubieMove, StickerMove, ActionToken, CubieBase, Phase15Coord
from rime.cubeplot import visualize_angular_lowrank, draw_training_curves
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

    def apply(self, cube_env: CubeEnv, num_samples=100):
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


class AngularLowRank(nn.Module):
    '''Δ(s)≈Ur(s)⋅VT

    径向过程是主导的:变化的主轴在径向
    角向调制是真实存在的:角向不是低维空间,角向变化的“有效自由度”是低维的
    壳层之间主方向不共享固定基
    低维弯曲流形:
    状态空间结构:  Phase1.5 状态空间不是低维线性空间，而是嵌入在高维角向空间中的 4~5 维连续流形。
    高维角向空间 (≈70维)
    ↓
    但所有变化被限制在
    ↓
    一个 4~5 维可变子空间
    ↓
    这个子空间随 r 连续漂移
    '''

    def __init__(self, n_rows, n_cols, rank):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_rows, rank) * 0.1)  # (7, d)
        self.V = nn.Parameter(torch.randn(n_cols, rank) * 0.1)  # (70, d)
        # self.bias = nn.Parameter(torch.zeros(1))

    def forward(self):
        return self.U @ self.V.t()  # + self.bias  # (7,70)


def masked_mse_loss(pred, target, mask):
    diff = pred - target
    return (diff[mask] ** 2).mean()  # 只取有效位置


def masked_relative_error(pred, target, mask):
    diff = pred - target
    return torch.norm(diff[mask]) / torch.norm(target[mask])


def train_rank(M, mask, rank=5, epochs=4000):
    """
    用神经网络优化去逼近 M≈UVT
    rank=5 可以精确拟合
    M_clean:
    Angular rank 1 | rel_err = 0.3454
    Angular rank 2 | rel_err = 0.2217
    Angular rank 3 | rel_err = 0.0949
    Angular rank 4 | rel_err = 0.0245
    Angular rank 5 | rel_err = 0.0000
    M_np:
    Angular rank 1 | rel_err = 0.4701
    Angular rank 2 | rel_err = 0.1734
    Angular rank 3 | rel_err = 0.0870
    Angular rank 4 | rel_err = 0.0259
    Angular rank 5 | rel_err = 0.0000
    低 rank 是因为角向模式高度可压缩
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M_torch = torch.from_numpy(M).float().to(device)  # M_clean
    mask_torch = torch.from_numpy(mask).to(device)  # ~torch.isnan(M_torch)
    # M_obs = M_torch[mask]# 1D tensor，只取观测值
    # observed_count = mask_torch.sum().item()# 490
    model = AngularLowRank(M_torch.shape[0], M_torch.shape[1], rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model()
        loss = masked_mse_loss(pred, M_torch, mask_torch)
        # reg = 1e-4 * (model.U.norm() + model.V.norm()) # 加轻微正则
        # total_loss = loss + reg
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # if epoch % 500 == 0:
        #     with torch.no_grad():
        #         rel = masked_relative_error(pred, M_torch, mask_torch)
        #         print(f"epoch {epoch:4d} | loss {loss.item():.6f} | rel {rel.item():.4f}")

    with torch.no_grad():
        rel_err = masked_relative_error(model(), M_torch, mask_torch)

    print(f"\nAngular rank {rank} | Final observed rel err = {rel_err.item():.4f}")
    return model


class FullLinearMove(nn.Module):
    """move 在这个 embedding 空间里到底是什么？"""

    def __init__(self, state_dim, num_moves=18):
        super().__init__()
        self.move_matrix = nn.Parameter(
            torch.eye(state_dim).unsqueeze(0).repeat(num_moves, 1, 1)
        )

    def forward(self, state, move_id):
        W = self.move_matrix[move_id]  # (B, D, D)
        out = torch.bmm(W, state.unsqueeze(-1))  # (B, D, 1)
        return out.squeeze(-1)  # (B, D)


def train_move_full(dataset, num_epochs=30, batch_size=128, use_coord=False, use_vec=False):
    '''
    move 线性可学,在数据流形上精确拟合
    如果 coord 是 cubie 投影：可能 135 维实际只有 100 左右自由度
    move norm avg :135 维是 ≈ 11.4,40 维是 ≈ 6.32
    coord 不是必须的。coord 只是 cubie 的一个投影。
    是否满足群闭合?
    '''

    if use_vec:  # if np.iscomplexobj(cubie_np_complex):
        cubie_np = np.array([x[3].vector for x in dataset])  # shape: (N,228)
        next_cubie_np = np.array([x[4].vector for x in dataset])
        cubie_np = np.concatenate([cubie_np.real, cubie_np.imag], axis=1)
        next_cubie_np = np.concatenate([next_cubie_np.real, next_cubie_np.imag], axis=1)
    else:
        cubie_np = np.array([x[3].state() for x in dataset])  # shape: (N, 40) cubie_encoding
        next_cubie_np = np.array([x[4].state() for x in dataset])

    input_np = cubie_np  # (N, 40) / (N,456)
    output_np = next_cubie_np
    if use_coord:
        coords_np = np.array([x[0].embedding() for x in dataset])  # shape: (N, embedding_dim)
        next_coords_np = np.array([x[2].embedding() for x in dataset])
        input_np = np.concatenate([cubie_np, coords_np], axis=1)  # 135
        output_np = np.concatenate([next_cubie_np, next_coords_np], axis=1)  # (N, 135)

    states = torch.from_numpy(input_np).float()  # to(torch.complex64) /nn.Linear 等默认不支持复数
    next_states = torch.from_numpy(output_np).float()

    moves = torch.tensor([x[1] for x in dataset], dtype=torch.long)

    train_dataset = TensorDataset(states, moves, next_states)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("states shape:", states.shape, "moves shape:", moves.shape, "next_states shape:",
          next_states.shape)  # 应为 (N, embedding_dim)

    state_dim = states.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullLinearMove(state_dim, num_moves=18).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    train_accuracy = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for s, move_id, s_next in train_loader:
            s = s.to(device)
            move_id = move_id.to(device)
            s_next = s_next.to(device)

            pred = model(s, move_id)  # 预测 s_next

            loss = criterion(pred, s_next)  # 原始 MSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * s.size(0)

            pred_idx = pred.argmax(dim=1)
            true_idx = s_next.argmax(dim=1)
            epoch_correct += (pred_idx == true_idx).sum().item()
            epoch_total += s.size(0)

        with torch.no_grad():
            e_norm = torch.norm(model.move_matrix, dim=(1, 2)).mean().item()
            print(f"move norm avg: {e_norm:.4f}")
            # print(torch.linalg.matrix_rank(model.move_matrix[0]))

        epoch_loss /= len(train_loader.dataset)
        epoch_accuracy = epoch_correct / epoch_total
        # epoch_delta_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} | MSE: {epoch_loss:.8f}| Accuracy: {epoch_accuracy:.6f}")
        """
        (cubie, coord) --move--> (next_cubie, next_coord) 是线性可分的
        结构性误差下界,coord_15 是子群投影，coord 是冗余的
        move norm 非常稳定
        use_coord:135
        move norm avg: 11.2484
        Epoch 1/30 | MSE: 0.89340256| Accuracy: 0.630122
        move norm avg: 11.2485
        Epoch 2/30 | MSE: 0.35570757| Accuracy: 0.682269
        move norm avg: 11.3812
        Epoch 3/30 | MSE: 0.11953534| Accuracy: 0.750232
        move norm avg: 11.5098
        Epoch 4/30 | MSE: 0.03625498| Accuracy: 0.868404
        move norm avg: 11.5932
        Epoch 5/30 | MSE: 0.01361325| Accuracy: 0.991949
        move norm avg: 11.6353
        Epoch 6/30 | MSE: 0.00820733| Accuracy: 0.995048
        move norm avg: 11.6524
        Epoch 7/30 | MSE: 0.00642215| Accuracy: 0.996177
        move norm avg: 11.6615
        Epoch 8/30 | MSE: 0.00550627| Accuracy: 0.996901
        move norm avg: 11.6649
        Epoch 9/30 | MSE: 0.00493999| Accuracy: 0.997794
        move norm avg: 11.6625
        Epoch 10/30 | MSE: 0.00459425| Accuracy: 0.998417
        move norm avg: 11.6594
        Epoch 11/30 | MSE: 0.00435327| Accuracy: 0.999192
        move norm avg: 11.6482
        Epoch 12/30 | MSE: 0.00421712| Accuracy: 0.999512
        move norm avg: 11.6363
        Epoch 13/30 | MSE: 0.00411543| Accuracy: 0.999815
        move norm avg: 11.6212
        Epoch 14/30 | MSE: 0.00402247| Accuracy: 0.999933
        move norm avg: 11.6058
        Epoch 15/30 | MSE: 0.00396494| Accuracy: 0.999933
        move norm avg: 11.5882
        Epoch 16/30 | MSE: 0.00389530| Accuracy: 0.999949
        move norm avg: 11.5709
        Epoch 17/30 | MSE: 0.00386255| Accuracy: 1.000000
        move norm avg: 11.5526
        Epoch 18/30 | MSE: 0.00381793| Accuracy: 1.000000
        move norm avg: 11.5363
        Epoch 19/30 | MSE: 0.00381021| Accuracy: 1.000000
        move norm avg: 11.5163
        Epoch 20/30 | MSE: 0.00377996| Accuracy: 1.000000
        ...
        Epoch 29/30 | MSE: 0.00375344| Accuracy: 1.000000
        move norm avg: 11.4264
        Epoch 30/30 | MSE: 0.00375314| Accuracy: 1.000000
        
        在 40 维实向量空间上构造魔方群的一个表示
        cubie 40维:
        每个 move 在 40 维空间上是一个精确线性映射
        move norm avg: 6.1071
        Epoch 1/30 | MSE: 2.98347276| Accuracy: 0.630139
        move norm avg: 6.0631
        Epoch 2/30 | MSE: 1.19569699| Accuracy: 0.682084
        move norm avg: 6.0997
        Epoch 3/30 | MSE: 0.40066187| Accuracy: 0.750484
        move norm avg: 6.1531
        Epoch 4/30 | MSE: 0.11509017| Accuracy: 0.867730
        move norm avg: 6.1944
        Epoch 5/30 | MSE: 0.03507560| Accuracy: 0.992252
        move norm avg: 6.2217
        Epoch 6/30 | MSE: 0.01517144| Accuracy: 0.994812
        move norm avg: 6.2408
        Epoch 7/30 | MSE: 0.00863457| Accuracy: 0.995941
        move norm avg: 6.2559
        Epoch 8/30 | MSE: 0.00542724| Accuracy: 0.996833
        move norm avg: 6.2687
        Epoch 9/30 | MSE: 0.00361668| Accuracy: 0.997726
        move norm avg: 6.2787
        Epoch 10/30 | MSE: 0.00249707| Accuracy: 0.998316
        move norm avg: 6.2866
        ...
        Epoch 28/30 | MSE: 0.00011159| Accuracy: 1.000000
        move norm avg: 6.3235
        Epoch 29/30 | MSE: 0.00004263| Accuracy: 1.000000
        move norm avg: 6.3234
        Epoch 30/30 | MSE: 0.00003755| Accuracy: 1.000000
        """
    return model, train_losses, train_accuracy


class StructuredMoveLayer(nn.Module):
    """
    ->变化低秩,低秩双线性调制结构
    Kociemba + 神经结构化群表示 Neural Group Theory
    低秩线性变换, 共享基变换,不同 move 共享结构
    s_next = s + A_m s
    A_m = W1 @ diag(e_m) @ W2
    (m1, m2, m3) 的组合等价于某个 coset shift
    在同一组“基向量”上开关不同通道
    约束:逆元一致性/组合一致性/共轭一致性
    state_dim = D
    rank = R   (16/32/64)
    Δ = (W_left(s) ⊙ e_move) W_right
    ρ(g) ≈ I + W_right (W_left(state) ⊙ e_g)
    state 先被投影到低秩空间
    move 只控制低秩通道
    再 lift 回高维
    """

    def __init__(self, state_dim, move_dim=8, rank=64):
        super().__init__()
        self.W_left = nn.Linear(state_dim, rank, bias=False)
        self.W_right = nn.Linear(rank, state_dim, bias=False)
        self.geo_proj = nn.Linear(move_dim, rank)  # 几何先验

    def forward(self, state, move):
        """
        (state_t, move_id, state_{t+1}) Phase15Coord.embedding() state_vec.shape = (95,)
        state: (B, D)
        move_id: (B,)
        loss = mse(pred_state, target_state) 先看模型自然会学成什么样
        """
        z = self.W_left(state)
        e = self.geo_proj(move)  # (B, R)
        # z = z * e  # elementwise scaling
        z_scaled = z * (1.0 + torch.tanh(e))  # 或 z + e，或 z * (1 + e) torch.sigmoid(e)  sigmoid 约束缩放范围
        delta = self.W_right(z_scaled)
        return state + delta


def train_move(dataset, num_epochs=30, rank=40, batch_size=128):
    cubie_np = np.array([x[3].state() for x in dataset])  # shape: (N, 40)
    # states_np = np.array([x[0].embedding() for x in dataset])  # shape: (N, embedding_dim)
    next_cubie_np = np.array([x[4].state() for x in dataset])
    # next_states_np = np.array([x[2].embedding() for x in dataset])
    moves_np = np.array([CubieMove.embedding(move_id=x[1]) for x in dataset])
    # input_np = np.concatenate([states_np, cubie_np], axis=1)

    states = torch.from_numpy(cubie_np).float()
    moves = torch.from_numpy(moves_np).float()
    next_states = torch.from_numpy(next_cubie_np).float()

    train_dataset = TensorDataset(states, moves, next_states)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("states shape:", states.shape, "moves shape:", moves.shape,
          "next_states shape:", next_states.shape)  # 应为 (N, embedding_dim)
    """
    states shape: torch.Size([59371, 40]) moves shape: torch.Size([59371, 8]) next_states shape: torch.Size([59371, 40])
    """

    state_dim = states.shape[1]  # 40
    move_dim = moves.shape[1]  # 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = torch.zeros(state_dim, state_dim, device=device)
    N = 0
    for s, move, s_next in train_loader:
        s = s.to(device)
        s_next = s_next.to(device)

        # 批量矩阵乘法
        M += s_next.T @ s
        N += s.size(0)

    M /= N

    print("rank:", torch.linalg.matrix_rank(M))  # rank: tensor(18)
    """Phase1.5 在 one-hot 表示下是 full rank，Phase1.5 状态的线性变换几乎满秩,线性,不是低秩"""

    model = StructuredMoveLayer(state_dim, move_dim, rank=rank)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    train_accuracy = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for s, move, s_next in train_loader:
            pred = model(s, move)  # 预测 s_next
            delta_true = s_next - s  # 真 delta
            delta_pred = pred - s  # 预测 delta

            # loss = criterion(pred, s_next)    # 原始 MSE
            loss = criterion(delta_pred, delta_true)  # delta MSE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * s.size(0)

            pred_idx = pred.argmax(dim=1)
            true_idx = s_next.argmax(dim=1)
            epoch_correct += (pred_idx == true_idx).sum().item()
            epoch_total += s.size(0)

        with torch.no_grad():
            e_norm = torch.norm(model.geo_proj.weight, dim=1).mean().item()
            sample_move = next(iter(train_loader))[1]
            e_std = model.geo_proj(sample_move).std().item()
            print(f"move proj norm avg: {e_norm:.4f},e std:{e_std:.4f}")

        epoch_loss /= len(train_loader.dataset)
        epoch_accuracy = epoch_correct / epoch_total
        # epoch_delta_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs} | MSE: {epoch_loss:.6f}| Accuracy: {epoch_accuracy:.6f}")
    """
    move proj norm avg: 0.7636,e std:0.6312
    Epoch 1/30 | MSE: 3.224592| Accuracy: 0.460814
    move proj norm avg: 1.1682,e std:0.8905
    Epoch 2/30 | MSE: 1.997367| Accuracy: 0.463560
    move proj norm avg: 1.4933,e std:1.0923
    Epoch 3/30 | MSE: 1.381364| Accuracy: 0.516279
    move proj norm avg: 1.7665,e std:1.3171
    Epoch 4/30 | MSE: 0.883138| Accuracy: 0.655943
    move proj norm avg: 1.9605,e std:1.4426
    Epoch 5/30 | MSE: 0.578673| Accuracy: 0.820215
    move proj norm avg: 2.1242,e std:1.5499
    Epoch 6/30 | MSE: 0.430045| Accuracy: 0.907817
    move proj norm avg: 2.2702,e std:1.6191
    Epoch 7/30 | MSE: 0.326392| Accuracy: 0.946607
    move proj norm avg: 2.3944,e std:1.6639
    ...
    move proj norm avg: 3.7942,e std:2.2706
    Epoch 27/30 | MSE: 0.049503| Accuracy: 0.992673
    move proj norm avg: 3.8637,e std:2.3653
    Epoch 28/30 | MSE: 0.047691| Accuracy: 0.993313
    move proj norm avg: 3.9341,e std:2.3951
    Epoch 29/30 | MSE: 0.046292| Accuracy: 0.993953
    move proj norm avg: 4.0069,e std:2.4519
    Epoch 30/30 | MSE: 0.045274| Accuracy: 0.994290
    
    模型在逼近某种结构极限
    模型只能逼近置换矩阵
    但永远学不到精确 permutation
    -->线性但不是低秩双线性
    """
    return model, train_losses, train_accuracy


class Phase15Model(nn.Module):
    def __init__(self, state_dim=95, hidden_dim=128, rank=32, num_moves=18):
        super().__init__()

        self.state_emb = nn.Linear(state_dim, hidden_dim)  # 映射到 128 或 256 维
        self.move_emb = nn.Embedding(num_moves, rank)  # Move embedding e_m ∈ R^R
        self.move_layer = StructuredMoveLayer(
            state_dim=hidden_dim,
            move_dim=num_moves,
            rank=rank
        )
        self.proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, move_id):
        """
        move_id: (B,)
        """
        z = self.state_emb(state)
        e = self.move_emb(move_id)
        z = self.move_layer(z, e)
        out = self.proj(z)
        return out


class GroupAwareMove(nn.Module):
    """
    纯表示论，李代数参数化群表示
    结构感知 Move 层 Group Representation Layer
    把“群作用”变成可微算子,学一个连续的群表示空间
    连续群表示 + 离散投影
    在连续 move 表示空间里找 e,优化 e 是可导的
    +群结构一致性 loss （Structure Loss）
    ρ(g) = exp(∑ e_g,i A_i)
    适合 Phase 1.5 坐标低维
    连乘发散？
    """

    def __init__(self, state_dim, rank, num_moves=18):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, state_dim, state_dim))  # (rank, D, D)
        self.move_emb = nn.Embedding(num_moves, rank)

    def forward(self, state, move_id):
        e = self.move_emb(move_id)  # (B, R)
        # A_comb = torch.tensordot(e, self.A, dims=([1], [0]))
        A_comb = torch.einsum("br,rdd->bdd", e, self.A)  # 'bm,mij->bij'
        W = torch.matrix_exp(A_comb)  # (B, D, D)
        delta = torch.bmm(W, state.unsqueeze(-1)).squeeze(-1)
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

        # Xp = nn.Parameter(torch.randn(d, d) * 0.01)
        # Xp = Xp - Xp.T
        # 
        # rho_p = torch.matrix_exp(Xp)
        # # loss += | | rho_p @ rho_p - I | |

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


def set_all_seeds(seed=47):
    # 1. Python 内置 random
    random.seed(seed)

    # 2. NumPy RNG
    np.random.seed(seed)

    # 3. PyTorch RNG（CPU 和 GPU 都设置）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 单 GPU
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 4. 强制 PyTorch 确定性（最严格的可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化，避免随机性


if __name__ == "__main__":
    critic_model = train_ranking_critic_15()
    print(critic_model)

    model = RankingCritic(obs_dim=6, act_dim=8)
    dataset = RankingDataset(num_samples=2000, obs_dim=6, act_dim=8)
    critic_model = train_ranking_critic(model, dataset)
    print(critic_model)
    """
    构造 M_np
    ↓
    SVD 验证 rank=5
    ↓
    神经网络逼近
    ↓
    rank=5 → rel_err=0
    
    状态差异核是否近似
    radial × angular low-rank structure
    """
    cube = CubieBase(n=3)
    cube.build_pruning_table()
    dist_3d, M_np, M_clean, mask = cube.get_phase15_M()
    for r in range(1, 6):
        train_rank(M_np, mask, r)  # or M_clean 中心化版本

    set_all_seeds(47)
    rank = 5
    model = train_rank(M_np, mask, rank)
    device = next(model.parameters()).device

    M_torch = torch.from_numpy(M_np).float().to(device)
    mask_torch = torch.from_numpy(mask).to(device)
    # U = model.U.detach().cpu().numpy()
    # V = model.V.detach().cpu().numpy()
    with torch.no_grad():
        pred = model()  # (n_layers, n_corners)
        rel = masked_relative_error(pred, M_torch, mask_torch)
        obs_err = torch.norm((model() - M_torch)[mask]) / torch.norm(M_torch[mask])
        print("Observed rel err (torch):", obs_err.item())

    print("Verified rel err:", rel.item())

    model.eval()
    with torch.no_grad():
        pred = model()

    pred_np = pred.cpu().numpy()
    true_np = M_torch.cpu().numpy()
    mask_np = mask_torch.cpu().numpy()

    import os

    # Adjust these paths to match your actual Tcl/Tk directories
    os.environ['TCL_LIBRARY'] = r'D:\Program Files\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'D:\Program Files\Python\Python313\tcl\tk8.6'
    # import tkinter as tk
    # root = tk.Tk()  # Should now work

    visualize_angular_lowrank(
        pred_np, true_np, mask_np,
        rank=rank,
        save_prefix=os.path.join(DATA_DIR, "angular_lowrank")
    )
    '''
    Angular rank 5 | Final observed rel err = 0.0000
    Observed rel err (torch): 4.485024618361422e-08
    Verified rel err: 4.485024618361422e-08
    
    Relative error (observed) ≈ 0.000000
    Explained variance by rank:
      mode 1: 77.70%
      mode 2: 19.70%
      mode 3: 1.21%
      mode 4: 1.09%
      mode 5: 0.30%
    '''

    import joblib
    import matplotlib.pyplot as plt

    gpd = True
    as_key = True
    path = os.path.join(DATA_DIR, "phase15_dataset_by_key.pkl") if as_key else os.path.join(DATA_DIR, "phase15_dataset.pkl")
    if gpd:
        # dataset = cube.generate_phase15_dataset(max_depth=10, num_starting_points=50, num_samples=50000, as_key=as_key)
        dataset = cube.generate_phase15_dataset(max_depth=16, num_starting_points=100, num_samples=20000, as_key=as_key,
                                                start_random=False)
        joblib.dump(dataset, path)
    else:
        dataset = joblib.load(path)

    if as_key:
        dataset = [(Phase15Coord(*d[0]), d[1], Phase15Coord(*d[2]),
                    CubieState.from_key(d[3]), CubieState.from_key(d[4]),
                    CubieState.from_key(d[5]), d[6]
                    ) for d in dataset]

    print(len(dataset),
          sum([x[3].is_phase1_solved() for x in dataset]), sum([x[4].is_phase1_solved() for x in dataset]),
          sum([x[4].is_phase2_ready() for x in dataset]))  # 275731 102499 59371 168 / 169655 44065 25853 311
    total_ready = {d[4].key: d[4] for d in dataset if d[4].is_phase2_ready()}
    print(f"unique phase2 ready:{len(total_ready)}")  # 43/24

    from collections import defaultdict

    transition_map = defaultdict(set)  # key: (state_key, move_id) → set of next_state_key
    for idx, (state, move_id, next_state, cubie, *_) in enumerate(dataset):
        group_key = (state.key, move_id, cubie.key)
        transition_map[group_key].add(next_state.key)

    non_unique_transitions = 0
    for group, next_set in transition_map.items():
        if not len(next_set) == 1:
            non_unique_transitions += 1
            # print(f"非唯一转移: {group} → {next_set} (共 {len(next_set)} 个 next_state)")

    total_groups = len(transition_map)
    unique_transitions = total_groups - non_unique_transitions
    print("\n" + "=" * 60)
    print(f"总 group 数量 (state.key + move_id): {total_groups}")
    print(f"唯一转移 (next_state.key 唯一): {unique_transitions} ({unique_transitions / total_groups:.2%})")
    print(f"非唯一转移: {non_unique_transitions} ({non_unique_transitions / total_groups:.2%})")

    stats = defaultdict(int)  # 观察不同动作破坏 Phase-1 的概率
    for _, move_id, _, cubie, next_cubie, *_ in dataset:
        key = (move_id, cubie.is_phase1_solved(), next_cubie.is_phase1_solved())
        stats[key] += 1

    for k, v in sorted(stats.items()):
        print(f"move {k[0]}:{CubieMove.basic_generators[k[0]]}, cubie/next_phase1={(k[1], k[2])} → {v} 次")

    """
    总 group 数量 (state.key + move_id): 261447
    唯一转移 (next_state.key 唯一): 261447 (100.00%)
    非唯一转移: 0 (0.00%)
    cubie + coord 唯一
    A 类 move
    比如 move 2,5,6,7,8,9,10,11,14,17：保持 Phase1 子群结构
    B 类 move
    比如 move 0,1,3,4,12,13,15,16：会打破 Phase1 条件
    
    move 0, cubie/next_phase1=(np.False_, np.False_) → 9261 次
    move 0, cubie/next_phase1=(np.False_, True) → 339 次
    move 0, cubie/next_phase1=(True, np.False_) → 5786 次
    move 1, cubie/next_phase1=(np.False_, np.False_) → 9490 次
    move 1, cubie/next_phase1=(np.False_, True) → 328 次
    move 1, cubie/next_phase1=(True, np.False_) → 5812 次
    move 2, cubie/next_phase1=(np.False_, np.False_) → 9627 次
    move 2, cubie/next_phase1=(True, True) → 5682 次
    move 3, cubie/next_phase1=(np.False_, np.False_) → 9213 次
    move 3, cubie/next_phase1=(np.False_, True) → 308 次
    move 3, cubie/next_phase1=(True, np.False_) → 5674 次
    move 4, cubie/next_phase1=(np.False_, np.False_) → 9192 次
    move 4, cubie/next_phase1=(np.False_, True) → 342 次
    move 4, cubie/next_phase1=(True, np.False_) → 5669 次
    move 5, cubie/next_phase1=(np.False_, np.False_) → 9491 次
    move 5, cubie/next_phase1=(True, True) → 5716 次
    move 6, cubie/next_phase1=(np.False_, np.False_) → 9578 次
    move 6, cubie/next_phase1=(True, True) → 5529 次
    ...

    """
    # from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE
    # pca = PCA(n_components=2)
    # X_pca= pca.fit_transform(X)
    # next_cubie_np = np.array([x[4].vector for x in dataset])
    # X = np.concatenate([next_cubie_np.real, next_cubie_np.imag], axis=1)
    # depths = [x[6] for x in dataset]


    dataset = [x for x in dataset if x[4].is_phase1_solved()]  # 只看 next_cubie

    # 数据驱动群表示构造
    model, train_losses, train_accuracy = train_move_full(dataset, num_epochs=30, use_coord=False, use_vec=False)
    mats = []
    for i in range(len(model.move_matrix)):
        W = model.move_matrix[i].detach().cpu().numpy()
        mats.append(W)
    # 找公共不变子空间
    stacked = np.concatenate(mats, axis=0)
    u, s, vh = np.linalg.svd(stacked)

    print("Singular values:")
    print(s[:20])

    all_moduli = []

    for i, M in enumerate(mats):
        eigvals = np.linalg.eigvals(M)
        moduli = np.abs(eigvals)
        all_moduli.extend(moduli)

        print(f"Move {i}:")
        print("  max |λ|:", moduli.max())
        print("  min |λ|:", moduli.min())
        print("  mean|λ|:", moduli.mean())

    print("\nOverall:")
    print("  max:", np.max(all_moduli))
    print("  min:", np.min(all_moduli))

    tol = 1e-2
    for i, M in enumerate(mats):
        row_counts = []
        col_counts = []

        for r in range(M.shape[0]):
            row_counts.append(np.sum(np.abs(M[r]) > tol))

        for c in range(M.shape[1]):
            col_counts.append(np.sum(np.abs(M[:, c]) > tol))

        print(f"Move {i}:")
        print("  avg nonzeros per row:", np.mean(row_counts))
        print("  avg nonzeros per col:", np.mean(col_counts))

    # 可视化训练曲线
    draw_training_curves(train_losses, train_accuracy, loss_label='MSE', acc_label='Accuracy',
                         title="FullLinearMove Training Curves (Loss & Accuracy)")

    _, train_losses, train_accuracy = train_move(dataset, num_epochs=30, rank=18)
    """
    用 MSE 拟合 one-hot。
    预测是连续值。
    但 move 是离散置换。
    它不是分类问题。
    """
    # 可视化训练曲线

    draw_training_curves(train_losses, train_accuracy, loss_label='Delta MSE', acc_label='Accuracy',
                         title="StructuredMoveLayer Training Curves (Loss & Accuracy)")
