from dataclasses import dataclass
from rime.base import class_property, class_cache, chainable_method, class_status
import numpy as np
import random, math
from collections import deque, defaultdict


class CubeBase:
    '''
    逻辑 → 几何 → 群论 → 可视化,状态系统 World / State Space
    运算必须封闭
    表示不能混层
    不变量是结构给的,不是定义的
    贴纸世界是表示的一种投影，投影不可逆
    魔方是一个噪声极低、结构极硬的实验宇宙
    axis = 0 → x → R/L
    axis = 1 → y → U/D
    axis = 2 → z → F/B
    # AXIS_FACE defines the canonical mapping between Cartesian axes and cube faces.

    Coordinate system:
      Right-handed Cartesian system.

    AXIS_FACE[axis] = (POS_FACE, NEG_FACE)
      POS_FACE: face whose outward normal aligns with +axis direction
                (outer layer: layer == mid / n-1)
      NEG_FACE: face whose outward normal aligns with -axis direction
                (outer layer: layer == -mid / 0)

    Rotation direction convention:
      A positive rotation `d` is defined as clockwise when viewed
      along the face outward normal (right-hand rule).

    NOTE:
      Face normal directions are fully defined by AXIS_FACE.
      No additional geometric inference (e.g. dot products) is required.
    '''
    # 哪些面在轴的正/负一侧(POS_FACE, NEG_FACE)_SIGN,几何法向,用于坐标推导
    # 约定：layer == +mid  → POS_FACE (法向 = +axis) 几何意义, 实际可见面 = AXIS_FACE[axis][0]
    #      layer == -mid  → NEG_FACE (法向 = -axis) 几何意义, 实际可见面 = AXIS_FACE[axis][1]
    AXIS_FACE = [
        ('R', 'L'),  # X axis (0), X+ → R, X− → L
        ('U', 'D'),  # Y axis (1), Y+ → U, Y− → D
        ('F', 'B'),  # Z axis (2), Z+ → F, Z− → B
    ]  # YOLO 正轴方向,物理右手坐标一致, VISIBLE_FACE 人眼观察标准,视角切换:法向 = −axis
    AXIS_STRIP = (
        ['U', 'F', 'D', 'B'],  # 'X', 从 +X 看 CCW ,+Y 作为参考
        ['F', 'R', 'B', 'L'],  # 'Y',+Z 作为参考
        ['U', 'L', 'D', 'R'],  # 'Z',+X 作为参考 ['R', 'U', 'L', 'D']
    )  # CCW 视角,从 +axis 方向看过去,4 元环路,trip 顺序,d 是从“面法向外看”时的顺时针旋转次数（右手规则）
    # AXIS_STRIP = (
    #     ['U', 'B', 'D', 'F'],  # X
    #     ['F', 'L', 'B', 'R'],  # Y
    #     ['U', 'R', 'D', 'L'],  # Z
    # )
    # === 依赖 FACES 生成派生结构 ===
    AXIS_VEC = np.eye(3, dtype=int)
    FACE_UP = {
        'U': -AXIS_VEC[2],  # -Z: [0, 0, -1],AXIS_VEC[0]
        'D': AXIS_VEC[2],  # Z:[0, 0, 1]

        'F': AXIS_VEC[1],  # Y:[0, 1, 0]
        'B': AXIS_VEC[1],

        'R': AXIS_VEC[1],
        'L': AXIS_VEC[1],
    }  # 固定局部坐标系
    # === 固定顺序 ===
    FACES = ['U', 'D', 'F', 'B', 'L', 'R']  # 面标识 上 下 前 后 左 右,通常 0=U, 1=D, 2=F, 3=B, 4=L, 5=R
    COLORS = ['W', 'Y', 'R', 'O', 'G', 'B']  # Rubiks 0:白色, 1:黄色, 2:红色, 3:橙色, 4:绿色, 5:蓝色
    # (x, y, z) ∈ {±1, ±1, ±1}
    CORNER_POS_SIGNS = [
        (+1, +1, +1),  # URF
        (-1, +1, +1),  # UFL
        (-1, +1, -1),  # ULB
        (+1, +1, -1),  # UBR
        (+1, -1, +1),  # DFR
        (-1, -1, +1),  # DLF
        (-1, -1, -1),  # DBL
        (+1, -1, -1),  # DRB
    ]  # 排序逻辑是魔方标准：从U层右前开始，顺时针绕一圈；D层类似（但右手调整）
    EDGE_POS_SIGNS = [
        (+1, +1, 0),  # UR
        (0, +1, +1),  # UF
        (-1, +1, 0),  # UL
        (0, +1, -1),  # UB
        (+1, 0, +1),  # RF
        (-1, 0, +1),  # LF
        (-1, 0, -1),  # LB
        (+1, 0, -1),  # RB
        (+1, -1, 0),  # DR
        (0, -1, +1),  # DF
        (-1, -1, 0),  # DL
        (0, -1, -1),  # DB
    ]  # EDGE_COORDS
    # 对 y 轴的几何分层 quotient，几何派生接口
    # corner positions
    U_CORNER_POSITIONS = tuple(i for i, (_, y, _) in enumerate(CORNER_POS_SIGNS) if y == +1)  # (0, 1, 2, 3)
    D_CORNER_POSITIONS = tuple(i for i, (_, y, _) in enumerate(CORNER_POS_SIGNS) if y == -1)  # (4, 5, 6, 7)
    # edge positions index（position on cube）
    SLICE_POSITIONS = tuple(i for i, (_, y, _) in enumerate(EDGE_POS_SIGNS) if y == 0)  # slice {FR, FL, BL, BR},从前向后扫描
    NON_SLICE_POSITIONS = tuple(i for i, (_, y, _) in enumerate(EDGE_POS_SIGNS) if y != 0)

    # [0, 1, 2, 3, 8, 9, 10, 11] 非 slice [i for i in range(12) if i not in SLICE_POSITIONS]

    # # Define corner and edge positions
    # corner_positions = np.array(CORNER_POS_SIGNS, dtype=int)
    # edge_positions = np.array(EDGE_POS_SIGNS, dtype=int)

    def __init__(self, n: int = 3):
        """纯几何 / move 定义（无状态）"""
        self.n = n
        self.solved_idx = np.arange(6 * n * n, dtype=np.uint32).reshape(6, n, n)
        self.solved = self.solved(n)
        self.SOLVED_CORNERS_MAP = self.solved_corners_map()
        self.SOLVED_EDGES_MAP = self.solved_edges_map()

    @property
    def mid(self) -> int:
        return self.n // 2

    @property
    def center_layers(self) -> list:
        return self.center_layers_list(self.n)

    @staticmethod
    def idx_to_state(state_idx: np.ndarray) -> np.ndarray:
        """颜色视图"""
        n = state_idx.shape[1]
        return (state_idx // (n * n)).astype(np.uint8)

    @classmethod
    def solved(cls, n: int) -> np.ndarray:
        solved = np.zeros((6, n, n), dtype=np.uint8)
        for f in range(6):
            solved[f, :, :] = f
        return solved

    def is_solved(self, state: np.ndarray) -> bool:
        return bool(np.array_equal(state, self.solved))  # not (state ^ self.solved).any()

    def diff_coords(self, state: np.ndarray) -> np.ndarray:
        diff_mask = state != self.solved
        return np.argwhere(diff_mask)  # nonzero

    def heuristic(self, state: np.ndarray):
        """
        估价函数：错误块的数量（简单启发）,对 BFS/IDA*/Beam search 可用,小魔方适用
        """
        errors = np.count_nonzero(state != self.solved)
        return errors // max(1, self.n)  # 每个错误影响多个面

    @staticmethod
    def encode(state: np.ndarray) -> bytes:
        return np.ascontiguousarray(state, dtype=np.uint8).tobytes()

    @staticmethod
    def embedding(state_idx: np.ndarray) -> np.ndarray:
        """
        Transformer / GNN 相对坐标 relative[color,u,v]
        返回: (batch, 6*n*n, 3)
        """
        if state_idx.ndim == 3:
            state_idx = state_idx[None]  # 加 batch dim (1, 6, n, n)

        b, f, n, _ = state_idx.shape
        # 颜色归一化
        color = (state_idx // (n * n)).astype(np.float32) / 5.0  # (b, 6, n, n)
        # id = (state_idx % (n * n)).astype(np.float32) / (n * n)

        # 相对坐标网格 ∈ [-1, 1]
        grid = np.linspace(-1.0, 1.0, n)
        u, v = np.meshgrid(grid, grid)  # (n,n) for each face

        uv = np.stack([u, v], axis=-1)[None, None]  # (1,1,n,n,2)
        uv = np.broadcast_to(uv, (b, f, n, n, 2))  # 广播到(b,6,n,n,2)

        emb = np.concatenate([color[..., None], uv], axis=-1)  # (b,6,n,n,3)#,  id[...,None]
        return emb.reshape(b, -1, 3)  # (b, 6*n*n, 3)

    @staticmethod
    def get_data(state: np.ndarray, coords_def: list | tuple) -> tuple:
        '''把 piece 看成一个“颜色集合”，忽略朝向，只关心它是哪个 piece'''
        return tuple(sorted(state[f, r, c].tolist() for f, r, c in coords_def))

    def get_corners(self, state: np.ndarray) -> np.ndarray:
        """
        返回 8 个角块的三颜色编号（按顺序）shape = (8, 3),
        角块位置： (face, row, col) 三元组的 3 个集合
        """
        res = np.empty((8, 3), dtype=state.dtype)
        for i, corner in enumerate(self.corner_coords(self.n)):
            # corner 是 [(face,row,col), ...]
            for j, (f, r, c) in enumerate(corner):
                res[i, j] = state[f, r, c]
        return res

    def get_edges(self, state: np.ndarray) -> np.ndarray:
        """返回 12 个角块的2颜色编号（按顺序）shape = (12,2)"""
        res = np.empty((12, 2), dtype=state.dtype)
        for i, edge in enumerate(self.edge_coords(self.n)):
            for j, (f, r, c) in enumerate(edge):
                res[i, j] = state[f, r, c]
        return res

    # def get_centers(self, state: np.ndarray) -> np.ndarray:
    #     """
    #     从 sticker-level 状态生成 orbit_perm
    #     state: np.ndarray, shape = (6, n, n), 每个元素是颜色或编号
    #     """
    #     flat = []  # flatten 所有中心贴纸，顺序严格对应 center_orbits
    #     for fidx, rings in enumerate(self.get_center_rings(self.n)):
    #         for ring in rings:
    #             for r, c, _ in ring:
    #                 flat.append(state[fidx, r, c])
    #     return np.array(flat, dtype=state.dtype)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """48: 8*3+12*2,物理块不去重"""
        return np.concatenate([self.get_corners(state).ravel(), self.get_edges(state).ravel()]).astype(np.uint8)

    def solved_corners_map(self) -> dict:
        """
        角块的 cubie id
        用字典加速 lookup，建立 solved corner 的颜色集合 → index 映射
        忽略朝向，否则同一个块旋转后会被误认为不同块,角块的 twist 信息在贴纸级处理里丢失
        slot 0 为 reference （基准面）
        """
        solved = self.get_corners(self.solved)  # (8, 3) np.sort(, axis=1)
        return {frozenset(c): (pid, c) for pid, c in enumerate(solved)}

    def corner_ids_ori(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        返回每个 corner 对应的 piece id（0~7）, corners_perm (8)
        perm 表示 piece 的重排，当前位置 i 的角块，原来是 solved 的哪一块？ 群作用矩阵里计算 twist
        corner_ori 不是物理量，是约定量，测不准
        """
        corners = self.get_corners(state)
        perm = np.empty(8, dtype=np.int8)
        ori = np.empty(8, dtype=np.int8)
        for i, c in enumerate(corners):
            pid, ref = self.SOLVED_CORNERS_MAP[frozenset(c)]
            perm[i] = pid
            for twist in range(3):
                rolled = np.roll(ref, -twist)
                if np.array_equal(rolled, c):
                    ori[i] = twist
                    break
            else:
                ori_raw = list(c).index(ref[0])  # 原始索引：0/1/2
                ori[i] = (-ori_raw) % 3
                print(f"{pid}:No matching twist found")

        # ori = (ori - ori[0]) % 3  # 全局 orientation gauge fix
        # ori = (3 - ori) % 3/ (- ori) % 3
        ori[-1] = (-ori[:-1].sum()) % 3  # 把 orientation 投影到合法子空间,修正最后一个角方向
        return perm, ori

    def solved_edges_map(self) -> dict:
        """
        标准顺序,依赖 solved_edges 的正确定义（顺序必须匹配参考色先）
        cubie id : ref  0/1
        """
        solved = self.get_edges(self.solved)  # (12, 2)
        edge_map = {}
        for pid in range(12):
            a, b = solved[pid]
            edge_map[(a, b)] = (pid, 0)  # forward 正向映射：标准颜色顺序 → id,ori[i] = 0
            edge_map[(b, a)] = (pid, 1)  # flipped 翻转映射：同一个块,ori[i] = 1
        return edge_map

    def edge_ids_ori(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ edges_perm  (12) 保留方向,按标准顺序"""
        edges = self.get_edges(state)  # current_edges
        perm = np.empty(12, dtype=np.int8)
        ori = np.empty(12, dtype=np.int8)
        for i, (a, b) in enumerate(edges):
            pid, o = self.SOLVED_EDGES_MAP[(a, b)]
            perm[i] = pid
            ori[i] = o
        ori[-1] = (-ori[:-1].sum()) % 2  # 修正最后一个边方向
        return perm, ori

    def orbit_perm(self, state: np.ndarray) -> list[np.ndarray]:
        """
           从 sticker-level 状态生成 orbit_perm
           state: np.ndarray, shape = (6, n, n), 每个元素是颜色或编号
        """
        centers = CubeBase.get_center_rings(self.n)
        flat = []  # flatten 所有中心贴纸，顺序严格对应 center_orbits
        for fidx, rings in enumerate(centers):
            for ring in rings:
                for r, c, _ in ring:
                    flat.append(state[fidx, r, c])

        center_colors = np.array(flat, dtype=np.int32)

        orbits: list[list[int]] = CubeBase.center_orbits(self.n)
        orbits = [o for o in orbits if len(o) > 1]
        orbit_perm: list[np.ndarray] = []  # 每个 array 长度 = len(orbit)
        for orbit_idx in orbits:
            pieces = center_colors[orbit_idx]  # 映射到 0..k-1（orbit 内相对编号）
            order = np.argsort(pieces)
            rel = np.empty_like(order)
            rel[order] = np.arange(len(order))
            orbit_perm.append(rel)

        return orbit_perm

    def basic_generators(self) -> list:
        """基础生成元,逻辑层（axis, layer, direction）与几何层解耦，返回 move list,有限邻域 moves（减枝！！）"""
        moves = []
        for axis in range(3):
            for layer in self.center_layers:
                for direction in (-1, 1, 2):  # direction 只用 ±1，2 步可视为两步重复
                    moves.append((axis, layer, direction))
        return moves

    def generate_moves(self, length: int = 20):
        """生成打乱序列"""
        for _ in range(length):
            axis = random.choice(range(3))
            layer = random.choice(self.center_layers)
            direction = random.choice((-1, 1, 2))  # 1, 2, 3
            yield axis, layer, direction  # -> act/apply

    @classmethod
    def act_moves(cls, state: np.ndarray, moves: list | tuple):
        """
        连续作用,replace
        action space, action:群生成元
        (axis, layer, direction)
        s_{t+1} = T(s_t, a_t)
        """
        if not isinstance(moves, list):
            moves = [moves]
        for axis, layer, direction in moves:
            cls.rotate_core(state, axis, layer, direction)

    @staticmethod
    def invert_moves(moves: list[tuple]) -> list[tuple]:
        """move 的逆 将 moves 转成可还原的逆操作序列（反向 + 方向反）"""
        return [(axis, layer, -direction) for (axis, layer, direction) in reversed(moves)]

    @staticmethod
    def is_inverse(path: list[tuple], axis: int, layer: int, direction: int) -> bool:
        """
        is_redundant
        禁止与上一个动作在同一面（axis+layer）上连续转动且总效果为 0 mod 4
        两个动作加起来等价于什么都没做
        """
        # forbid immediate reversal
        if not path:
            return False
        pa, pl, pd = path[-1]
        if axis == pa and layer == pl:
            return (pd + direction) % 4 == 0
        return False

    @staticmethod
    def permutation_parity(perm: np.ndarray | list) -> int:
        """
        Return 0 for even, 1 for odd permutation
        0（偶置换）或 1（奇置换）
        """
        visited = np.zeros(len(perm), dtype=bool)
        parity = 0
        for i in range(len(perm)):
            if visited[i]:
                continue
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len > 0:
                parity ^= (cycle_len - 1) & 1  # 奇长循环贡献奇置换 (cycle_len % 2)
        return parity

    @staticmethod
    def encode_perm(perm: list[int]) -> int:
        """
        perm: 长度 n 的排列，值域 0..n-1
        返回 [0, n!-1]
        """
        n = len(perm)
        code = 0
        factor = 1
        for i in range(n - 1, -1, -1):
            cnt = 0
            for j in range(i + 1, n):
                if perm[j] < perm[i]:
                    cnt += 1
            code += cnt * factor
            factor *= (n - i)
        return code

    @class_cache(key=lambda code, n: (code, n))
    @staticmethod
    def decode_perm(code: int, n: int) -> list[int]:
        """
        code: 0 .. n!-1
        返回 perm，值域 0..n-1
        """
        elems = list(range(n))
        perm = [0] * n

        for i in range(n):
            fact = math.factorial(n - 1 - i)
            idx = code // fact
            code %= fact
            perm[i] = elems.pop(idx)
            # perm[i], perm[i + idx] = perm[i + idx], perm[i]

        return perm

    @staticmethod
    def comb_to_index(bits: list[int], n: int, k: int) -> int:
        """
        bits: 长度 n 的 0/1，恰有 k 个 1
        返回 [0, C(n,k))
        """
        idx = 0
        r = k
        for i in range(n):
            if bits[i]:
                idx += math.comb(n - i - 1, r)
                r -= 1
                if r == 0:
                    break
        return idx

    @staticmethod
    def index_to_comb(idx: int, n: int, k: int) -> list[int]:
        """逆字典序的降序解码"""
        bits = [0] * n
        r = k
        for i in range(n):
            if r == 0:
                break
            c = math.comb(n - i - 1, r)
            if idx >= c:
                bits[i] = 1
                idx -= c
                r -= 1
        return bits

    def dfs(self, state: np.ndarray, depth: int, bound, visited, path, max_depth: int = 25):
        key = self.encode_state(state)
        if key in visited:
            return math.inf, None
        visited.add(key)  # 对象状态快照

        h = self.heuristic(state)
        f = depth + h
        if f > bound:
            visited.remove(key)
            return f, None

        if h == 0:
            return True, path.copy()

        if depth >= max_depth:
            visited.remove(key)
            return math.inf, None

        best = math.inf
        for move in self.basic_generators():
            if self.is_inverse(path, *move):
                continue

            next_state = self.rotate_state(state, *move)
            path.append(move)

            t, sol = self.dfs(next_state, depth + 1, bound, visited, path, max_depth)

            path.pop()
            if t is True:
                return True, sol
            best = min(best, t)

        visited.remove(key)
        return best, None

    def heuristic_center(self, state: np.ndarray, r: int = 1) -> int:
        """
        计算中心错误数量，默认中心启发：统计以 mid 为中心的 (2k+1)x(2k+1) 区域中不等于 center color 的数目。
        r 控制区域大小 (2r+1)x(2r+1) 这里默认取 k= (n//2)//2 令中心区域足够大；可以改成只统计 (mid,mid) 周围 3x3。
        越小越接近目标（可用于 IDA*）
        """
        mid = self.mid
        k = max(1, r)  # 可调整为 1 (3x3) 或更大，跳过十字、边缘、角落，只取 3x3 / 5x5 / ... 中心块
        wrong = 0
        for f in range(6):
            face = state[f]
            target = self.solved[f, mid, mid]  # face[mid, mid]

            region = face[mid - k:mid + k + 1, mid - k:mid + k + 1]
            wrong += np.count_nonzero(region != target)
        return int(wrong)

    @classmethod
    def get_vars(cls):
        """获取类中的变量名"""
        return [name for name, value in vars(cls).items() if
                not (callable(value) or isinstance(value, (classmethod, staticmethod)))]

    @class_property('FACE_NORMAL')
    def face_normal(cls) -> dict[str, np.ndarray]:
        '''FACE_NORMAL = {
        'R': (1, 0, 0),
        'L': (-1, 0, 0),
        'U': (0, 1, 0),
        'D': (0, -1, 0),
        'F': (0, 0, 1),
        'B': (0, 0, -1),
        }'''
        mapping = {}
        for axis, (pos_face, neg_face) in enumerate(cls.AXIS_FACE):
            mapping[pos_face] = cls.AXIS_VEC[axis]
            mapping[neg_face] = -cls.AXIS_VEC[axis]
        return mapping

    @class_property('FACE_DEF')
    def face_def(cls) -> dict[str, tuple]:
        mapping = {}  # 六个面的法向和基向 dx, dy
        for face, normal in cls.face_normal.items():  # face 法向量
            up = cls.FACE_UP[face]
            right = np.cross(normal, up)  # 面内基向量（右手系）,right2 = world_up × forward = -right
            mapping[face] = (normal, right, up)

        return mapping

    @class_property('FACE_AXIS')
    def face_axis(cls) -> dict[str, tuple]:
        # 哪个面属于哪个轴
        return {face: (axis, side)
                for axis, pair in enumerate(cls.AXIS_FACE)
                for side, face in enumerate(pair)}

    @class_property('FACE_DEF_IDX')
    def face_idx(cls) -> dict:
        return {f: i for i, f in enumerate(cls.FACES)}

    @staticmethod
    def center_layers_list(n: int) -> list:
        mid, c = divmod(n, 2)
        if c == 1:
            return list(range(-mid, mid + 1))  # 奇数阶：中心在 0
        return [i for i in range(-mid, mid + 1) if i != 0]  # 偶数阶：无中心层

    @staticmethod
    def layer_index(layer: int, n: int) -> int:
        """数组索引空间"""
        mid, c = divmod(n, 2)
        layer_idx = layer + mid
        if c == 0 and layer >= mid:
            layer_idx -= 1
        return layer_idx

    @staticmethod
    def layer_to_logic(layer: int, n: int) -> float:
        """ 逻辑 / 拓扑坐标空间,逻辑世界坐标,半连续,abstract ℤ / ℤ +½ """
        c = n % 2
        if c == 1:  # 奇数阶：整数 → 整数
            return float(layer)
        # 偶数阶：整数 layer → 半整数坐标
        return layer - 0.5 if layer > 0 else layer + 0.5

    @staticmethod
    def layer_to_geom(layer: int, n: int) -> float:
        """几何模型空间,ℝ³ model space 动画"""
        return -n / 2 + 0.5 + layer

    @classmethod
    def face_rc_to_xyz(cls, face: str, r: int, c: int, n: int):
        """
        把 某个面上的 row/col 映射到世界坐标系 XYZ 返回 [x,y,z]
        把离散拓扑嵌入到连续空间中保持方向一致性
        face 上的 row / col 方向（固定定义）
        约定：
        #   row 方向 = Y×normal,Z
        #   col 方向 = normal×row
         n = 5 [-2, -1, 0, 1, 2]
         n = 4 [-1.5, -0.5, 0.5, 1.5]
        """
        # 面中心
        k = (n - 1) / 2.0
        dr = k - r  # r 向下 → y 减小
        dc = c - k  # c 向右 → x 增大
        normal, right, up = cls.face_def[face]

        # 世界坐标,normal 是面中心，dc*right + dr*up2 扩展到局部坐标
        pos = normal + dc * right + dr * up
        if n % 2 == 1:  # 离散化,保证严格整数
            return np.round(pos).astype(int)
        return np.sign(pos) * np.floor(np.abs(pos) + 0.5).astype(int)  # 原点缺失,parity 成为全局约束

    @classmethod
    def face_basis(cls, face: str):
        """
        带法向的面内正交基,面自身的几何事实
        确定局部行列方向,纯几何面内坐标系
        约定：
        - u_dir：面内向上（对应 row）
        - v_dir：面内向右（对应 col 正方向）
        - normal：面外法向
        """
        normal = cls.face_normal[face].astype(float)

        u_dir = cls.FACE_UP[face].astype(float)
        u_dir /= np.linalg.norm(u_dir)  # 归一化

        v_dir = np.cross(normal, u_dir)
        v_dir /= np.linalg.norm(v_dir)

        assert np.dot(np.cross(u_dir, v_dir), normal) > 0
        return normal, u_dir, v_dir

    @staticmethod
    def sticker_pos(normal, u_dir, v_dir, r: int, c: int, n: int) -> np.ndarray:
        """
         返回贴纸中心在世界坐标系中的 (x, y, z), world 连续浮点坐标
         与 face_rc_to_xyz 一致的面内基
         - 立方体中心在原点，坐标范围 [-center, center]
         - r, c: 从 0 到 n-1
         """
        center = (n - 1) / 2.0  # 连续浮点
        face_center = normal * center
        # 局部坐标映射到中心坐标 [-center, center]
        s_v = c - center  # col → right x:j
        s_u = r - center  # row → down y:i
        pos_rel = u_dir * s_u + v_dir * s_v
        return face_center + pos_rel  # float, 保留 ±0.5

    @classmethod
    def pos_to_face_rc(cls, pos: tuple, n: int) -> tuple:
        """lookup 直接查表  xyz → (face, r, c)"""

        # stickers = cls.get_face_stickers(n=n)
        # coord_map = {
        #     tuple(xyz.astype(int)): (cls.face_idx[f], r, c)
        #     for f, lst in stickers.items()
        #     for r, c, xyz in lst
        # }
        # return coord_map[pos]
        best_face = None
        best_dot = -float('inf')

        for f, nvec in cls.face_normal.items():
            dot = np.dot(pos, nvec)
            if dot > best_dot:
                best_dot = dot
                best_face = f

        face = best_face

        normal, u_dir, v_dir = cls.face_basis(face)
        u = np.dot(pos, u_dir)  # projection onto row direction
        v = np.dot(pos, v_dir)  # projection onto col direction

        center = (n - 1) / 2.0
        r = int(round(u + center))  # u_dir → row
        c = int(round(v + center))  # v_dir → col
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"out of face: {face}, r={r}, c={c}")

        return cls.face_idx[face], r, c

    @class_cache(cache_name='_LAYER_CACHE', key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def get_layer_stickers(cls, axis: int, layer: int, n: int = 3):
        """
        Returns a list of (fidx, r, c, pos) for stickers in the given layer along the axis.
        Assumes center at origin, layers from -x to x where x = (n-1)/2.
        """
        stickers = defaultdict(list)
        axis_vec = cls.AXIS_VEC[axis].astype(float)
        layer_coord = cls.layer_to_logic(layer, n)
        # R = cls.rot90_matrix(axis, 1)
        for idx, face in enumerate(cls.AXIS_STRIP[axis]):  # 已调整为统一 CCW 顺序
            normal, u_dir, v_dir = cls.face_basis(face)
            # normal_rot = R @ normal
            if abs(np.dot(normal, axis_vec)) > 1e-6:
                continue  # 整个面或不在该 layer
            for r in range(n):
                for c in range(n):
                    xyz = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)  # 中间态
                    # k = int(round(np.dot(xyz, axis_vec)))
                    # xyz_rot= R @ xyz
                    if np.isclose(xyz[axis], layer_coord, atol=1e-6):  # 中心坐标,abs(xyz[axis] - layer) < 1e-6
                        stickers[face].append((r, c, xyz))

        return stickers

    @class_cache(cache_name='_STRIP_CACHE', key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def strip_coords_from_axis(cls, axis: int, layer: int, n: int) -> list:
        """
          返回某 axis, face, layer 对应的条带坐标列表，已按中心原点计算
          返回: [(face, r, c), ...]  按 strip 顺序排列
          layer 在“世界坐标系”里定义
          row / col 在“face 局部坐标系”里定义
          strip 级别旋转 ≠ piece 级别置换
          几何排序 ≠ 拓扑顺序
        """
        strips = []
        axis_vec = cls.AXIS_VEC[axis].astype(float)
        face_stickers = cls.get_layer_stickers(axis, layer, n)
        for face, coords in face_stickers.items():
            if not coords:  # face_stickers.get(face, [])
                continue

            fidx = cls.face_idx[face]
            normal, u_dir, v_dir = cls.face_basis(face)
            # if np.dot(normal, axis_vec) < 0:  # 负侧面
            #     v_dir = -v_dir  # 只翻转向右方向,让所有 face 的 strip 方向在旋转轴视角下保持一致
            #     print('axis', axis, 'face', face, '负侧面统一翻转!')
            # u_dir = -u_dir
            strip_dir = np.cross(axis_vec, normal)
            strip_dir /= np.linalg.norm(strip_dir)  # 该面对应的旋转条带方向或法向量
            # 确定沿哪个方向 (v 或 u)，并计算 align
            align_u = np.dot(strip_dir, u_dir)
            align_v = np.dot(strip_dir, v_dir)
            if abs(align_u) > abs(align_v):
                key_dir = u_dir  # key = lambda x: x[0]  # r
                reverse = align_u < 0  # 如果 align <0,reverse 排序，使顺序沿局部正方向
            else:
                key_dir = v_dir  # key = lambda x: x[1]  # c
                reverse = align_v < 0
            # print('axis', axis, 'face', face, 'u', align_u, 'v', align_v, reverse)
            # coords_sorted = sorted(coords, key=lambda x: np.dot(x[2], strip_dir))
            coords_sorted = sorted(coords, key=lambda x: np.dot(x[2], key_dir), reverse=reverse)  # 世界坐标投影排序
            strip = [(fidx, r, c) for r, c, _ in coords_sorted]
            strips.append(strip)

        return strips

    @classmethod
    @class_status('参考方法')
    def strip_coords_from_axis_old(cls, axis: int, layer: int, n: int) -> list:
        face_normal = cls.face_normal()
        axis_vec = cls.AXIS_VEC[axis]
        strips = []
        # prev_strip_dir = None  # 用于检查连续性
        # prev_v_dir=None
        for idx, face in enumerate(cls.AXIS_STRIP[axis]):
            fidx = cls.face_idx[face]  # self.FACES.index(face)
            normal = face_normal[face]
            # 相邻面之间“向右”和“向上”的递进方向不连续,选择一致的参考系
            reverse = (axis == 2 and idx % 2 == 1)  # 第1、3个面（R 和 L 对于 Z 轴）,每隔一个面（奇数位）需要翻转 u_dir，使“向右”方向在环路上连续
            # 收集该 face 上属于 layer 的所有贴纸
            coords = []
            for r in range(n):
                for c in range(n):
                    xyz = cls.face_rc_to_xyz(face, r, c, n)
                    if np.isclose(xyz[axis], layer):
                        if reverse:
                            coords.append((c, r, xyz))  # Z₂ flip 对整个面内平面做一次 180° 旋转
                        else:
                            coords.append((r, c, xyz))

            if not coords:
                continue

            # center = np.mean([p for *_, p in coords], axis=0)
            # proj = center - dot(center, axis_vec) * axis_vec  # 投影到旋转平面（垂直于 axis）
            # 条带内部顺序,判断是水平行还是垂直列
            rs = {s[0] for s in coords}
            cs = {s[1] for s in coords}
            if len(rs) == 1:  # 水平行,按列排
                r = rs.pop()
                strip = [(fidx, r, c) for _, c, _ in sorted(coords, key=lambda x: x[1])]
            elif len(cs) == 1:  # 竖直列,按行排
                c = cs.pop()
                strip = [(fidx, r, c) for r, _, _ in sorted(coords, key=lambda x: x[0])]
            else:
                # 整个 face（最外层），按行排
                for r in sorted(rs):
                    strip = [(fidx, r, c) for rr, c, _ in coords if rr == r]
                    strips.append(strip)
                continue

            strip_dir = np.cross(axis_vec, normal)  # 该面对应的旋转条带方向或法向量
            # 检查与前一面的连续性: 如果 prev_strip_dir 存在, dot(strip_dir, prev_u_or_v) <0 则翻转
            # if prev_strip_dir is not None:
            #     if np.dot(strip_dir, prev_v_dir) < 0:  # 如果与前面的 v_dir 反向, 翻转本面基
            #         u_dir, v_dir = -u_dir, -v_dir  # 180° 旋转, 保持正交
            #         strip_dir = -strip_dir  # 相应翻转 strip_dir 以匹配
            #
            # # 更新 prev for next
            # prev_strip_dir = strip_dir
            # prev_v_dir = v_dir  # 或 u_dir, 取决于环是否沿col 方向绕,由于约定 v_dir=向右 (col), 用 v_dir 作为“前进”代理
            # dir_idx = np.argmax(np.abs(strip_dir))
            inc_dir = coords[-1][2] - coords[0][2]  # last - first
            if np.dot(inc_dir, strip_dir) < 0:  # 世界坐标方向向量，用整个条带判断是否需要反转
                strip.reverse()  # SO(3) 群, 逆向轴方向进场，需要 reverse
            strips.append(strip)
        return strips

    @classmethod
    def rotate_slice(cls, state: np.ndarray, axis: int, layer: int, shift: int, n: int = None):
        """旋转一层, inplace,当前的 strip 机制：丢失了顺序信息"""
        if shift == 0:
            return
        n = n or state.shape[1]
        # 生成每一条 strip 的坐标列表 (f, r, c)
        strips = cls.strip_coords_from_axis(axis, layer, n)  # 获取每一面条带的坐标序列
        # 读取每条 strip 的值（list of lists）,by strip_coords
        vals = [[state[f, r, c] for (f, r, c) in strip] for strip in strips]
        # 循环环移,正向移位
        vals = vals[-shift:] + vals[:-shift]  # CCW rotation
        # 写回
        for strip, val in zip(strips, vals):
            for (f, r, c), v in zip(strip, val):
                state[f, r, c] = v

    @classmethod
    def rotate_core(cls, state: np.ndarray, axis: int, layer: int, direction: int):
        """
        inplace 版本
        生成元 move(axis, layer, dir), SE(3) 中旋转生成元在立方晶格上的离散表示
          state ∈ Sticker(SO(3))
          axis ∈ {0,1,2} 离散化的旋转轴方向（单位向量）: 'x', 'y', 'z'
          layer ∈ {..., -2, -1, 0, 1, 2, ...} 沿旋转轴法向方向的离散标量坐标
          dir ∈ {+1, -1, 2}  θ ∈ {π/2, -π/2, π} 离散化的旋转角 / 旋量大小
        贴纸态 ≠ CubieState（不可逆）
        """
        d = direction % 4
        if d == 0:
            return  # 旋转 0 次
        n = state.shape[1]
        mid = n // 2

        # 处理最外层面本体旋转，x轴 → R/L, y轴 → U/D ,z轴 → F/B
        if abs(layer) == mid:
            side = 0 if layer > 0 else 1  # layer == +mid → 使用几何正向面,[0]（pos face）
            dd = -d if side == 0 else d  # 方向修正：视角翻转补偿  d if face in ["U", "R", "F"] else -d
            face = cls.AXIS_FACE[axis][side]
            fidx = cls.face_idx[face]
            cls.rotate_inplace(state[fidx], dd)  # 使用的是「观察者正对该面」的顺时针定义
        # 中层处理
        cls.rotate_slice(state, axis, layer, shift=d, n=n)

    @classmethod
    def rotate_state(cls, state: np.ndarray, axis: int, layer: int, direction: int) -> np.ndarray:
        """
        纯函数版本：不修改传入 state，返回新状态 next_state 副本（已经应用旋转）。rotate_state 只存在于贴纸层
        用于 BFS/IDA*/并行扩展时的安全调用。区别实例方法“就地旋转”，完全独立
        """
        arr = state.copy()
        cls.rotate_core(arr, axis, layer, direction)
        return arr  # new_state

    @classmethod
    def get_corner_faces(cls, pos_sign: tuple[int, int, int]) -> tuple[str, str, str]:
        '''
        dict(enumerate(cls.AXIS_FACE)) 轴到面映射: {0: ('R','L'), 1: ('U','D'), 2: ('F','B')}
        标准顺序，确保 solved 时 U/D 在位置 0,第1位永远是 U 或 D（由 Y)
        U面：从 +Y 看（向上），D面：从 -Y 看（从下方向上看）
        先Y (U/D)，然后X Z (右手: 对于U, Z→X顺时针)
        '''
        sx, sy, sz = pos_sign  # (sx, sy, sz) ∈ {±1, ±1, ±1}
        # 索引：正方向 -> 0, 负方向 -> n-1,每个轴取正方向的面（+1 取正，-1 取反）
        idx = lambda s: 0 if s > 0 else - 1
        face_x = cls.AXIS_FACE[0][idx(sx)]  # R/L
        face_y = cls.AXIS_FACE[1][idx(sy)]  # U/D
        face_z = cls.AXIS_FACE[2][idx(sz)]  # F/B
        # 三个方向向量（单位向量）
        vec_x = np.array([sx, 0, 0])
        vec_z = np.array([0, 0, sz])
        outward_normal = np.array([0, -sy, 0])  # -vec_y
        # 计算叉乘：outward_normal × vec_x 应该指向 vec_z 的方向（右手规则）
        cross = np.cross(outward_normal, vec_x)
        if np.dot(cross, vec_z) > 0:  # 顺序正确：vec_x  →  vec_z 是顺时针
            return face_y, face_x, face_z  # 102
        return face_y, face_z, face_x  # 120

    @class_property('CORNER_FACE_CYCLE')
    def corner_face_cycle(cls) -> list[tuple]:
        '''
         标准顺序，确保 solved 时 U/D 在位置 0,第1位永远是 U 或 D（由 Y
         生成角块, 对应：UFR, URB, UBL, ULF, DLF, DFR, DRB, DBL
         ('U', 'R', 'F'),0: URF
         ('U', 'F', 'L'),1: UFL
         ('U', 'L', 'B'),2: ULB
         ('U', 'B', 'R'),3: UBR
         ('D', 'F', 'R'),4: DFR
         ('D', 'L', 'F'),5: DLF
         ('D', 'B', 'L'),6: DBL
         ('D', 'R', 'B'),7: DRB
         '''
        return [cls.get_corner_faces(pos_sign) for pos_sign in cls.CORNER_POS_SIGNS]

    @classmethod
    def get_edge_faces(cls, pos_sign: tuple[int, int, int]) -> tuple[str, str]:
        """
        针对边块，只涉及两个面: 非零坐标的两个轴对应的面。
        对于一个边块的位置符号 (sx, sy, sz)，其中一个坐标为0，另两个为 ±1
        顺序：先 Y 轴（U/D 优先），然后按右手规则排序剩余两个。
        """
        sx, sy, sz = pos_sign
        assert sx * sy * sz == 0  # 必须有一个为0
        assert len([i for i, s in enumerate([sx, sy, sz]) if s != 0]) == 2, "Edge must have exactly two non-zero coords"
        idx = lambda s: 0 if s > 0 else - 1
        face_x = cls.AXIS_FACE[0][idx(sx)]  # R if +x else L
        face_y = cls.AXIS_FACE[1][idx(sy)]  # U if +y else D
        face_z = cls.AXIS_FACE[2][idx(sz)]  # F if +z else B
        # 优先 U/D → F/B → R/L（与社区最常见命名顺序一致）
        if sy != 0:  # U/D 边的四个
            primary = face_y  # U or D
            if sz != 0:  # UF / UB / DF / DB
                secondary = face_z
            else:  # UR / UL / DR / DL
                secondary = face_x
        else:  # 中层边：FR, FL, BL, BR
            primary = face_z  # F or B
            secondary = face_x  # R or L

        return primary, secondary

    @class_property('EDGE_FACE_CYCLE')
    def edge_face_cycle(cls) -> list[tuple]:
        '''
         标准顺序，顺序：先 Y 轴（U/D 优先），然后按右手规则排序剩余两个。
         参考色放第一位（e.g., [U, R], [F, R]）
         生成角块, 对应：
        EDGE_INDEX_ORDER = [
        ('U', 'R'), ('U', 'F'), ('U', 'L'), ('U', 'B'),
        ('F', 'R'), ('F', 'L'), ('B', 'L'), ('B', 'R'),
        ('D', 'R'), ('D', 'F'), ('D', 'L'), ('D', 'B'),
        ]  # edge cubie 的“世界坐标顺序”
        '''
        return [cls.get_edge_faces(pos_sign) for pos_sign in cls.EDGE_POS_SIGNS]

    @class_cache(cache_name='_FACE_CACHE', key=lambda n: n)
    @classmethod
    def get_face_stickers(cls, n: int):
        """sticker_pos_from_face,不依赖 axis / layer / strip，返回贴纸中心的 3D 世界坐标"""
        stickers = defaultdict(list)
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            # origin = normal * (n / 2)
            for r in range(n):
                for c in range(n):
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    # p = origin + dx * (c - n / 2 + 0.5) + dy * (r - n / 2 + 0.5)
                    stickers[face].append((r, c, pos))  # 抽象网格顺序
        return stickers

    @classmethod
    def get_corner_stickers(cls, n: int) -> list:
        """
        不考虑面环路连续性,几何点本身不应该知道邻接
        """
        stickers = []
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            for r in (0, n - 1):
                for c in (0, n - 1):
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    stickers.append((face, r, c, pos))
        return stickers

    @classmethod
    def get_edge_stickers(cls, n: int) -> list[tuple]:
        """
        返回所有 edge 贴纸:
        [(face, r, c, pos), ...]
        不考虑邻接 / 环路 / strip
        """
        stickers = []
        mid, c = divmod(n, 2)
        center = mid if c == 1 else mid - 1
        # 奇数阶选 layer=0（中心中线）,偶数阶选（偏左/下的中线）
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            # 四条边，去掉角
            for r, c in (
                    (0, center),  # 上边
                    (center, n - 1),  # 右边
                    (n - 1, center),  # 下边
                    (center, 0),  # 左边
            ):
                pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                stickers.append((face, r, c, pos))
        return stickers

    @class_cache(cache_name='_CENTER_CACHE', key=lambda n: n)
    @classmethod
    def get_center_stickers(cls, n: int) -> list[tuple]:
        """
        返回每个面的中心贴纸坐标,6
        """
        mid, c = divmod(n, 2)  # 中心永远是几何中心
        cr = cc = mid if c == 1 else mid - 1  # 奇数取正中，偶数取偏左/上
        stickers = []
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            pos = cls.sticker_pos(normal, u_dir, v_dir, cr, cc, n)
            stickers.append((cls.face_idx[face], cr, cc, pos))  # 通常不需要 pos，直接返回坐标
        return stickers

    @class_cache(cache_name='_CENTER_RINGS_CACHE', key=lambda n: n)
    @classmethod
    def get_center_rings(cls, n: int) -> list[list[list[tuple]]]:
        """
        返回每个 face 的所有中心贴纸坐标（带 pos）
        返回结构: list[face_idx] -> list[rings] -> list[(r, c, pos)]
        - face_idx 0~5 对应 cls.FACES 顺序
        - 最内层 ring 通常是中心单块（n奇数时只有一个元素）
        """
        rings = []
        mid = n // 2
        max_dist = mid - 1
        for face in cls.FACES:
            normal, u_dir, v_dir = cls.face_basis(face)
            face_rings = [[] for _ in range(max_dist + 1)]  # 每个距离一个 ring
            # 遍历内层区域
            for r in range(1, n - 1):
                for c in range(1, n - 1):
                    dist = max(abs(r - mid), abs(c - mid))  # 计算到中心的曼哈顿距离
                    pos = cls.sticker_pos(normal, u_dir, v_dir, r, c, n)
                    face_rings[dist].append((r, c, pos))
            # for ring in face_rings:
            #     ring.sort(key=lambda x: (x[0], x[1]))# 按 row, col 排序
            rings.append(face_rings)
        return rings

    @class_cache(cache_name='_CENTER_ORBITS_CACHE', key=lambda n: n)
    @classmethod
    def center_orbits(cls, n: int) -> list[list[int]]:
        """
        返回 center orbits（按 ring / face 划分）
        每个 orbit 是 center index 的列表
        """
        centers = []
        for face in range(6):
            for r in range(1, n - 1):
                for c in range(1, n - 1):
                    centers.append((face, r, c))

        index_map = {p: i for i, p in enumerate(centers)}  # SOLVED_CENTER_MAP

        face_rings = cls.get_center_rings(n)
        orbits = []
        for fidx, rings in enumerate(face_rings):
            for ring in rings:
                orbits.append([index_map[(fidx, r, c)] for r, c, _ in ring])
        total_centers = sum(len(o) for o in orbits)
        assert total_centers == (n - 2) ** 2 * 6, f"Center count mismatch: {total_centers}"
        return orbits

    @class_cache(cache_name='EDGES_CACHE', key=lambda n: n)
    @classmethod
    def edge_coords(cls, n: int) -> list[list[tuple[int, int, int]]]:
        """
        返回固定顺序的 12 条边的贴纸坐标，按标准顺序：
        [[(face_idx, r, c), (face_idx, r, c)],
          ...
        ]
        生成魔方所有 central edges 的贴纸坐标,12组，每条 edge 两个贴纸
        奇：基于 3D 世界坐标 + EDGE_POS_SIGNS 的 edge 定义
        偶：顺序严格对应 EDGE_FACE_CYCLE 的标准顺序（UR, UF, ..., DB）
        """
        stickers = cls.get_edge_stickers(n)
        assert len(stickers) == 24, f"Expected 24 edge pieces, got {len(stickers)}"
        result = [[] for _ in range(len(cls.EDGE_POS_SIGNS))]  # list[12][2]
        if n % 2 == 1:
            edges = {k: [] for k in cls.EDGE_POS_SIGNS}  # 12 条 edge
            for face, r, c, pos in stickers:
                sign = np.sign(pos).astype(int)
                sign = tuple(sign.tolist())
                if sign in edges:
                    edges[sign].append((face, r, c, pos))

            for eid, sign in enumerate(cls.EDGE_POS_SIGNS):
                group = edges[sign]
                if len(group) != 2:
                    raise ValueError(f"illegal edge {sign}: {len(group)}")

                center = np.mean([p for *_, p in group], axis=0)  # 几何一致性校验
                assert tuple(np.sign(center)) == sign, f"Sign mismatch for edge {eid}: expected {sign}"
                # 排序：哪个贴纸法向量更“外”，就排前
                group.sort(key=lambda x: np.argmax(np.abs(x[3])))
                result[eid] = [(cls.face_idx[f], r, c) for f, r, c, _ in group]
        else:
            pos_groups = defaultdict(list)  # 按位置坐标（四舍五入）分组
            for face, r, c, pos in stickers:
                key = tuple(np.round(pos, decimals=0).astype(int))  # key 是纯坐标，不带 sign
                pos_groups[key].append((face, r, c, pos))
                # for s in cls.EDGE_POS_SIGNS:
                #     if np.dot(s, np.sign(pos)) == 2:
                #         break

            # 筛选出正好有两个贴纸的位置组（这就是中棱）
            edge_groups = [g for g in pos_groups.values() if len(g) == 2]
            if len(edge_groups) != 12:
                raise ValueError(f"Expected 12 edge groups, got {len(edge_groups)}")

            standard_edges = cls.edge_face_cycle()  # list[tuple[str,str]] 长度12，如 [("U","R"), ("U","F"), ...]
            # 建立 边索引 ← 面对集合 的快速查找,预先生成标准顺序映射
            face_pair_to_id = {frozenset(pair): idx for idx, pair in enumerate(standard_edges)}
            for group in edge_groups:
                if len(group) != 2:  # 是长度为 2 的列表，每项是 (face_name, r, c, pos_vec)
                    raise ValueError(f"Edge piece should have exactly 2 stickers, got {group}")
                faces = frozenset(g[0] for g in group)
                eid = face_pair_to_id.get(faces)
                if eid is None:
                    raise ValueError(f"Unknown edge face pair: {faces}")
                group.sort(key=lambda x: np.argmax(np.abs(x[3])))  # 根据 pos 排序：哪个轴的绝对值最大，就认为它更“外侧”
                std_f1 = standard_edges[eid][0]
                if group[0][0] != std_f1:  # 根据标准顺序决定贴纸存储顺序
                    print(group)
                    group.reverse()

                result[eid] = [(cls.face_idx[f], r, c) for f, r, c, _ in group]

        assert len(result) == 12, f"Expected 12 edges, got {len(result)}:{result}"
        return result  # List[12] of [[(fidx,r,c), (fidx,r,c)]]

    @class_cache(cache_name='CORNERS_CACHE', key=lambda n: n)
    @classmethod
    def corner_coords(cls, n: int) -> list[list]:
        """
        [[(face_idx, r, c), (face_idx, r, c), (face_idx, r, c)],
          ...
        ]
        xyz pos(center)-> corner_id
        8 个角法向量组合: 每个角由三个轴的正负组成,3轴各2方向
        signs = [(sx, sy, sz)
                 for sx in (+1, -1)
                 for sy in (+1, -1)
                 for sz in (+1, -1)]  # list(product([1, -1], repeat=3))
        定义基准：UFR
        corner_ori = 0   如果 U 或 D sticker 在 U/D 面
        corner_ori = 1   顺时针旋转
        corner_ori = 2   逆时针旋转
        """
        stickers = cls.get_corner_stickers(n)
        corners = {k: [] for k in cls.CORNER_POS_SIGNS}  # 8 个角

        for face, r, c, pos in stickers:
            signs = np.sign(np.round(pos, decimals=5)).astype(int)
            sx, sy, sz = signs
            corners[(sx, sy, sz)].append((face, r, c, pos))

        result = [[] for _ in range(len(cls.CORNER_POS_SIGNS))]
        for cid, sign in enumerate(cls.CORNER_POS_SIGNS):
            group = corners[sign]
            if len(group) != 3:
                raise ValueError(f"illegal corner {sign}: {len(group)}")
            center = np.mean([p for *_, p in group], axis=0)
            assert tuple(np.sign(center)) == sign, f"Sign mismatch for corner {cid}: expected {sign}"
            # 已正确分组后,再做主轴排序,能让起点更一致
            group.sort(key=lambda x: np.argmax(np.abs(x[3])))

            # 用法向量排序（代替 pos） 叉积排序，确保右手定则顺序
            v_corner = np.array(sign, dtype=float)
            # 先找该角的“U/D”面（如果有），作为起点 标准顺序 standard: ['U', 'F', 'R']
            face_to_item = {item[0]: item for item in group}
            start_face = next((f for f in ('U', 'D') if f in face_to_item), list(face_to_item)[0])  # 优先用 U/D 面作为起点
            start = face_to_item[start_face]

            sorted_group = [start]  # 起点
            remaining = [item for item in group if item != start]

            for _ in range(2):
                curr_f = sorted_group[-1][0]
                curr_normal = cls.face_normal[curr_f]  # curr_pos = sorted_group[-1][3]
                for i, next_item in enumerate(remaining):
                    next_f = next_item[0]
                    next_normal = cls.face_normal[next_f]  # next_pos = next_item[3]
                    cross = np.cross(curr_normal, next_normal)
                    if np.dot(cross, v_corner) < 0:  # 右手定则 顺时针
                        sorted_group.append(remaining.pop(i))
                        break
                else:  # 如果叉积失败，fallback 到面优先级
                    remaining.sort(key=lambda x: cls.face_idx[x[0]])
                    sorted_group.extend(remaining)
                    # raise ValueError(f"Cannot find consistent order for corner {cid}")
                    break

            result[cid] = [(cls.face_idx[f], r, c) for f, r, c, _ in sorted_group]

        assert len(result) == 8, f"Expected 8 corners, got {len(result)}"
        print([''.join([cls.FACES[y[0]] for y in x]) for x in result], '\n', cls.corner_face_cycle())
        # ['URF', 'UFL', 'ULB', 'UBR', 'DFR', 'DLF', 'DBL', 'DRB']
        return result

    @staticmethod
    def rotate_inplace(mat: np.ndarray, direction: int = 1) -> None:
        """
        rotate square matrix mat by direction*90 degrees clockwise.
        dir_sign: integer (positive/negative allowed). direction % 4 gives action:
          0 -> no-op
          1 -> 90 deg CW
          2 -> 180 deg
          3 -> 270 deg CW (or 90 CCW)
        The function mutates mat and returns None.
        和 rotate_coord 的旋转方向是完全一致的，都是顺时针（CW）。
        """
        dir_sign = direction % 4
        if dir_sign == 0:
            return
        elif dir_sign == 1:
            mat[:] = np.flip(mat.T, axis=1)  # 90 CW : transpose + flip LR
        elif dir_sign == 2:
            mat[:] = np.flip(np.flip(mat, axis=0), axis=1)  # 180 : flip LR + flip UD
        elif dir_sign == 3:  # i.e. 270 CW = 90 CCW
            mat[:] = np.flip(mat.T, axis=0)  # 90 CCW : transpose + flip UD
        else:
            raise ValueError(f"Invalid direction:{direction}")

    @staticmethod
    def rotate_coord(coord, axis: int, dir_sign: int = 1):
        """
        直接对一个3D坐标点[x, y, z]应用90度旋转，返回新坐标
        支持 cw (1) 和 ccw (-1) 的坐标旋转 right-hand
        dir_sign = 1 表示CW，dir_sign = -1 表示CCW
        """
        x, y, z = coord
        if axis == 0:  # X (R/L)  around x
            if dir_sign == 1:  # cw
                return [x, z, -y]
            else:  # ccw
                return [x, -z, y]
        elif axis == 1:  # Y (U/D)
            if dir_sign == 1:
                return [-z, y, x]
            else:
                return [z, y, -x]
        elif axis == 2:  # Z (F/B)
            if dir_sign == 1:
                return [y, -x, z]
            else:
                return [-y, x, z]
        raise ValueError("Invalid axis")

    @staticmethod
    def rot90_matrix(axis: int, dir: int) -> np.ndarray:
        """
        生成一个3x3的旋转矩阵,dir = +1 表示逆时针（CCW，从轴正方向看），dir = -1 表示顺时针（CW）
        axis: 0=x, 1=y, 2=z
        k: +1 = CCW, -1 = CW （从轴正方向看 +axis）
        """
        assert dir in (+1, -1)
        if axis == 0:  # X
            return np.array([
                [1, 0, 0],
                [0, 0, -dir],
                [0, dir, 0],
            ])
        if axis == 1:  # Y
            return np.array([
                [0, 0, dir],
                [0, 1, 0],
                [-dir, 0, 0],
            ])
        if axis == 2:  # Z
            return np.array([
                [0, -dir, 0],
                [dir, 0, 0],
                [0, 0, 1],
            ])
        raise ValueError(axis)

    @staticmethod
    def rotation_matrix(angle: tuple | np.ndarray) -> np.ndarray:
        # --- 合成基本旋转矩阵,整体旋转或欧拉角旋转 ---
        ax, ay, az = angle
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)]
        ])
        Ry = np.array([
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)]
        ])
        Rz = np.array([
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1]
        ])
        if np.abs(np.abs(ay) - np.pi / 2) < 1e-6:
            print(f"接近万向节锁! ay={ay} 接近 ±90°")
        return Rz @ Ry @ Rx  # 从右向左执行，实际顺序是 X -> Y -> Z

    @class_property('ROTATION_MATRICES')
    def rotation_matrices(cls) -> np.ndarray:
        """
        生成所有：3x3 正交矩阵，直接用于坐标变换,24 个 SO(3) 旋转
        立方体的 24 个旋转矩阵 (SO(3) 部分, det = +1)
        """
        # rot_x_90 = CubeBase.rot90_matrix(0, 1)
        # rot_y_90 = CubeBase.rot90_matrix(1, 1)
        # rot_z_90 = CubeBase.rot90_matrix(2, 1)
        # R = rz @ ry @ rx
        mats = []
        import itertools
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([-1, 1], repeat=3):
                M = np.zeros((3, 3), dtype=np.int8)
                for i, j in enumerate(perm):
                    M[i, j] = signs[i]
                if abs(np.linalg.det(M) - 1.0) < 1e-8 and np.allclose(M @ M.T, np.eye(3), atol=1e-8):
                    mats.append(M)

        mats.sort(key=lambda M: -int(np.linalg.det(M)))
        return np.array(mats)  # 24 个

    @class_property('SYMMETRY_MATRICES')
    def symmetry_matrices(cls) -> np.ndarray:
        """
        生成完整的 48 个立方体对称矩阵（24 旋转 + 24 镜像）
        每个矩阵 M: 3×3 带符号置换矩阵, det = ±1
        """
        rot_mats = cls.rotation_matrices  # (24, 3, 3)
        mirror = np.diag(np.array([1, 1, -1], dtype=np.int8))
        sym_mats = []
        for R in rot_mats:
            sym_mats.append(R)
            sym_mats.append(mirror @ R)  # 镜像版本
        return np.array(sym_mats)  # shape: (48, 3, 3)

    @classmethod
    def apply_rotation(cls, state: np.ndarray, R: np.ndarray):
        """
        对整个 sticker 数组应用 3x3 旋转矩阵 R（坐标变换）
        - state: (6, n, n) 贴纸数组
        - R: 3x3 旋转矩阵
        返回旋转后的 sticker
        把每个 sticker 的位置 (x,y,z) 应用 R 变换，然后映射回新 face/row/col
        """
        n = state.shape[1]
        new_state = np.zeros_like(state)
        face_stickers = cls.get_face_stickers(n=n)
        for fidx, face in enumerate(cls.FACES):
            for r, c, xyz in face_stickers[face]:
                xyz_rot = R @ xyz
                xyz_rot = tuple(np.round(xyz_rot).astype(int))
                new_f, new_r, new_c = cls.pos_to_face_rc(xyz_rot, n)
                new_state[new_f, new_r, new_c] = state[fidx, r, c]
        return new_state

    @classmethod
    def normalize_align(cls, state: np.ndarray, max_depth: int = 8):
        """
        使用 BFS 遍历 layer=0 旋转，使 U 和 F 面中心对齐
        角块保持不动
        返回：
        - new_state: 对齐后的 state
        - move_list: 初步旋转列表 [(axis, layer, dir), ...]
        """
        n = state.shape[1]
        mid = n // 2

        U = cls.face_idx['U']  # 0
        F = cls.face_idx['F']  # 2

        solved = cls.solved(n=n)
        target_u = solved[U, mid, mid]
        target_f = solved[F, mid, mid]
        if state[U, mid, mid] == target_u and state[F, mid, mid] == target_f:
            return state.copy(), []

        # 只绕 xyz 三个轴做 layer=0 旋转
        moves = [(axis, 0, dir) for axis in [0, 1, 2] for dir in [1, 2, -1]]

        q = deque()
        q.append((state.copy(), []))
        visited = set()
        visited.add(tuple(state.flatten()))

        while q:
            current_state, move_list = q.popleft()
            if len(move_list) > max_depth:
                continue
            # 检查目标 U/F 对齐
            if current_state[U, mid, mid] == target_u and current_state[F, mid, mid] == target_f:
                return current_state, move_list

            # 遍历所有 layer=0 的旋转
            for move in moves:
                new_state = cls.rotate_state(current_state, *move)
                state_key = tuple(new_state.flatten())
                if state_key not in visited:
                    visited.add(state_key)
                    q.append((new_state, move_list + [move]))

        raise ValueError("无法对齐 U/F 面")

    @class_property('SYMMETRY_INVERSE_ID')
    def symmetry_inverse_id(cls) -> np.ndarray:
        """symmetry_inverse_id[i] = j 表示第 i 个对称的逆是第 j 个"""
        mats = cls.symmetry_matrices
        inv = np.zeros(48, dtype=np.int8)
        for i, M in enumerate(mats):
            MT = M.T
            for j, N in enumerate(mats):
                if np.array_equal(MT, N):
                    inv[i] = j
                    break
        return inv

    @classmethod
    def apply_symmetry(cls, state: np.ndarray, sym_id: int) -> np.ndarray:
        """将第 sym_id 种对称操作作用于贴纸状态 (6, n, n)"""
        M = cls.symmetry_matrices[sym_id]  # (3, 3)
        n = state.shape[1]
        mid = (n - 1) / 2.0
        new_state = np.zeros_like(state)

        for old_fidx in range(6):
            old_face = cls.FACES[old_fidx]
            normal_old = cls.face_normal[old_face]
            normal_new = M @ normal_old
            # 通过法向量匹配找到新面
            new_fidx = None
            for fidx, face in enumerate(cls.FACES):
                if np.array_equal(normal_new, cls.face_normal[face]):
                    new_fidx = fidx
                    break
            assert new_fidx is not None, f'sym {sym_id}: face {old_face} normal not matched'
            # 计算面内 2D 变换: (dr, dc) -> (dr_new, dc_new)
            # pos_new = M @ pos_old  ⇒  u_new·dr_new + v_new·dc_new = Mu·dr + Mv·dc
            _, u_old, v_old = cls.face_basis(old_face)
            _, u_new, v_new = cls.face_basis(cls.FACES[new_fidx])
            Mu, Mv = M @ u_old, M @ v_old
            a, b = np.dot(Mu, u_new), np.dot(Mv, u_new)  # dr 的 u_new 分量 + dc 的 u_new 分量
            c, d = np.dot(Mu, v_new), np.dot(Mv, v_new)  # dr 的 v_new 分量 + dc 的 v_new 分量
            # 应用变换到每个贴纸
            for r in range(n):
                for col in range(n):
                    dr, dc = r - mid, col - mid
                    r_new = int(round(mid + a * dr + b * dc))
                    c_new = int(round(mid + c * dr + d * dc))
                    if 0 <= r_new < n and 0 <= c_new < n:
                        new_state[new_fidx, r_new, c_new] = state[old_fidx, r, col]
        return new_state

    @class_cache(key=lambda face, n: (face, n))
    @classmethod
    def face_quads(cls, face: str, n: int) -> list:
        """
        生成某一面上的所有小方块的四边形坐标->get_face_stickers
        返回给定面 U/D/F/B/L/R 上 n×n 个小方块的 3D quad 数组
        """
        normal, dx, dy = cls.face_def[face]
        origin = normal * (n / 2)

        result = []
        for i in range(n):
            for j in range(n):
                # 当前小贴纸左上角中心点
                p = origin + dx * (j - n / 2 + 0.5) + dy * (i - n / 2 + 0.5)
                # 小方块 4 个角
                quad = [
                    p + (-dx - dy) * 0.5,
                    p + (dx - dy) * 0.5,
                    p + (dx + dy) * 0.5,
                    p + (-dx + dy) * 0.5,
                ]
                result.append(quad)
        return result

    @classmethod
    def rotate_around_layer(cls, quad: np.ndarray, axis: int, layer_geom: float, ang: float) -> np.ndarray:
        """
        根据给定的旋转轴和角度生成旋转矩阵,任意层的局部轴
        计算旋转层的中心点（该层相对于立方体中心的位置）
        对该层的每个点进行旋转，保证旋转发生在该层平面上
        R = I + (sinθ) * K + (1 - cosθ) * K^2 罗德里格斯公式推导
        """

        # rotate points around the plane of the layer (centered at layer plane)
        # compute layer plane center
        def axis_rot_matrix(axis_vec: np.ndarray, theta: float):
            # Rodrigues' rotation formula
            k = axis_vec / np.linalg.norm(axis_vec)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            I = np.eye(3)
            return I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

        rot = axis_rot_matrix(cls.AXIS_VEC[axis], ang)
        center = np.zeros(3)
        center[axis] = layer_geom  # translation to layer center:layer - n / 2 + 0.5
        return np.array([rot @ (v - center) + center for v in quad])

    @class_cache(key=lambda axis, layer, n: (axis, layer, n))
    @classmethod
    def layer_sticker_set(cls, axis: int, layer: int, n: int):
        stickers = cls.get_layer_stickers(axis, layer, n)
        return {
            (f, r, c)
            for f, lst in stickers.items()
            for r, c, _ in lst
        }

    @classmethod
    def should_rotate_by_sticker(cls, face: str, r: int, c: int, axis: int, layer: int, n: int):
        return (face, r, c) in cls.layer_sticker_set(axis, layer, n)

    @classmethod
    @class_status('待完成')
    def rotated_coord(cls, quad: np.ndarray, axis: int, layer: int, n: int, ang: float):
        """
        返回动画阶段的临时颜色
        quad: 4 个角的世界坐标 4x3 的 3D 点 → 只是渲染,贴纸的可视外壳
        axis/layer/ang: 当前旋转信息
        """
        # 找到该 quad 对应的逻辑位置
        # coords = quad[:, axis]
        # min_c, max_c = coords.min(), coords.max()
        stickers = cls.get_layer_stickers(axis, layer, n)  # dict[face] = [(r,c,pos), ...]
        for face, lst in stickers.items():
            for r, c, pos in lst:
                center = np.mean(quad, axis=0)
                # 如果该 quad 的中心接近任何一个贴纸的 pos，就认为它属于旋转层
                if np.allclose(center, pos, atol=0.5):  # 判断 quad 是否接近 pos
                    #  根据旋转角度计算新逻辑位置
                    layer_coord = pos[axis]
                    if abs(ang - np.pi / 2) < 1e-6:  # 顺时针 90°
                        new_r, new_c = c, n - 1 - r
                    elif abs(ang + np.pi / 2) < 1e-6:  # 逆时针 90°
                        new_r, new_c = n - 1 - c, r
                    elif abs(abs(ang) - np.pi) < 1e-6:  # 180°
                        new_r, new_c = n - 1 - r, n - 1 - c
                    else:  # 0°
                        new_r, new_c = r, c

                    return face, new_r, new_c
        return None


@dataclass(frozen=True)
class ActionToken:
    axis: int
    layer: int  # layer=side*mid
    direction: int

    @property
    def key(self) -> tuple:
        return self.axis, self.layer, self.direction

    def invert(self) -> 'ActionToken':
        return ActionToken(axis=self.axis, layer=self.layer, direction=-self.direction)

    @classmethod
    def identity(cls) -> "ActionToken":
        return cls(0, 0, 0)

    @classmethod
    def from_path(cls, path: list[tuple] | tuple) -> list['ActionToken']:
        """从三元组快速创建"""
        if not path:
            return [ActionToken.identity()]
        if isinstance(path, tuple):
            return [cls(*path)]
        return [cls(*t) for t in path]

    @class_cache('BASIC_MOVES', key=lambda n=3: n)
    @classmethod
    def basic_generators(cls, n: int = 3) -> list['ActionToken']:
        """基础生成元,逻辑层（axis, layer, direction）与几何层解耦"""
        center_layers = CubeBase.center_layers_list(n)
        moves = []
        for axis in range(3):
            for layer in center_layers:
                for direction in (-1, 1, 2):  # direction 只用 ±1，2 步可视为两步重复
                    moves.append(cls(axis, layer, direction))
        return moves

    @classmethod
    def from_cubie_move(cls, axis: int, side: int, direction: int, n: int) -> 'ActionToken':
        """
        side == 0 表示中心层转动：
        - 奇数阶：layer = 0（物理中心）
        - 偶数阶：layer = -1（约定表示中心 slice）
        """
        mid, c = divmod(n, 2)
        if side == 0:
            layer = 0 if c == 1 else -1
        else:
            layer = side * mid
        return cls(axis=axis, layer=layer, direction=direction)

    def to_cubie_move(self, n: int = 3) -> tuple | None:
        """
        side
        最外层 → ±1
        中心层 → 0（奇数:0, 偶数:-1）
        其他中间层 → None
        """
        mid, c = divmod(n, 2)
        side = None
        if abs(self.layer) == mid:
            side = 1 if self.layer > 0 else -1
        if self.layer == 0 or (c == 0 and self.layer == -1):
            side = 0
        if side is None:
            return None
        dir_norm = self.direction % 4
        if dir_norm == 3:
            dir_norm = -1
        return self.axis, side, dir_norm

    @classmethod
    def transform(cls, move: str, n: int = 3) -> list['ActionToken']:
        """
        解析层 Human Move String → transform → sticker simulation（仅用于构表）
        通用 NxN 解析，支持标准记法：
            U, U', U2
            R, L, F, B, D
            Rw, Rw', Rw2, Uw, Fw ...
            2Rw, 3Uw',2Rw2,3U,3Uw',2Fw2  等 |A| = 18
        返回实际执行的 primitive move 列表：[(axis, layer, direction), ...]
        """
        if not move:
            raise ValueError("动作不能为空")

        mid, c = divmod(n, 2)
        turn_times = 1
        direction = 1
        import re
        # --- 解析方向 ---
        if move.endswith("2"):
            turn_times = 2
            move = move[:-1]

        if move.endswith("'"):
            direction = -1
            move = move[:-1]

        # --- 正则解析宽度（前缀数字）
        m = re.match(r"(\d*)([URFDLB])(w?)$", move)
        if not m:
            raise ValueError(f"无法解析动作: {move}")

        width_txt, face, wide_flag = m.groups()
        if face not in CubeBase.FACES:
            raise ValueError(f"未知面: {face}")

        # 宽度：无数字 → 默认 1；如果有 'w' 则默认 = 2
        if width_txt:
            width = int(width_txt)
            if width < 1 or width > n:
                raise ValueError(f"宽度 {width} 超出魔方阶数 {n}")
        else:
            width = 2 if wide_flag else 1

        axis, side = CubeBase.face_axis[face]
        if side == 0:  # positive_side 正轴面
            layers = list(range(width))  # 正面方向,U:顶部向下 width 层
        else:
            layers = [n - 1 - i for i in range(width)]  # 反: list(range(N - 1, N - width - 1, -1))

        # ---- primitive moves ----
        prim_moves = []
        for _ in range(turn_times):
            for layer_idx in layers:
                final_dir = direction if side == 0 else -direction  # “正轴面的顺时针” 在 cube 标准中方向不同
                layer = layer_idx - mid
                if c == 0 and layer_idx >= mid:  # 偶数阶需要跳过中心 0
                    layer += 1
                prim_moves.append(cls(axis, layer, final_dir))

        return prim_moves  # 记录步骤

    def embedding(self, n: int = 3) -> np.ndarray:
        """
        动作的几何性质，几何 embedding
        axis:      0,1,2 (X,Y,Z)
        layer:     -mid .. +mid
        direction: -1 (逆90), 2 (180), +1 (顺90)
        n:         魔方阶数

        返回: shape (9,) 或 (7,) 的向量
        """
        dir_norm = self.direction % 4
        if dir_norm == 0:
            raise ValueError(f"Invalid direction:{self.direction}")

        mid = n // 2

        # 1. axis one-hot (3 dim)
        axis_oh = np.zeros(3)
        axis_oh[self.axis] = 1.0

        # 2. depth ∈ [-1, 1], coset 内 vs coset 间作用强度
        depth = self.layer / mid
        # layer_embed = np.array([np.sin(depth * np.pi), np.cos(depth * np.pi)])

        # 3. direction one-hot (3 dim)，更易学习
        dir_idx = dir_norm - 1
        dir_oh = np.zeros(3)
        dir_oh[dir_idx] = 1.0

        # 4. outer-ness: 1=最外层,是否触发 coset 跳变
        is_outer = 1.0 if abs(self.layer) == mid else 0.0
        # distance_to_center = 1.0 - abs(self.layer) / mid

        # 组合
        return np.concatenate([
            axis_oh,  # 3
            [depth],  # 1
            dir_oh,  # 3
            [is_outer]  # 1
        ])  # total 8 dim

    def __add__(self, other) -> list["ActionToken"]:
        """combine_with 列表组合"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, ActionToken):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __neg__(self):
        return self.invert()

    def __hash__(self):
        return hash(self.key)

    def __str__(self) -> str:
        """标准魔方记法，与 transform() 互逆:
        外层: U/U'/U2, R/R'/R2, F/F'/F2, D/D'/D2, L/L'/L2, B/B'/B2
        中心层: M/M'/M2, E/E'/E2, S/S'/S2
        宽层: 2Rw/2Rw'/2Rw2 等
        """
        if self.direction == 0:
            return 'I'  # identity
        dir_norm = self.direction % 4
        pos_face, neg_face = CubeBase.AXIS_FACE[self.axis]
        # 判断 layer 属于哪个面 (基于 n=3 的约定: mid=1, layer=±1/0)
        # 对于一般 n: |layer| == mid → 外层, layer=0 → 中心层, 其他 → 宽层
        n_est = max(3, abs(self.layer) * 2 + 1)  # 从 |layer| 估算 n
        mid = n_est // 2
        is_outer = abs(self.layer) == mid

        if is_outer:
            face = pos_face if self.layer > 0 else neg_face
        elif self.layer == 0:
            face = ['M', 'E', 'S'][self.axis]
        else:  # 宽层: |layer| < mid → 从正面算起第 (mid - |layer|) 层 = width
            width = mid - abs(self.layer) if self.layer > 0 else abs(self.layer) + mid - n_est + 1
            # 简化: 用 layer 偏移表示
            face_char = pos_face if self.layer > 0 else neg_face
            suffix = '2' if dir_norm == 2 else ("'" if dir_norm == 3 else '')
            return f"{abs(self.layer)}{face_char}w{suffix}"

        # direction: +1 → face, -1(≡3) → face', 2 → face2
        if dir_norm == 2:
            return f"{face}2"
        elif dir_norm == 3:
            return f"{face}'"
        else:
            return face

    def __repr__(self) -> str:
        return f"ActionToken({self})"

    @staticmethod
    def invert_moves(moves: list['ActionToken']) -> list['ActionToken']:
        """move 的逆 将 moves 转成可还原的逆操作序列（反向 + 方向反）"""
        return [m.invert() for m in reversed(moves)]

    @staticmethod
    def commutator(A: list, B: list) -> list['ActionToken']:
        """
        交换子,制造局部扰动,奇偶性不变
        A, B: move list
        return: [A, B] = A B A⁻¹ B⁻¹
        """
        return A + B + ActionToken.invert_moves(A) + ActionToken.invert_moves(B)

    @staticmethod
    def conjugate(A: list, B: list) -> list['ActionToken']:
        """
        共轭 A B A⁻¹ 改变作用位置,保持结构不变 or A⁻¹ B A
        """
        return A + B + ActionToken.invert_moves(A)

    @staticmethod
    def cycle3(A: list, B: list, P: list) -> list['ActionToken']:
        """
        experimental
        在 P(A,B) 定义的位置制造一个 3-cycle
        A, B: 产生 3-cycle 的基元
        P: 定位用的 conjugate position_moves
        P · [A, B] · P⁻¹
        """
        base = ActionToken.commutator(A, B)
        return ActionToken.conjugate(P, base)

    @staticmethod
    def at(position_moves: list, base_cycle: list) -> list['ActionToken']:
        """
        在指定位置制造一个贴纸 3-cycle
        position_moves: 把目标贴纸搬到工作区的 moves
        base_cycle: 已知在固定工作区的 cycle
        cycle_at = P · base · P⁻¹
        """
        return ActionToken.conjugate(position_moves, base_cycle)


class StickerCube(CubeBase):
    """贴纸态"""

    def __init__(self, state: np.ndarray | dict = None, n: int = 3):
        super().__init__(n)

        if state is None:  # 初始化已解决状态
            self.cube = self.solved.copy()
            # for face, color in zip(self.FACES, self.COLORS):
            #     self.cube[face] = [[color] * n for _ in range(n)]
        elif isinstance(state, np.ndarray):
            # state 应当是 (6,n,n) 的数值
            self.cube = state.astype(np.uint8)
            self.n = self.cube.shape[1]
        elif isinstance(state, dict):
            self.cube = self.from_color(state)
            self.n = self.cube.shape[1]
            # 假定传入的 state 是面->二维列表的映射，复制一份以免外部修改
            # self.cube = {f: [row.copy() for row in state[f]] for f in self.FACES}
            # self.n = len(self.cube)
        else:
            raise ValueError('wrong state')
        if self.n != n:
            super().__init__(n)

        # total_stickers = 6 * n ^ 2
        # per_face_outer = 4 * n - 4
        # per_face_inner = (n - 2) ^ 2
        # edge_wings = 12 * (n - 2)  # 边翼, 角块（corners）恒为 8

        # self.__class__.axis_face_idx = {axis: (self.FACES.index(pos), self.FACES.index(neg))
        #                                 for axis, (pos, neg) in enumerate(self.AXIS_FACE)}
        assert list(range(6)) == [self.FACES.index(f) for f in self.FACES]
        random.seed(47)

    def clone(self):
        """深拷贝当前魔方并返回新的实例"""
        return StickerCube(state=self.cube.copy(), n=self.n)

    def reset(self):
        self.cube = self.solved.copy()

    def get_state(self) -> np.ndarray:
        """返回当前魔方状态（用于序列化）"""
        return self.cube.copy()

    def is_solved(self, state: np.ndarray = None) -> bool:
        if state is None:
            state = self.cube
        return super().is_solved(state)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.n != other.n:
            return False
        return bool(np.array_equal(self.cube, other.cube))  # np.all(self.cube == other.cube)

    def __hash__(self):
        return hash(super().encode(self.cube))

    @property
    def key(self) -> tuple:
        """把 cube 转成 tuple"""
        return tuple(self.cube.flatten())

    @property
    def faces_colors(self) -> dict:
        return self.get_colors(self.cube)

    @classmethod
    def get_colors(cls, state: np.ndarray) -> dict:
        """返回原来使用的 face->二维字符串颜色矩阵"""
        n = state.shape[1]
        face_stickers = cls.get_face_stickers(n=n)  # 返回 [(r, c, pos), ...] 按渲染顺序
        idx_color = {i: c for i, c in enumerate(cls.COLORS)}
        result = {}
        for fidx, face in enumerate(cls.FACES):
            # col_mat = np.vectorize(idx_color.get)(state[fidx, :, :])
            # result[face] = col_mat.tolist()  # 转为普通列表[[str,str]]
            mat = np.empty((n, n), dtype='<U1')
            coords = face_stickers[face]
            for (r, c, _), color_idx in zip(coords, state[fidx].flatten()):
                mat[r, c] = idx_color[color_idx]
            result[face] = mat.tolist()

        return result

    @classmethod
    def from_color(cls, cube_color: dict) -> np.ndarray:
        # 传入的 state 是面->二维列表的映射
        color_index = {color: i for i, color in enumerate(cls.COLORS)}
        n = len(cube_color)
        arr = np.zeros((6, n, n), dtype=np.uint8)
        for fidx, face in enumerate(cls.FACES):
            face_mat = cube_color[face]
            arr[fidx, :, :] = np.vectorize(color_index.get)(face_mat)
        return arr

    @class_status('参考方法')
    def rotate_face(self, face: str, direction: int = 1):
        """旋转一个面，direction=1顺时针，-1逆时针,axis 与 face.normal 必然平行"""
        fidx = self.face_idx[face]
        axis, side = self.face_axis[face]
        axis_vec = self.AXIS_VEC[axis]
        normal = self.face_normal[face]
        sign = np.dot(normal, axis_vec)
        assert (side == 0 and sign > 0) or (side == 1 and sign < 0)
        layer = self.mid if side == 0 else -self.mid

        d = direction % 4
        dd = d if side == 0 else -d
        self.rotate_inplace(self.cube[fidx], dd)  # np.rot90(arr, -direction)
        self.rotate_slice(self.cube, axis, layer, shift=d, n=self.n)

    @chainable_method
    def rotate(self, axis: int, layer: int, direction: int = 1):
        """
        统一旋转入口,AXIS-SLICE 旋转
        axis: 'x' | 'y' | 'z':0,1,2
        layer: 0 ~ n-1
        direction: 1 = 顺时针, -1 = 逆时针
        """
        # print('rotate:', axis, layer, direction)
        assert 0 <= axis <= 2, f"unknown axis: {axis}"
        assert -self.mid <= layer <= self.mid, f"layer out of range: {layer}"
        self.rotate_core(self.cube, axis, layer, direction)

    def normalize(self) -> list[tuple]:
        # self.cube = self.normalize_sticker(self.cube)
        self.cube, path = self.normalize_align(self.cube)
        return path

    @chainable_method
    def apply(self, moves: list | tuple):
        self.act_moves(self.cube, moves)

    def apply_move(self, move: str):
        primitive_moves = [t.key for t in ActionToken.transform(move)]
        self.apply(primitive_moves)
        return primitive_moves


if __name__ == "__main__":
    # 测试已迁移到 test/test_cube.py
    pass
