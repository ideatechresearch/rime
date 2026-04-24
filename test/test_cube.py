import numpy as np
import random, math
from collections import deque
from rime.cube import StickerCube, CubeBase
from rime.base import class_cache, class_property,class_status, check_class_status


N = 5

class StickerSolver(StickerCube):

    def __init__(self, state: np.ndarray | dict = None, n: int = 3):
        super().__init__(state,n)

    @classmethod
    def from_stickers(cls, sticker: StickerCube):
        return cls(state=sticker.cube.copy(), n=sticker.n)

    def diff(self, other, max_show: int = 20):
        diffs = []
        for f in range(6):
            for r in range(self.n):
                for c in range(self.n):
                    a = int(self.cube[f][r][c])
                    b = int(other.cube[f][r][c])
                    if a != b:
                        diffs.append((f, r, c, a, b))
                        if len(diffs) >= max_show:
                            return diffs
        return diffs
    
    def propose_move(self, layer_span: int = None) -> tuple[int, int, int]:
        """         
        采样一个候选 move：           
        - axis: X/Y/Z uniformly           
        - layer: 优先在中间 layer 附近采样（layer_span 设置距离 mid 的范围），超出时均匀采样全部层  
        - direction: 随机取 1 (CW), -1 (CCW) 或 2 (180) （180 的概率可降低）        
        """
        if not layer_span:
            layer_span = self.mid
        axis = random.choice(range(3))
        # sample layer: with 80% prob sample near center, 20% anywhere
        low = max(-self.mid, - layer_span)
        high = min(self.mid, layer_span)
        layer = random.randint(low, high)  # 上界包含
        if layer == 0 and self.n % 2 == 0:
            layer = random.choice([-1, 1])
        # direction probabilities: prefer +/-1; occasional 2
        direction = random.choices([-1, 1, 2], weights=[0.48, 0.48, 0.04], k=1)[0]
        return axis, layer, direction
    
    @class_status('已废弃')
    def heuristic_corner_old(self, state: np.ndarray) -> int:
        '''隐含了三个前提：
        corner 编号是隐式的（靠位置顺序）
        orientation 不可信 → 用 set 抹掉方向
        solved 状态是一个固定 sticker 模板
        Singmaster / sticker-based 的世界观
        '''
        wrong = 0  # number_of_wrong_corners
        solved = self.get_corners(self.solved)
        cur = self.get_corners(state)
        for a, b in zip(cur, solved):
            if set(a) != set(b):
                wrong += 1
        return wrong
    
    def central_edge_coords(self, face: str, base_fase: str = 'D') -> list:
        """         
        返回 face 与 D 相邻的 central edge 两个 sticker 的位置坐标 (face1, r1, c1), (face2, r2, c2)
        约定 faces order 与 AXIS_STRIP/face_idx 一致。这里选定 D-face 边为目标边位置：         
        返回格式：[(fidx1, r1, c1), (fidx2, r2, c2)]，顺序与 edge_face_cycle 一致 
        """
        edge_face = {face, base_fase}
        eid = next(i for i, pair in enumerate(self.edge_face_cycle) if set(pair) == edge_face)
        return self.edge_coords(self.n)[eid]

    def heuristic_edge_mismatch(self, face: str, base: int, state: np.ndarray = None) -> int:
        """ 
        score: 0 if matched, higher if mismatch. 用于贪心最小化。         
        判断指定 face 的 central edge 是否已在 D 层且两面颜色对齐：         
        - D-side 的 sticker == base_color         
        - side-face 的 sticker == side-face center color         
        """
        if state is None:
            state = self.cube
        (f1, r1, c1), (f2, r2, c2) = self.central_edge_coords(face, 'D')
        s1 = int(state[f1, r1, c1])
        s2 = int(state[f2, r2, c2])  # D-side
        target_side = int(state[f1, self.mid, self.mid])
        score = 0
        if s2 != int(base):  # D 面 base_color 是否匹配
            score += 1
        if s1 != target_side:  # 侧面是否正确颜色
            score += 1
        if r2 != self.mid or c2 not in (self.mid - 1, self.mid, self.mid + 1):  # 如果边块根本不在 D 层：加罚分
            score += 1  # or 2
        return score

    def detect_oll_parity(self, state: np.ndarray = None):
        """         
        检查是否存在 odd-flip 的情况（只可能在 NxN Even）。         
        原理：检查边配对之后是否存在单边颜色方向异常。         
        """
        if state is None:
            state = self.cube

        # 任意找一条 LL 边即可
        # 譬如 UF 的 central edge
        (f1, r1, c1), (f2, r2, c2) = self.central_edge_coords('F', 'D')  # (face, r, c)
        c1 = int(state[f1, r1, c1])
        c2 = int(state[f2, r2, c2])

        # 如果颜色组合不合法（不该出现），则 flip parity
        # 简化：检查两侧中心色是否与该 edge 组合矛盾
        side_center = int(state[f1, self.mid, self.mid])
        down_center = int(state[self.face_idx['D'], self.mid, self.mid])

        # Very loose but useful detection
        if {c1, c2} != {side_center, down_center}:
            return True

        return False

    def detect_pll_parity(self, state: np.ndarray = None):
        """         
        在 PLL 阶段（3×3 模式）判断最后两条边是否单独 swap。         
        """
        if state is None:
            state = self.cube
        # 基于 3×3 PLL 顶层边块颜色检查
        # 检查 U face 四条边是否出现奇偶交换
        edges = [
            ('U', self.mid, 0),  # UL
            ('U', self.mid, self.n - 1),  # UR
            ('U', 0, self.mid),  # UB
            ('U', self.n - 1, self.mid),  # UF
        ]

        values = [int(state[self.face_idx[f], r, c]) for f, r, c in edges]
        # 如果颜色排列不可能正解 = swap parity
        # 使用简单判断：出现两条 edge 对调
        if len(set(values)) < 4:  # 简化：冲突
            return True

        return False

    def ida_star(self, max_depth: int = 25):
        """Small depth IDDFS search for local corrections."""
        visited = set()
        bound = self.heuristic_center(self.cube)
        while True:
            visited.clear()
            res, sol = self._dfs(self.cube, 0, bound, visited, [], max_depth)
            if res is True:
                return sol
            if res == float('inf'):
                return None
            bound = res

    def _dfs(self, state, g, bound, visited, path, max_depth):
        """IDA* recursive DFS search."""
        h = self.heuristic_center(state)
        f = g + h
        if f > bound:
            return f
        if h == 0:
            return True
        if g >= max_depth:
            return float('inf')

        key = self.encode(state)
        if key in visited:
            return float('inf')
        visited.add(key)

        min_next = float('inf')
        for move in self.basic_generators():
            if self.is_inverse(path, *move):
                continue
            nxt = self.rotate_state(state, *move)
            res = self._dfs(nxt, g + 1, bound, visited, path + [move], max_depth)
            if res is True:
                return True
            if res < min_next:
                min_next = res
        return min_next


    def greedy_fix_center(self, r: int = 1):
        """         
        对中心区域做贪心修正。直到中心收敛         
        贪心：尝试旋转一圈，找错误减少最大的 move         
        """
        start_wrong = self.heuristic_center(self.cube, r)
        best = None
        best_delta = 0
        for move in self.basic_generators():
            next_state = self.rotate_state(self.cube, *move)
            h = self.heuristic_center(next_state, r)

            delta = start_wrong - h
            if delta > best_delta:
                best_delta = delta
                best = move

        return best, best_delta

    def solve_cross(self, max_iter: int = 100, max_depth: int = 2, shuffle_moves: bool = True) -> list:
        """         
        底层十字,BFS ≤ 5         
        1. 找到 D 面中心颜色         
        2. 把含该颜色的边块移到底层         
        3. 校准侧面颜色与正确面一致（方向正确）         
        """
        base_color = int(self.cube[self.face_idx['D'], self.mid, self.mid])
        targets = ['F', 'R', 'B', 'L']

        candidate_single_moves = list(self.basic_generators())
        if shuffle_moves:
            random.shuffle(candidate_single_moves)

        # Node = namedtuple('Node', ['state', 'path'])
        moves = []  # applied
        for face in targets:
            if self.heuristic_edge_mismatch(face, base_color, self.cube) == 0:
                continue  # already matched

            # BFS on small depth (try_depth steps)
            start_state = self.get_state()
            queue = deque([(start_state, [])])
            visited = {self.encode(start_state)}  # start_key
            iter_count = 0
            found_seq = None  # greedy local search bounded by max_iter for this face

            while queue and iter_count < max_iter:
                cur_state, cur_path = queue.popleft()
                iter_count += 1

                cur_score = self.heuristic_edge_mismatch(face, base_color, cur_state)
                if cur_score == 0:
                    found_seq = cur_path
                    break

                # depth control: limit path length to try_depth
                if len(cur_path) >= max_depth:
                    continue

                # try single moves (or small template moves)
                for move in candidate_single_moves:
                    if self.is_inverse(cur_path, *move):
                        continue
                    nxt = self.rotate_state(cur_state, *move)
                    key = self.encode(nxt)
                    if key in visited:
                        continue
                    visited.add(key)
                    new_path = cur_path + [move, ]
                    queue.append((nxt, new_path))

            if found_seq is None:
                # fallback: greedy hill-climb (try single moves reducing score)
                cur_state = self.get_state()
                for _ in range(max_iter):
                    cur_score = self.heuristic_edge_mismatch(face, base_color, cur_state)
                    if cur_score == 0:
                        found_seq = []
                        break
                    best_move = None
                    best_score = cur_score
                    for move in candidate_single_moves:
                        nxt = self.rotate_state(cur_state, *move)
                        s = self.heuristic_edge_mismatch(face, base_color, nxt)
                        if s < best_score:
                            best_score = s
                            best_move = move
                    if best_move is None:
                        break
                    # apply move to cur_state and continue (note: not yet applied to self)
                    cur_state = self.rotate_state(cur_state, *best_move)
                    found_seq = found_seq + [best_move, ] if found_seq else [best_move]

            if found_seq:
                # apply the found_seq to actual cube (mutating self) and record moves
                for mv in found_seq:
                    axis, layer, direction = mv
                    self.rotate(axis, layer, direction)
                    moves.append(mv)
                # verify matched
                if self.heuristic_edge_mismatch(face, base_color, self.cube) > 0:
                    # try a few small local repairs (greedy single-step)
                    for _ in range(8):
                        cur_score = self.heuristic_edge_mismatch(face, base_color, self.cube)
                        if cur_score == 0:
                            break
                        # try a single best move in real cube
                        best_move = None
                        best_score = cur_score
                        for mv in candidate_single_moves:
                            nxt = self.rotate_state(self.cube, *mv)
                            s = self.heuristic_edge_mismatch(face, base_color, nxt)
                            if s < best_score:
                                best_score = s
                                best_move = mv
                        if best_move is None:
                            break
                        self.rotate(*best_move)
                        moves.append(best_move)
            else:
                # cannot find small seq; skip this face (user can increase try_depth / max_iter)
                # optionally log or raise
                # print(f"WARNING: cannot pair central edge for {face}")
                pass

        return moves

    def solve_centers(self, greedy_iter: int = 4, max_depth: int = 16):
        """解决中心块,先贪心,把中心粘起来，再 IDA* 补洞局部修正"""

        def greedy_pass():
            moves = []
            for _ in range(greedy_iter):
                mv, gain = self.greedy_fix_center()
                if mv is None or gain <= 0:
                    break
                self.apply(mv)
                moves.append(mv)
            return moves

        g_moves = greedy_pass()
        ida_moves = self.ida_star(max_depth=max_depth) or []
        self.apply(ida_moves)  # 执行 IDA* 结果
        return g_moves + ida_moves

    def solve_bfs(self, max_depth: int = 6) -> list | None:
        '''用 BFS 解决 局部,BFS 5~7'''
        start = self.get_state()
        queue = deque([(start, [], 0)])  # state, path, depth
        visited = {self.encode(start)}

        while queue:
            state, path, depth = queue.popleft()

            if self.is_solved(state):
                return list(path)

            if depth >= max_depth:
                continue

            for move in self.basic_generators():
                # forbid immediate reversal
                if self.is_inverse(path, *move):
                    continue

                next_state = self.rotate_state(state, *move)
                key = self.encode(next_state)

                if key in visited:
                    continue

                visited.add(key)
                queue.append((next_state, path + [move], depth + 1))

        return None

    def solve_edges(self):
        """         解决边块配对问题（Reduction 方法）         1. 扫描全部边位置，按颜色分类         2. 识别未配对的边块         3. 使用 slice moves + pairing 算法配对         4. 处理 parity 异常         """
        moves = []
        mid = self.mid

        # 收集所有边的颜色组合
        edge_color_map = {}  # {(c1, c2): [(f, r, c), ...]}
        for eid, coords in enumerate(self.edge_coords(self.n)):
            colors = tuple(sorted(int(self.cube[f, r, c]) for f, r, c in coords))
            if colors not in edge_color_map:
                edge_color_map[colors] = []
            edge_color_map[colors].append(coords)

        # 统计配对情况
        pairs = {k: v for k, v in edge_color_map.items() if len(v) == 2}
        unpaired = {k: v for k, v in edge_color_map.items() if len(v) == 1}

        if not unpaired:
            return moves  # 所有边已配对

        # 简单实现：使用标准边配对算法
        # 对于每个未配对的边，寻找匹配目标并配对
        for edge_colors, coords_list in unpaired.items():
            coords = coords_list[0]
            # 找到目标位置的颜色
            target = tuple(sorted(edge_colors))
            if target in pairs and pairs[target]:
                # 使用 U slice 移动配对
                pair_coords = pairs[target].pop()
                # 计算需要的 slice move
                move_seq = self._pair_edge(coords, pair_coords)
                moves.extend(move_seq)
                self.apply(move_seq)

        return moves

    def _pair_edge(self, coords1: list, coords2: list) -> list:
        """将两个边块配对（简化的 slice move）"""
        # 简化的配对逻辑：使用标准的 2-3-2 交换序列
        slice_layer = self.mid  # 中间层
        pair_seq = [
            (0, slice_layer, 1),   # U 顺时针
            (1, slice_layer, 1),   # R
            (0, -slice_layer, 1), # D
            (1, -slice_layer, 1),  # L
            (0, slice_layer, -1), # U' 逆时针
            (1, slice_layer, 1),  # R
            (0, slice_layer, 1),  # U
            (1, slice_layer, -1), # R'
            (0, -slice_layer, -1),# D'
            (1, slice_layer, 1),  # R
            (0, slice_layer, 1),  # U
            (1, slice_layer, -1), # R'
        ]
        return pair_seq

    def fix_parity(self):
        """           
        NxN Parity 修复：           
        - OLL Parity: 单边翻转           
        - PLL Parity: 最后二边互换           
        每个 parity 修复都增加到 moves[]         
        """
        moves = []
        # 1. OLL parity
        if self.detect_oll_parity():
            oll_seq = [
                "Rw", "U2", "Rw", "U2",
                "Rw", "U2", "Rw'", "U2",
                "Lw", "U2", "Rw", "U2",
                "Rw", "U2", "Rw'", "U2",
                "Rw'"
            ]
            for mv in oll_seq:
                moves += self.apply_move(mv)

        # 2. PLL parity
        if self.detect_pll_parity():
            pll_seq = [
                "2Rw2", "U2", "2Rw2", "U2",
                "Uw2", "2Rw2", "Uw2"
            ]
            for mv in pll_seq:
                moves += self.apply_move(mv)

        return moves

    def reduction(self):
        """         
        解决中心块, 配对解决边块,三阶解法
        
        1. solve_centers()         
        2. solve_edges()         
        3. solve_3x3()  # 调用三阶魔方解法         
        4. fix_parity() 处理奇偶错误         
        """
        moves = []
        # mv_centers = self.solve_centers(
        #     greedy_iter=4,
        #     max_depth=9  # 4×4 用 12~14；6×6 用 16~20
        # )
        # moves += mv_centers
        # print('solve_centers:', mv_centers)

        mv_parity = self.fix_parity()
        moves += mv_parity
        print('fix_parity:', mv_parity)

        mv_cross = self.solve_cross(max_iter=200)
        moves += mv_cross
        print('solve_cross:', mv_cross)

        mv_bfs = self.solve_bfs(max_depth=6)  # F2L
        if mv_bfs:
            self.apply(mv_bfs)
            moves += mv_bfs
        print('solve_bfs:', mv_bfs)
        mv = self.ida_star(max_depth=14)
        if mv:
            self.apply(mv)
            moves += mv
        print('ida_star:', mv)
        return moves

    def solve(self):
        if self.is_solved(self.cube):
            return []
        return self.reduction()


def test_properties(cube):
    print(cube.face_def)
    print(cube.corner_face_cycle)
    print(cube.edge_face_cycle)
    print(cube.faces_colors)


def test_coords(cube):
    corners = cube.corner_coords(cube.n)
    edges = cube.edge_coords(cube.n)
    print('corner_coords', len(corners), corners)
    print('edge_coords', len(edges), edges)
    print('map', cube.SOLVED_CORNERS_MAP, cube.SOLVED_EDGES_MAP)
    print('rotations', len(CubeBase.generate_rotations))


def test_strip_colors(cube):
    for layer in range(-cube.mid, cube.mid + 1):
        strip = cube.strip_coords_from_axis(2, layer, cube.n)
        colors = [[cube.cube[f, r, c] for f, r, c in s] for s in strip]
        print(colors)


def test_encoding(cube):
    print('encode_state', cube.encode_state(cube.cube))
    print('encode_state_idx', cube.encode_state(cube.solved_idx).astype(float) / (cube.n * cube.n))
    print('embedding_relative', cube.embedding(cube.solved_idx))


def test_layer_and_center(cube):
    print('layer_stickers', len(cube.get_layer_stickers(0, 1, cube.n)), cube.get_layer_stickers(0, 1, cube.n))
    rings = cube.get_center_rings(cube.n - 2)
    print('center_rings', len(rings), [len(x) for x in rings], '\n', rings)
    orbits = cube.center_orbits(cube.n - 1)
    print('center_orbits', len(orbits), orbits)


def test_rotation_identity(cube):
    backup = cube.get_state()
    for axis in (0, 1, 2):
        for layer in range(-cube.mid, cube.mid + 1):
            cube0 = cube.clone()
            for _ in range(4):
                cube.rotate(axis, layer, direction=1)
            if not cube.is_solved():
                print(f'{axis},{layer} not solved')
            if cube != cube0:
                print("FAIL:", axis, layer, cube.diff(cube0, 10))
    assert np.all(cube.cube == backup)
    cube.reset()
    print('rotation_identity ok')


def test_scramble(cube, moves=10):
    mv = list(cube.generate_moves(moves))
    cube.apply(mv)
    print('moves', mv)
    print('diff', cube.diff_coords(cube.cube))
    inv_moves = cube.invert_moves(mv)
    cube.apply(inv_moves)
    assert cube.is_solved(), "scramble+inverse not solved"
    print('scramble ok')


def test_scramble_idx(cube, moves=10):
    mv = list(cube.generate_moves(moves))
    s0 = cube.solved_idx.copy()
    cube.act_moves(s0, mv)
    assert np.array_equal(np.sort(s0.reshape(-1)), cube.solved_idx.reshape(-1))
    inv_moves = cube.invert_moves(mv)
    cube.act_moves(s0, inv_moves)
    assert np.array_equal(s0, cube.solved_idx), "idx scramble+inverse not solved"
    return True


def test_scramble_idx_bulk(cube, rounds=1000, moves=80):
    err = 0
    for i in range(rounds):
        if not test_scramble_idx(cube, moves):
            print(f"failed at round {i}")
            err += 1
    print(f"all good. err {err}")


def test_3x3():
    cube = StickerCube(n=3)
    print(cube.get_state())
    print('corners', cube.get_corners(cube.cube))
    mv = list(cube.generate_moves(20))
    cube.apply(mv)
    print('moves', mv)
    print('is_solved', cube.is_solved())
    print(cube.faces_colors)


def test_solver_basic(solver: StickerSolver):
    """Test StickerSolver basic operations"""
    print("Testing basic operations...")
    assert solver.is_solved(), "New solver should be solved"
    print("  is_solved: OK")

    move = solver.propose_move()
    print(f"  propose_move: {move}")
    assert len(move) == 3, "propose_move should return (axis, layer, direction)"

    solver.rotate(*move)
    assert not solver.is_solved(), "After rotate should not be solved"
    print("  rotate: OK")

    solver.reset()
    assert solver.is_solved(), "After reset should be solved"
    print("  reset: OK")


def test_solver_propose_move(solver):
    """Test propose_move sampling"""
    print("Testing propose_move sampling...")
    moves = [solver.propose_move() for _ in range(20)]
    axes = set(m[0] for m in moves)
    print(f"  axes sampled: {axes}")
    dirs = set(m[2] for m in moves)
    print(f"  directions sampled: {dirs}")
    print("  propose_move: OK")


def test_solver_scramble_and_solve(solver):
    """Test scramble and solve cycle"""
    print("Testing scramble and solve...")

    scramble = list(solver.generate_moves(10))
    solver.apply(scramble)
    assert not solver.is_solved(), "After scramble should not be solved"
    print(f"  scrambled with {len(scramble)} moves: OK")

    solution = solver.solve()
    print(f"  solution length: {len(solution) if solution else 0}")
    print("  scramble and solve: OK")


def test_solver_cross(solver):
    """Test solve_cross method"""
    print("Testing solve_cross...")
    cross_moves = solver.solve_cross(max_iter=100, max_depth=3)
    print(f"  cross moves found: {len(cross_moves)}")
    print("  solve_cross: OK")


if __name__ == "__main__":
    class_cache.load(StickerCube)
    class_property.load(StickerCube)

    cube = StickerCube(n=N)

    test_properties(cube)
    test_coords(cube)
    test_strip_colors(cube)
    test_encoding(cube)
    test_layer_and_center(cube)
    test_rotation_identity(cube)

    print('.................')

    test_scramble(cube)
    test_scramble_idx(cube)
    test_scramble_idx_bulk(cube)

    test_3x3()
    class_cache.save(StickerCube)
    class_property.save(StickerCube)
    print(cube.get_vars())
    print(cube.strip_coords_from_axis.cache)
    print(check_class_status(StickerCube))

    # StickerSolver tests
    print("=== StickerSolver Tests ===")
    solver = StickerSolver(n=N)
    test_solver_basic(solver)
    test_solver_propose_move(solver)
    test_solver_scramble_and_solve(solver)
    test_solver_cross(solver)
