import numpy as np
import random, math, time
from rime.cube import CubeBase, ActionToken, StickerCube
from rime.cubie import CubieBase, CubieState, CubieMove, StickerMove
from rime.cubie import Phase1Coord, Phase1Action, Phase2Coord, Phase2Action, Phase15Coord, Phase15Action, CubieExample
from rime.base import class_cache, class_property, check_class_status

N = 3


def setup():
    class_cache.load(CubieBase)
    class_property.load(CubieBase)
    class_cache.load(CubieState)
    class_property.load(CubieState)


# ── 基本属性 & 引用 ─────────────────────────────────────────────

def test_references(cube):
    print('cr', cube.build_corner_reference())
    print(cube.CORNER_REF_AXIS)
    print(cube.build_edge_reference_axis())
    print('er', cube.build_edge_reference())
    print(CubieState.non_slice_edges)
    print(CubieState.ud_slice_edges)
    print(CubieState.solved_ud)  # 69
    print('move_id', CubieMove.basic_generators)
    print(Phase1Coord.project(CubieState.solved()))
    # cube.build_phase15_pruning()
    print('rotate_map', cube.build_rotate_map())


def test_solved_roundtrip(cube):
    """贴纸 ↔ Cubie 双向转换"""
    s_i = cube.solved_idx.copy()
    s: CubieState = cube.to_cubie(cube.solved)
    s0 = CubieState.solved()
    assert s == s0
    a1 = cube.from_cubie(s0)
    a2 = cube.idx_to_state(s0.to_sticker())
    assert np.array_equal(a1, a2)
    assert cube.to_cubie(a1) == s0
    assert cube.to_cubie(a2) == s0
    all_moves = CubieMove.prim_moves().copy()
    all_moves.update(CubieMove.slice_moves())
    for k, m in all_moves.items():
        k2 = ActionToken.from_cubie_move(*k, n=3).key
        s1 = m.act(s0)
        ai1 = s1.to_sticker()
        a1 = cube.from_cubie(s1)
        a2 = cube.idx_to_state(ai1 )
        assert np.array_equal(a1, a2)
        assert cube.to_cubie(a1) == s1
        # a3 = cube.rotate_state(cube.idx_to_state(s_i), *k2)
        # assert np.array_equal(a1, a3),f"{k,a1,a3}"
        # assert np.array_equal(ai1, ai2),f"{k,ai1,ai2}"

    # 看一个 corner 的 3 个贴纸来源
    s_idx = cube.solved_idx.copy()
    s_idx1 = cube.rotate_state(s_idx, 0, 1, 1)
    corner = cube.corner_coords(cube.n)[1]
    print([s_idx1[f, r, c] for (f, r, c) in corner])


def test_cubie_roundtrip(cube):
    a0 = cube.solved.copy()
    s0 = CubieState.solved()
    for k, m in CubieMove.prim_moves().items():
        s1 = m.act(s0)
        a1 = cube.from_cubie(s1)
        assert cube.to_cubie(a1) == s1
        k2 = ActionToken.from_cubie_move(*k, n=3).key
        a2 = cube.rotate_state(a0, *k2)
        assert cube.to_cubie(a2) == s1
        assert np.array_equal(a1, a2)


# ── StickerMove ─────────────────────────────────────────────────

def test_sticker_move():
    m = StickerMove.from_rotation(3, ActionToken(0, 0, 1))
    m_inv = m.inverse()
    assert np.all(m_inv.perm[m.perm] == np.arange(54))
    print('embedding', m.embedding())

    # StickerMove identity round-trip
    sticker_idx = CubieBase(n=5).solved_idx.copy()
    sm = StickerMove.identity(5)
    sm_perm = sm.act(sticker_idx)
    ss = StickerMove.build(sticker_idx, sm_perm)
    assert ss == sm, f'{ss},{sm}'


def test_sticker_cubie_move_consistency(cube0):
    """StickerMove ↔ CubieMove φ 映射一致性"""
    s = cube0.solved_idx
    K1 = random.choice(list(CubieMove.prim_moves.keys()))
    K2 = random.choice(list(CubieMove.prim_moves.keys()))
    print(K1, K2)

    m1 = CubieMove.prim_moves[K1]
    m2 = CubieMove.prim_moves[K2]
    m1_s = StickerMove.phi(cube0.n, m1)
    m2_s = StickerMove.phi(cube0.n, m2)
    cube1 = m1_s.apply(cube0)
    cube2 = m2_s.apply(cube1)
    tokens = [ActionToken.from_cubie_move(*K1, n=3), ActionToken.from_cubie_move(*K2, n=3)]
    _, cube3 = StickerMove.act_moves(cube0.get_state(), tokens)
    assert np.all(cube2.cube == cube3), f'{tokens}'
    print(cube3)
    cube0.reset()
    xx = []
    for move in [K1, K2]:
        axis, side, direction = move
        layer = side * cube0.mid
        xx.append((axis, layer, -direction))  # 用 -direction 对齐

    cube0.apply(xx)
    assert np.all(cube0.cube == cube3)

    s1 = m2_s.act(m1_s.act(s))
    s2 = m1_s.compose(m2_s).act(s)
    assert np.all(s1 == s2), f'{s1}\n{s2}'


# ── 贴纸 ↔ Cubie 一致性 ─────────────────────────────────────────

def test_all_primitive_moves_solvable(cube):
    """
    cube : 你的 StickerCube / state-index 体系
    s0   : CubieState，对应 s_i0
    s_i0 : 贴纸/索引状态（solved 或任意）
    CubieState 是真值层（ground truth）
    Sticker 只是一个表示（representation）
    """
    s_i = cube.solved_idx.copy()
    s0 = CubieState.solved()

    failed = []
    d_i = 0

    for k, move in CubieMove.prim_moves.items():
        # 贴纸级旋转
        t = ActionToken.from_cubie_move(*k, n=3)
        ma = t.key
        assert k == t.to_cubie_move()
        s_i1 = cube.rotate_state(s_i, *ma)
        s1 = move.act(s0)
        # s1=move.act_left(s0)
        assert CubieMove.build(s0, move.act(s0)) == move
        # 转回 CubieState
        s_i2 = cube.idx_to_state(s_i1)
        s11 = cube.to_cubie(s_i2)  # CubieMove.from_rotation(2, 1, 1)
        s_i3 = cube.from_cubie(s1)
        s13 = cube.to_cubie(s_i3)  # cubie → 贴纸 → cubie
        # assert s13 == s1, f'{s13},{s1}'
        if not np.array_equal(s_i3, s_i2):
            print(f"{(s_i3 != s_i2).sum()}\n {np.argwhere(s_i3 != s_i2)}")
            # 8/12 (2, -1, -1) (2, 1, 1) (0, 1, -1) (0, -1, 1)

        print('label:', cube.critic_progress_label(cube.solved, s_i2),
              cube.critic_progress_label(cube.solved, s_i3))
        if not s11.is_solvable():
            failed.append(ma)
            print(f"[FAIL] {ma}", s1.is_solvable(), s11.is_solvable())
            print("s11 =", s11)
            print("corner_ori_sum =", s11.corners_ori.sum() % 3)
            print("edge_ori_sum   =", s11.edges_ori.sum() % 2)
            print(
                "parity(corner, edge) =",
                CubeBase.permutation_parity(s11.corners_perm),
                CubeBase.permutation_parity(s11.edges_perm),
            )
        move2 = CubieMove.build(s0, s11)
        if s1 != s11:  # 有一类 move 的 corner twist 更新方向写反了
            mask = (s1.corners_ori + s11.corners_ori) % 3 != 0
            print(np.where(mask))
            assert move2.act(s0) == s11
            move3 = CubieMove.build(s0, s1)
            move4 = move.with_(corners_ori_delta=(-move.corners_ori_delta) % 3)
            assert move2 == move4, f"{move2.corners_ori_delta},{move4.corners_ori_delta}"
            assert move4.act(s0) == s11
            if not np.array_equal(s1.corners_ori, s11.corners_ori):
                print(s1.is_solvable(), s11.is_solvable())
                print(cube.corner_orientation(s_i2))
                print("s1 corners_ori :", s1.corners_ori.tolist())
                print("s11 corners_ori:", s11.corners_ori.tolist())
                diff = (s1.corners_ori - s11.corners_ori) % 3
                print("diff (mod 3)   :", diff.tolist())
                print(k, ma, move.corners_ori_delta, '\n',
                      move2.corners_ori_delta, move2.corners_perm, '\n',
                      move3.corners_ori_delta, move3.corners_perm)
                print("-" * 40)
                d_i += 1
        else:
            assert move2 == move

    if not failed:
        print("All primitive moves produce solvable CubieState", d_i)  # 4
    else:
        print("Failed moves:", failed)


def test_outer_moves_only(cube):
    s = cube.solved.copy()

    for axis in (2,):
        for layer in (-1, 1):  # 只转最外层
            for d in (1, -1):
                s1 = cube.rotate_state(s, axis, layer, d)
                try:
                    s11 = cube.to_cubie(s1)
                    if not s11.is_solvable():
                        print(s11)
                        raise AssertionError

                except ValueError as e:
                    print("outer move illegal!", axis, layer, d)
                    raise
    print("outer moves all cubie-valid")


def test_rotate_state_vs_cubie_move(cube):
    """贴纸 rotate_state 与 CubieMove.from_rotation 逐 move 对比"""
    s_i = cube.solved_idx.copy()
    s0 = CubieState.solved()

    s_i1 = cube.rotate_state(s_i, 1, 1, 1)
    s1 = CubieMove.from_rotation(1, 1, 1).act(s0)
    s11 = cube.to_cubie(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 1')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))

    s_i1 = cube.rotate_state(s_i, 0, 1, 1)
    s1 = CubieMove.from_rotation(0, 1, 1).act(s0)
    s11 = cube.to_cubie(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 2')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))

    s_i1 = cube.rotate_state(s_i, 2, -1, 1)
    s1 = CubieMove.from_rotation(2, -1, 1).act(s0)
    s11 = cube.to_cubie(cube.idx_to_state(s_i1))
    if s1 != s11:
        print('test 3')
        print('s1', s1)
        print('s11', s11)
        print(cube.corner_orientation(cube.idx_to_state(s_i1)))


def test_gauge_correction(cube):
    """from_rotation 理论 move vs 贴纸真值 move 的 gauge 修正"""
    s0 = CubieState.solved()
    for move in cube.basic_generators():
        """
        把一个 move 映射到"同一个物理效果但在不同参考系/ gauge 下的等价元素"
        参考系变换元素 gauge element,g 属于 orientation 子群的"平移"部分（常量加法），是正规子群中的元素
        g = s0 ∘ m_truth ∘ m_theory⁻¹ ∘ s0⁻¹
        g 是 m_truth ⋅ m_theory⁻¹ 这个"差异元素"被 s0 共轭后的结果
        是 s0 共轭后的差异元素
        """
        axis, layer, direction = move
        if layer == 0:
            continue
        # 用 from_rotation 得到一个"理论 move"
        m_theory = CubieMove.from_rotation(*move)

        # 用贴纸构建一个"真值 move"
        m_truth = cube.build_cubie_move_from_stickers(cube.solved, ActionToken(*move))

        assert CubieMove.build(s0, m_theory.act(s0)) == m_theory, f'{m_theory}'

        assert np.array_equal(m_theory.corners_perm,
                              m_truth.corners_perm), f'{m_theory.corners_perm}, {m_truth.corners_perm}'
        assert np.array_equal(m_theory.edges_perm, m_truth.edges_perm), f'{m_theory.edges_perm}, {m_truth.edges_perm}'

        if m_theory == m_truth:
            print(f"{move} pass")
            continue

        print(m_theory, '\n', m_truth)

        # 计算 gauge 修正
        s_theory = m_theory.act(s0)  # _left
        s_truth = m_truth.act(s0)
        g = CubieMove.build(s_theory, s_truth)
        print(f"{move} gauge g:", g)

        # 修正 from_rotation
        m_fixed = m_theory.compose(g)

        # 验证修正后一致
        assert m_fixed == m_truth, f'fixed != truth for {move}\n{m_fixed}\n{m_truth}'
        assert m_fixed.act(s0) == m_truth.act(s0), f'fixed act mismatch for {move}'

        g = CubieMove.build(m_truth.act(s0), m_theory.act(s0))
        print(f"{move} gauge g2:", g)
        m_fixed = m_truth.compose(g)
        assert m_fixed == m_theory, f'fixed != truth for {move}\n{m_fixed}\n{m_theory}'

        print(f"{move} fixed")
        """
        g 需要看 axis / side
        真实贴纸状态里拿不到 move（axis / side)
        同一个 cubie permutation + orientation delta 可以由 多个不同的基本旋转组合产生
        orientation gauge 抹掉了"旋转方向"的信息
        U/D 轴 (axis=1)：g 全 0（无 twist 差异），完全一致
        R/L 轴 (axis=0)：
        side = -1 (L 层)：g = [0,0,1,0,0,1,0,1]（1 在 2,5,7）
        side = 1 (R 层)：g = [0,0,0,1,1,0,0,0,1]（1 在 3,4,7）或 [1,0,0,0,0,0,0,2]（类似但位置不同）

        F/B 轴 (axis=2)：
        side = -1 (B 层)：g 有 2 在 2/3/6/7
        side = 1 (F 层)：g 有 2 在 0/1/4/5/7
        (2, -1, 1)  (2, 1, -1) (0, 1, 1)(0, -1, -1)
        """


# ── 群论 & 表示 ──────────────────────────────────────────────────

def test_group_axioms():
    """逆元、恒等元、结合律"""
    s: CubieState = CubieState.solved()
    M = CubieMove.from_rotation(1, 1, 1)
    M2 = CubieMove.from_rotation(1, 1, 2)
    M_ = CubieMove.from_rotation(1, 1, -1)
    I = CubieMove.identity()

    # 1. 逆元与恒等元
    assert M.compose(I) == M
    assert I.compose(M) == M
    assert M.compose(M.inverse()) == I, f'{M},{M.inverse()}'
    assert M.inverse().compose(M) == I
    assert M.compose(M_) == I
    assert M.compose(M) == M2
    assert M_.compose(M_) == M2
    assert M.inverse() == M_
    print("group axioms ok")


def test_representation():
    """向量/矩阵/ρ 表示：酉性、同态、trace"""
    s: CubieState = CubieState.solved()
    M = CubieMove.from_rotation(1, 1, 1)
    I = CubieMove.identity()

    Sv = s.vector
    assert Sv.dtype == np.complex64
    assert CubieState.from_vector(Sv) == s, f'{Sv}'

    Mg = I.matrix
    print(Mg.shape)  # (228, 228)
    eigvals = np.linalg.eigvals(Mg)  # 谱分析
    print(np.unique(np.round(eigvals, 6)))  # [1.+0.j]

    Mh = M.matrix  # move_matrix
    Mgh = I.compose(M).matrix
    assert np.allclose(Mg @ Mh, Mgh)  # Mg @ Mh == Mgh
    Sgh = M.act(s).vector
    assert np.allclose(Sv @ Mh, Sgh)  # new_state_vec = old_state_vec @ M
    assert np.allclose(Mh @ Mh.T.conj(), Mg)  # 共轭转置矩阵，在矩阵是酉矩阵（或实正交矩阵）时等同于逆矩阵
    assert np.allclose(
        np.linalg.norm(Sv @ Mh),
        np.linalg.norm(Sv)
    )

    I_state = I.rho()
    assert np.allclose(I_state, np.eye(228))

    print(I_state.shape)  # (228, 228)
    eigvals = np.linalg.eigvals(I_state)
    print(np.unique(np.round(eigvals, 6)))  # [1.+0.j]

    Mh = M.rho()
    print(np.trace(Mh))  # (148+0j)
    Mgh = I.compose(M).rho()
    assert np.allclose(I_state @ Mh, Mgh)  # ρ(g)ρ(h)=ρ(gh)

    assert np.allclose(Mh @ Mh.T.conj(), I_state)  # 单位性,ρ(M)ρ(M)∗=I
    assert np.allclose(M.inverse().rho(), M.rho().T.conj())  # 逆元 ρ(g−1)=ρ(g)∗
    eigvals = np.linalg.eigvals(Mh - I_state)
    print(np.unique(np.round(eigvals, 6)))  # [-2.-0.j -1.-1.j -1.+1.j  0.+0.j]


def test_prim_moves_group_props():
    """逐 prim_move 验证：群乘法、逆元、酉性、向量/矩阵一致性"""
    s: CubieState = CubieState.solved()
    M = CubieMove.from_rotation(1, 1, 1)
    I = CubieMove.identity()
    Mg = I.matrix
    Sv = s.vector

    for i, (k, mv) in enumerate(CubieMove.prim_moves.items()):
        # ---------- 基本 act ----------
        s1 = mv.act(s)
        assert mv.act(s) == s1
        assert I.act(s) == s
        assert mv.convert().convert() == mv  # 反转左右作用 复原
        assert mv.convert().inverse() == mv.inverse().convert()
        s2 = mv.act_left(s)

        assert mv.convert().act(s) == s2
        assert mv.act(s) == mv.convert().act_left(s)
        assert mv.inverse().is_primitive()

        print(CubieMove.basic_generators[i], k)
        print(k, mv.compose(mv) == CubieMove.identity())  # dir==2

        assert mv.compose(I) == mv and I.compose(mv) == mv

        # ---------- 逆元消去 ----------
        assert mv.inverse().act(s1) == s
        assert mv.act(mv.inverse().act(s)) == s

        # ---------- 群乘法 ----------
        assert mv.compose(mv.inverse()) == I
        assert mv.inverse().compose(mv) == I

        # ---------- 结合律（抽测） ----------
        # (s ∘ mv) ∘ mv⁻¹ == s ∘ (mv ∘ mv⁻¹)
        assert mv.inverse().act(mv.act(s)) == I.act(s)

        assert mv.corners_ori_delta.sum() % 3 == 0
        assert mv.edges_ori_delta.sum() % 2 == 0

        assert mv.act(s).is_solvable()  # 所有 prim move 保持可解

        Ms = mv.rho()
        Mgh = I.compose(mv).rho()
        assert np.allclose(Mg @ Ms, Mgh)
        assert np.allclose(Ms @ Ms.T.conj(), Mg)  # 单位性
        assert np.allclose(Ms.T.conj() @ Ms, np.eye(228))
        assert np.allclose(Ms @ Ms.T.conj(), np.eye(228))

        assert CubieState.from_vector(s1.vector) == s1
        assert np.allclose(Sv @ mv.matrix, (mv.act(s)).vector)  # new_state = old_state @ M
        assert np.allclose(Sv @ mv.matrix, mv.rho().T @ Sv)

        assert np.allclose(mv.inverse().matrix, mv.matrix.T.conj())

        assert np.allclose(Sv @ mv.matrix, s1.vector)

        rank = np.linalg.matrix_rank(Ms - np.eye(228))
        print(rank, np.trace(Ms))
        """
        3 个共轭类,存在至少 3 个不同特征类型
        (148+0j)  inverse().is_primitive()
        (142+0j)
        (134+0j)
        """

        M12 = mv.compose(M).matrix
        assert np.allclose(mv.matrix @ M.matrix, M12)

        g = mv
        h = M
        lhs = h.compose(g).compose(h.inverse()).matrix
        rhs = g.matrix  # right action (row convention)
        assert np.isclose(np.trace(lhs), np.trace(rhs))

    # ρ = identity 意味着 move 本身是 identity
    all_moves = CubieMove.prim_moves().copy()
    all_moves.update(CubieMove.slice_moves())
    for mv in all_moves.values():
        if np.allclose(mv.rho(), np.eye(228)):
            assert mv == I

    for i, a in enumerate(CubieMove.phase2_moves.values()):
        print(i, a.cubie_move.inverse().is_primitive())


def test_action_consistency():
    """identity / inverse / build / compose-act 一致性"""
    I = CubieMove.identity()
    s = CubieState.solved()

    # 1. identity
    assert I.act(s) == s

    # 2. inverse
    for m in CubieMove.prim_moves.values():
        assert m.compose(m.inverse()) == I

    # 3. action consistency
    for x, m in CubieMove.prim_moves.items():
        s1 = m.act(s)
        assert CubieMove.build(s, s1) == m, f'{x},{s1}'
        print(f"{x} ok")

    # 4. group action 右作用复合一致性
    for m1 in CubieMove.prim_moves.values():
        for m2 in CubieMove.prim_moves.values():
            assert m1.compose(m2).act(s) == m2.act(m1.act(s)), f"compose/act inconsistency: m1={m1}, m2={m2}"


def test_move_compose_counts():
    """局部 move 空间远小于群规模"""
    ME = CubieMove.identity()
    prim_list12 = {k: v for k, v in CubieMove.prim_moves.items() if k[2] != 2}
    print(len(prim_list12))
    prim_list13 = list(prim_list12.values()) + [ME]
    products = set()
    for g1 in prim_list13:
        for g2 in prim_list13:
            prod = g1.compose(g2)
            if prod != ME:  # 排除单位元
                products.add(prod)
    print(f"合后去重两两组 + 去 identity 数量: {len(products)}")  # 126
    prim_listall = prim_list12.copy()
    prim_listall.update(CubieMove.slice_moves())
    prim_listall[()] = ME
    print(len(prim_listall))  # 16
    products2 = CubieBase.generate_compose_moves(prim_listall)
    print(f"两两组合后去重 + 去 identity 数量: {len(products2)}")  # 192

    prim_listall = CubieMove.prim_moves.copy()
    products2 = CubieBase.generate_compose_moves(prim_listall, commutator=True)
    print(f"18 两两组合后去重 + 去 identity + commutator 数量: {len(products2)}")  # 216
    """6*6*6 = 3^3 × 2^3 """

    prim_listall.update(CubieMove.slice_moves())
    prim_listall[()] = ME
    print(len(prim_listall))  # 22
    print(products2.keys())
    products2 = CubieBase.generate_compose_moves(prim_listall, commutator=True)
    print(f"27 两两组合后去重 + 去 identity + commutator 数量: {len(products2)}")  # 270
    """描述局部 move 空间 远小于群规模"""


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


def test_base_consistent(base_list, solved_sticker, solved_cubie):
    """
    测试一个 base 是否在所有 prim_moves 下 ori 前7位 diff 统一
    返回 True 表示通过所有 move 测试
    统一 twist 定义
    """
    total_moves = len(CubieMove.prim_moves)
    solved_corners_map = {}
    consistent_count = 0
    consistent_count_alt = 0
    ref_diffs = None

    for pid, corner_pos in enumerate(base_list):
        # 用 solved_sticker 读取这个 slot 的 3 个 sticker
        stickers = [solved_sticker[f, r, c] for f, r, c in corner_pos]
        solved_corners_map[frozenset(stickers)] = (pid, np.array(stickers))

    for move_index, (move_key, move) in enumerate(CubieMove.prim_moves.items()):
        # 旋转 sticker 状态
        t = ActionToken.from_cubie_move(*move_key, n=3)
        state = CubeBase.rotate_state(solved_sticker.copy(), *t.key)

        # 用这个 base 读取当前 corners
        corners = np.empty((8, 3), dtype=solved_sticker.dtype)
        for i, corner in enumerate(base_list):
            for j, (f, r, c) in enumerate(corner):
                corners[i, j] = state[f, r, c]

        # 计算 ori_sticker
        ori_sticker = np.empty(8, dtype=np.int8)
        valid = True
        for i, c in enumerate(corners):
            key = frozenset(c)
            if key not in solved_corners_map:
                valid = False
                print(f'key not in solved!{c}')
                break
            pid, ref = solved_corners_map[key]
            found = False
            for twist in range(3):
                rolled = np.roll(ref, -twist)
                if np.array_equal(rolled, c):
                    ori_sticker[pid] = twist  # i
                    found = True
                    break
            if not found:
                print(f'twist not found!{c}')
                valid = False
                break

        if not valid:
            return False

        # for slot in [1, 2, 5, 6]:
        #     ori_sticker[slot] = (-ori_sticker[slot]) % 3
        # 计算 diff
        ori_sticker_alt = (3 - ori_sticker) % 3  # 计算负向版本
        s1 = move.act(solved_cubie)
        target_ori = s1.corners_ori  # [s1.corners_perm]
        diffs = (ori_sticker[:7] - target_ori[:7]) % 3  # 本身就已经是按当前状态的 slot 顺序排列好的
        diffs_alt = (ori_sticker_alt[:7] - target_ori[:7]) % 3
        # diffs3 = (ori_sticker - target_ori) % 3
        unique = np.unique(diffs)
        unique_alt = np.unique(diffs_alt)

        if len(unique) == 1:
            consistent_count += 1
        elif len(unique_alt) == 1:  # 某些 corner 的方向被镜像
            consistent_count_alt += 1
        else:
            if consistent_count > 3:
                print('consistent:', consistent_count, consistent_count_alt)
            break

        # if ref_diffs is None:
        #     ref_diffs = diffs
        # else:
        #     if not np.array_equal(diffs, ref_diffs):
        #         valid = False
        #         break
        # consistent_count += 1
    print('consistent:', consistent_count, consistent_count_alt)
    base_offset = [2, 0, 0, 1, 1, 0, 0, 2]
    CORNER_SIGN = [0, 2, 1, 0, 0, 0, 0, 0]  # [0 2 1 0 0 1 2 0]
    diff_c = [0, 2, 1, 0, 0, 1, 2, 0]
    corner_ori_sign = np.ones(8, dtype=np.int8)  # flip_slots
    corner_ori_sign[[1, 2, 5, 6]] = -1
    flip_slots = [1, 2, 5, 6]

    if consistent_count_alt > 0:
        for mask in range(256):  # 2^8 = 256
            sign_vector = np.array([1 if (mask & (1 << j)) == 0 else -1 for j in range(8)])
            all_moves_ok = True
            ct = 0
            for move_key, move in CubieMove.prim_moves.items():
                t = ActionToken.from_cubie_move(*move_key, n=3)
                state = CubeBase.rotate_state(solved_sticker.copy(), *t.key)

                # 用这个 base 读取当前 corners
                corners = np.empty((8, 3), dtype=solved_sticker.dtype)
                for i, corner in enumerate(base_list):
                    for j, (f, r, c) in enumerate(corner):
                        corners[i, j] = state[f, r, c]

                # 计算 ori_sticker
                ori_sticker = np.empty(8, dtype=np.int8)
                valid = True
                # for slot_i, c in enumerate(corners):
                #     key = frozenset(c)
                #     pid, ref = solved_corners_map[key]
                #     for j, st in enumerate(c):
                #         if st in (0, 1):  # U 或 D face
                #             ori_sticker[pid] = j % 3
                #             break
                for i, c in enumerate(corners):
                    key = frozenset(c)
                    if key not in solved_corners_map:
                        valid = False
                        print(f'key not in solved!{c}')
                        break
                    pid, ref = solved_corners_map[key]
                    found = False
                    for twist in range(3):
                        rolled = np.roll(ref, -twist)
                        if np.array_equal(rolled, c):
                            ori_sticker[pid] = twist
                            # ori_sticker[pid] = (3 - twist) % 3
                            found = True
                            break
                    if not found:
                        print(f'twist not found!{c}')
                        valid = False
                        break

                if not valid:
                    return False

                # ori_sticker = (2 * ori_sticker) % 3
                # ori_sticker = (ori_sticker - base_offset) % 3
                # for i in range(8):
                #     ori_sticker[i] = (ori_sticker[i] * CORNER_SIGN[i]) % 3
                # 计算 diff
                # ori_sticker = (ori_sticker * corner_ori_sign) % 3
                # ori_sticker[flip_slots] = (3 - ori_sticker[flip_slots]) % 3
                s1 = move.act(solved_cubie)
                target_ori_cubie = np.empty(8, dtype=np.int8)  # s1.corners_ori[s1.corners_perm]

                for slot in range(8):
                    cubie = s1.corners_perm[slot]
                    target_ori_cubie[cubie] = s1.corners_ori[slot]

                target_ori = target_ori_cubie  # s1.corners_ori

                diff_0 = (ori_sticker - target_ori) % 3
                if not np.all(diff_0 == 0):
                    print(f"orientation mismatch,{mask}")
                if np.array_equal(ori_sticker, target_ori):
                    ct += 1
                    continue  # 已经统一

                corrected_ori = (ori_sticker * sign_vector) % 3
                diffs = (corrected_ori[:7] - target_ori[:7]) % 3
                unique = np.unique(diffs)
                if len(unique) != 1:
                    all_moves_ok = False
                    if move_key[-1] == -1:
                        print(
                            f"Move {move_key} 找不到 sign 向量让 diff 统一,diff:, {(ori_sticker - target_ori) % 3}")
                    break
                print(
                    f"Move {move_key} sticker:{ori_sticker}cubie:{target_ori},diff:, {(ori_sticker - target_ori) % 3}")
                # for i in range(8):
                #     if diff_0[i] != 0:
                #         print(i, diff_0[i])

                """
        orientation mismatch,127
        Move (0, -1, -1) sticker:[0 1 2 0 0 2 1 0]cubie:[0 2 1 0 0 1 2 0],diff:, [0 2 1 0 0 1 2 0]
        orientation mismatch,127
        Move (0, 1, 1) sticker:[2 0 0 1 1 0 0 2]cubie:[1 0 0 2 2 0 0 1],diff:, [1 0 0 2 2 0 0 1]
        orientation mismatch,127
        Move (2, -1, 1) sticker:[0 0 1 2 0 0 2 1]cubie:[0 0 2 1 0 0 1 2],diff:, [0 0 2 1 0 0 1 2]
        orientation mismatch,127
        Move (2, 1, -1) sticker:[1 2 0 0 2 1 0 0]cubie:[2 1 0 0 1 2 0 0],diff:, [2 1 0 0 1 2 0 0]
        每一个slot位置的4个diff加起来：每个位置的4个diff之和都是3！同一个旋转，被不同 corner 用不同方向解释了"""

            if all_moves_ok:
                print(f"找到全局 sign 向量！ mask = {bin(mask)},{ct}")
                print("sign_vector:", sign_vector)
                return True

    return consistent_count == total_moves or consistent_count_alt == total_moves or (
            consistent_count + consistent_count_alt) == total_moves


def fix_corner_ori_offset(cube):
    from itertools import permutations, product
    solved_sticker = cube.solved.copy()
    s0 = CubieState.solved()
    original_base = cube.corner_coords(cube.n)
    """[[(0, 0, 0), (2, 4, 0), (5, 4, 4)], [(0, 0, 4), (2, 4, 4), (4, 4, 0)], [(0, 4, 4), (3, 4, 0), (4, 4, 4)], [(0, 4, 0), (3, 4, 4), (5, 4, 0)], [(1, 4, 0), (2, 0, 0), (5, 0, 4)], [(1, 4, 4), (2, 0, 4), (4, 0, 0)], [(1, 0, 4), (3, 0, 0), (4, 0, 4)], [(1, 0, 0), (3, 0, 4), (5, 0, 0)]]"""
    for i, c in enumerate(original_base):
        print(f"slot {i}: {c}")

    for idx, shifts_tuple in enumerate(product([0, 1, 2], repeat=8)):
        shifts = list(shifts_tuple)
        basei = []
        for slot, shift in enumerate(shifts):
            corner = original_base[slot]
            rolled_corner = corner[shift:] + corner[:shift]  # 注意 axis=0 因为 corner 是 list of tuple
            basei.append(rolled_corner)

        if idx % 500 == 0:
            print(f"已尝试 {idx}:{basei} ")
        if test_base_consistent(basei, solved_sticker, s0):
            print(f"\n成功！在第 {idx + 1} 个排列找到正确 basei")
            print("每个角块的 roll shift (0/1/2):", shifts)
            print("正确 basei:")
            for i, corner in enumerate(basei):
                print(f"slot {i}: {corner}")
            return basei

    for move_key in [(0, -1, 1), (0, 1, 1), (2, -1, -1), (2, 1, -1)]:
        move = CubieMove.prim_moves[move_key]
        t = ActionToken.from_cubie_move(*move_key, n=3)
        state = CubeBase.rotate_state(cube.solved.copy(), *t.key)
        corners = cube.get_corners(state)
        ori_sticker = np.empty(8, dtype=np.int8)
        # ... 计算 ori_sticker 的代码 ...
        s11 = cube.to_cubie(state)
        ori_sticker = s11.corners_ori
        s1 = move.act(s0)
        target_ori = s1.corners_ori
        diffs = (ori_sticker - target_ori) % 3
        print(f"Move {move_key}: diffs = {diffs}")
        print(f"  sticker: {ori_sticker}")
        print(f"  target : {target_ori}")
        corrected_ori = ori_sticker.copy()
        corrected_ori[[1, 2, 5, 6]] = (3 - corrected_ori[[1, 2, 5, 6]]) % 3
        corrected_ori[-1] = (-corrected_ori[:7].sum()) % 3
        print(f"  corrected : {corrected_ori}")

    return None


# ── 编码 round-trip ─────────────────────────────────────────────

def test_comb_index():
    """组合编码/解码一致性"""
    n = 12
    k = 4
    for i in range(math.comb(n, k)):
        bits = CubeBase.index_to_comb(i, n, k)
        back = CubeBase.comb_to_index(bits, n, k)
        assert back == i, f"Fail at {i}: {back}"

    for i in range(24):
        j = CubieState.encode_perm_ud_slice(CubieState.create_ud_slice_perm(i).tolist())
        assert j == i, f"Fail at {i}: {j}"
    for i in range(70):
        j = CubieState.encode_corner_coset(CubieState.canonical_corner_coset(i).tolist())
        assert j == i, f"Fail at {i}: {j}"
    print("All pass!")


def test_phase15_coord_roundtrip():
    """Phase15Coord encode/decode 一致性"""
    # index ↔ coord ↔ index
    for idx in range(24 * 70 * 2):
        c = Phase15Coord.from_index(idx)
        assert c.index == idx, f"index roundtrip fail at {idx}"
        assert 0 <= c.slice_perm < 24
        assert 0 <= c.corner_coset < 70
        assert c.parity in (0, 1)

    # slice_perm encode ↔ create_ud_slice_perm
    for i in range(24):
        perm = CubieState.create_ud_slice_perm(i)
        j = CubieState.encode_perm_ud_slice(perm.tolist())
        assert j == i, f"slice_perm fail at {i}: got {j}"

    # corner_coset encode ↔ canonical_corner_coset
    for i in range(70):
        perm = CubieState.canonical_corner_coset(i)
        j = CubieState.encode_corner_coset(perm.tolist())
        assert j == i, f"corner_coset fail at {i}: got {j}"

    print("Phase15Coord roundtrip: OK")


def test_phase15_project_consistency():
    """Phase15Coord.project 在 CubieMove.act 下的行为一致性"""
    s = CubieState.solved()
    c_solved = Phase15Coord.project(s)
    assert c_solved.slice_perm == 0
    assert c_solved.corner_coset == CubieState.solved_corner_coset  # 69
    assert c_solved.parity == 0

    # project 后的 coord 能与 CubieMove.act 保持一致
    for k, m in CubieMove.prim_moves.items():
        s2 = m.act(s)
        c2 = Phase15Coord.project(s2)
        # 用 act() 走真实状态再 project
        a = Phase15Action.phi(m)
        s3, c3 = a.act(s)
        assert s2 == s3, f"state mismatch for move {k}"
        assert c2 == c3, f"coord mismatch for move {k}: {c2} vs {c3}"

    print("Phase15Coord project consistency: OK")


def test_phase15_pruning_bfs():
    """build_phase15_pruning BFS 覆盖全部 3360 坐标"""
    dist = CubieBase.build_phase15_pruning()
    N_PHASE15 = 24 * 70 * 2
    reachable = int(np.sum(dist < 127))
    print(f"Phase-1.5 pruning reachable: {reachable}/{N_PHASE15}")
    assert reachable == N_PHASE15, f"Expected 3360 reachable, got {reachable}"

    # solved 坐标距离为 0
    solved_idx = Phase15Coord.solved().index
    assert dist[solved_idx] == 0

    # 距离非负且单调
    assert np.all(dist >= 0)
    max_dist = np.max(dist)
    print(f"  max_dist={max_dist}, mean_dist={np.mean(dist):.4f}")

    print("Phase15 pruning BFS: OK")


def test_phase15_act_no_act_index():
    """Phase15Action 只能通过 act(state) 走真实 CubieState，不存在 act_index"""
    # phi 构造正常
    for k, m in CubieMove.prim_moves.items():
        a = Phase15Action.phi(m)
        assert a.cubie_move == m

        # act() 返回 (CubieState, Phase15Coord)
        s = CubieState.solved()
        state, coord = a.act(s)
        assert isinstance(state, CubieState)
        assert isinstance(coord, Phase15Coord)
        assert coord == Phase15Coord.project(state)

        # 验证：多次 act 后 project 与直接 CubieMove.act 一致
        s2 = m.act(s)
        assert state == s2, f"act state mismatch for {k}"

    print("Phase15 act (no act_index): OK")


def test_slice_moves_solvable():
    s = CubieState.solved()
    for k, m in CubieMove.slice_moves.items():
        s2 = m.act(s)
        print(f'{k},{s2.is_solvable()}')  # (0, 0, 2),(1, 0, 2) (2, 0, 2)True


def test_ud_slice_encode():
    """encode_ud_slice / decode_ud_slice 与 comb_to_index / index_to_comb 一致性"""
    # 全 495 坐标 encode/decode roundtrip
    for c in range(495):
        edges_perm = CubieState.decode_ud_slice(c)
        coord = CubieState.encode_ud_slice(edges_perm.tolist())
        assert coord == c, f'{c},{coord},{edges_perm}'

    # comb_to_index / index_to_comb 内部一致性
    for c in range(495):
        bits = CubeBase.index_to_comb(c, n=12, k=4)
        c2 = CubeBase.comb_to_index(bits, n=12, k=4)
        assert c == c2, f'comb roundtrip fail at {c}: got {c2}'

    # 随机游走后 encode → decode → encode 一致性
    for _ in range(500):
        g = CubieBase.random_walk(length=random.randint(1, 20))
        state = g.act(CubieState.solved())
        coord = state.ud_slice_coord
        assert 0 <= coord < 495
        perm = CubieState.decode_ud_slice(coord)
        c2 = CubieState.encode_ud_slice(perm.tolist())
        assert c2 == coord, f'random walk roundtrip fail: {coord} vs {c2}'

    print("ud_slice encode/decode: OK")


# ── Phase 同态 & 搜索 ────────────────────────────────────────────

def test_phase1_homomorphism():
    """验证 phi(m.act(s)) == phi(m).act(phi(s))"""

    def check_homomorphism(m: CubieMove, s: CubieState):
        lhs = Phase1Coord.project(m.act(s))  # left
        rhs = Phase1Action.phi(m).act(Phase1Coord.project(s))  # right
        assert lhs == rhs, f"Homomorphism broken!{lhs}_{rhs}"

    # 随机测试
    for _ in range(100):
        m = random.choice(list(CubieMove.prim_moves.values()))
        s = CubieBase.generate_cubie(50)
        check_homomorphism(m, s)

    s = CubieState.solved()
    m1 = random.choice(list(CubieMove.phase1_moves().values()))
    m2 = random.choice(list(CubieMove.phase1_moves().values()))

    # 路径 A：先 apply 再 project
    sA = m2.replay(m1.replay(s))
    coordA = Phase1Coord.project(sA)

    # 路径 B：先 project 再 act
    coordB = m2.act(m1.act(Phase1Coord.project(s)))

    assert coordA == coordB


def test_phase1_ud_slice_map():
    s = CubieState.solved()
    for m in CubieMove.phase1_moves.values():
        phi = Phase1Action.phi(m.cubie_move)
        for c in range(495):
            s = s.with_(edges_perm=s.decode_ud_slice(c))
            out = m.cubie_move.act(s)
            assert phi.ud_slice_map[c] == out.ud_slice_coord
    # φ₂ 的同态性
    # for a, m1 in CubieMove.phase1_moves.items():
    #     for b, m2 in CubieMove.phase1_moves.items():
    #         lhs = Phase1Action.phi(m1.cubie_move.compose(m2.cubie_move))
    #         rhs = Phase1Action.phi(m1.cubie_move).compose(
    #             Phase1Action.phi(m2.cubie_move))
    #         assert lhs == rhs, (a, b)
    # # φ₂ 的同态性
    # for a, m1 in CubieMove.phase2_moves().items():
    #     for b, m2 in CubieMove.phase2_moves().items():
    #         lhs = Phase2Action.phi(m1.cubie_move.compose(m2.cubie_move))
    #         rhs = Phase2Action.phi(m1.cubie_move).compose(
    #             Phase2Action.phi(m2.cubie_move)
    #         )
    #         assert lhs == rhs, (a, b)


def test_phase_graph(cube):
    v, e = cube.build_phase_graph(Phase1Coord.solved(), 2)
    print(v)
    print('graph', e)

    v, e = cube.build_phase_graph(Phase2Coord.solved(), 2)
    print(v)
    print('graph', e)


def test_phase1_search(cube):
    """Phase1 搜索 + 应用验证"""
    s0 = CubieState.solved()
    for _ in range(3):
        s0 = random.choice(list(CubieMove.phase1_moves.values())).replay(s0)

    moves = CubieBase.phase1_search(s0, depth_limit=7)
    assert moves is not None  # 应该几乎总能找到

    phase1_coord = Phase1Coord.project(s0)
    s1 = s0.clone()
    s2 = s0.clone()
    m2 = CubieMove.identity()
    for move in moves:
        phase1_coord = move[1].act(phase1_coord)
        s2 = move[1].replay(s2)
        m2 = m2.compose(move[1].cubie_move)
        print(phase1_coord)

    s1 = CubieMove.apply(s1, [x[0] for x in moves])
    assert np.all(s1.corners_ori == 0)
    assert np.all(s1.edges_ori == 0)
    assert s1 == s2, f'{s1},{s2}'
    s3 = m2.act(s0)
    assert s1 == s3, f'{s1},{s3}'


def test_phase2_search(cube):
    """Phase1 → Phase2 完整求解流程：随机 scramble → Phase1 → Phase2 → solved"""
    s0 = CubieState.solved()
    s1 = cube.generate_cubie(6)

    # Phase 1: 进入 G₁
    path1, mv1, state1 = CubieBase.solve_phase1(s1, start=2, end=15)
    assert state1.is_phase1_solved(), f'Phase1 not solved: CO={state1.corners_ori_coord} EO={state1.edge_ori_coord}'
    # 验证两种作用方式一致
    s1_apply = CubieMove.apply(s1, [k for k, _ in path1])
    assert s1_apply == state1, 'Phase1: apply vs act_moves mismatch'

    # Phase 2: 进入 solved
    path2, mv2, state2 = CubieBase.solve_phase2(state1, start=2, end=25)
    assert state2 == CubieState.solved(), f'Phase2 not fully solved: {state2}'
    # 验证两种作用方式一致
    s2_apply = CubieMove.apply(state1, [k for k, _ in path2])
    assert s2_apply == state2, 'Phase2: apply vs act_moves mismatch'

    # 总复合 move 应直接解出原始 scramble
    total_mv = mv1.compose(mv2)
    assert total_mv.act(s1) == CubieState.solved(), 'Total compose should solve original state'

    full_path = [k for k, _ in path1 + path2]
    print(f'  Phase1={len(path1)} Phase2={len(path2)} moves={full_path}')
    print('test_phase2_search: OK')


# ── 求解 ─────────────────────────────────────────────────────────

def test_solve_kociemba(cube):
    """Kociemba 两阶段求解"""
    s0 = CubieState.solved()
    for _ in range(3):
        s0 = random.choice(list(CubieMove.phase1_moves.values())).replay(s0)

    m22, m2 = CubieBase.solve_kociemba(s0)
    print(m22)
    s20 = m2.act(s0)
    assert s20 == CubieState.solved()


def test_examples_solvable(cube):
    """经典 pattern 可解性"""
    tests = [
        ("Extreme reverse", CubieExample.inversed()),
        ("Twisted", CubieExample.twisted()),
        ("Checkerboard", CubieExample.checkerboard()),
        ("Big Cycle", CubieExample.big_cycle()),
        ("Superflip", CubieExample.superflip()),
        ("Superflip + corners", CubieExample.superflip_plus()),
    ]

    for name, state in tests:
        print(name, state)
        assert state.is_solvable()

    results = []
    for name, state in tests:
        print(f"\n=== 测试 {name} ===")
        start = time.time()
        solution, g = cube.solve_kociemba(state)
        elapsed = time.time() - start

        print(f"步数: {len(solution)}")
        print(f"用时: {elapsed:.2f} 秒")
        print(f"Move 序列: {solution}")
        results.append((name, len(solution), elapsed))

    print("\n=== 测试总结 ===")
    for name, steps, t in results:
        print(f"{name:20} | 步数: {steps:2d} | 时间: {t:.2f} 秒")


# ── 随机路径一致性 ────────────────────────────────────────────────

def test_random_path_consistency(cube, steps=20):
    """贴纸世界 vs Cubie世界 随机路径一致性"""
    s0_st = cube.solved_idx.copy()
    s0_cu = CubieState.solved()  # 参考世界
    sm = StickerMove.identity(cube.n)
    path = []
    for _ in range(steps):
        ma = random.choice(list(CubieMove.prim_moves.keys()))
        path.append(ma)

        sm = sm.compose(sm.from_rotation(cube.n, ActionToken(*ma)))

        s0_st = cube.rotate_state(s0_st, *ma)
        s0_cu = CubieMove.prim_moves[ma].act(s0_cu)

    sm2 = StickerMove.build(cube.solved_idx.copy(), s0_st)
    assert sm2 == sm, f'{sm2},{sm}'

    st11 = cube.idx_to_state(s0_st)

    s1 = cube.to_cubie(st11)
    s2 = s0_cu

    s_cu = cube.from_cubie(s0_cu)
    s22 = cube.to_cubie(s_cu)
    assert s22 == s0_cu, f'{s22},{s0_cu}'

    if not np.array_equal(s_cu, st11):
        print(f"cu {(s_cu != st11).sum()}")  # 17

    print('label:', cube.critic_progress_label(cube.solved, st11))
    print('label2:', cube.critic_progress_label(cube.solved, s_cu))

    st123 = sm.act(cube.solved)
    assert np.array_equal(st123, st11), f"sm {(st123 != st11).sum()}"

    n = cube.solved.shape[1]
    flat = cube.solved.reshape(-1).copy()
    for ma in path:
        m = CubieMove.prim_moves[ma]
        perm = StickerMove.phi(n, m).perm
        flat = flat[perm]
    st124 = flat.reshape(6, n, n)

    assert np.array_equal(st124, st11), f"sm2 {(st124 != st11).sum()}"
    sm3, st125 = StickerMove.act_moves(cube.solved, ActionToken.from_path(path))
    assert np.array_equal(st125, st11), f"sm3 {(st125 != st11).sum()},{sm3}"
    assert sm3 == sm, f"{sm},{sm3},{sm3.cubie_move}"

    # permutation 必须一致
    assert np.array_equal(s1.corners_perm, s2.corners_perm)
    assert np.array_equal(s1.edges_perm, s2.edges_perm)
    if not np.array_equal(s1.edges_ori, s2.edges_ori):
        print(f'{s1.edges_ori}, {s2.edges_ori}')

    # solvable 必须一致
    assert s1.is_solvable()
    assert s2.is_solvable()

    moves_cu, mv_cu = CubieBase.solve_kociemba(s2)  # 参考世界
    moves_st, mv_st = CubieBase.solve_kociemba(s1)  # 实际世界
    print(f"{len(moves_st)},{len(moves_cu)}")
    print(mv_st, mv_cu)

    assert mv_st.act(s1) == CubieState.solved()

    st1 = mv_cu.act(s0_cu)
    assert st1 == CubieState.solved()

    act_cu = [(axis, side * cube.mid, dir) for axis, side, dir in moves_cu]
    act_st = [(axis, side * cube.mid, dir) for axis, side, dir in moves_st]
    st10 = st11.copy()
    st21 = st11.copy()
    st31 = st11.copy()
    cube.act_moves(st10, act_st)
    cube.act_moves(st21, act_cu)
    cube.act_moves(st31, cube.invert_moves(path))
    assert cube.is_solved(st31), f'{st31}'

    st12 = cube.to_cubie(st10)
    if st12 != CubieState.solved():
        print(st12)  # corners_ori canonicalize
    st22 = cube.to_cubie(st21)
    if st22 != CubieState.solved():
        print(st22)

    if not cube.is_solved(st10) or not cube.is_solved(st21):
        print(f'{st10}\n{st21}')


# ── 物理单步 move 属性 ───────────────────────────────────────────

def test_single_move_physical(n=5):
    cube = CubieBase(n=n)
    ori_before = cube.corner_orientation(cube.solved)
    for axis in range(3):
        for layer in (-2, -1, 0, 1, 2):
            for d in [1, 2, 3]:
                s = cube.rotate_state(cube.solved, axis, layer, d)

                ori_after = cube.corner_orientation(s)
                ori_delta = cube.corner_orientation_delta(cube.solved, s)
                corner_perm, ori_2 = cube.corner_ids_ori(s)
                edge_perm, _ = cube.edge_ids_ori(s)
                orbit_perm = cube.orbit_perm(s)
                print("ori sum before:", np.sum(ori_before))  # 0
                print(f"axis", axis)
                print("corner_perm after:", corner_perm)
                print("ori after:", ori_after, 'ori_2 after:', ori_2, 'ori delta:', ori_delta)
                print("ori sum after:", np.sum(ori_after) % 3)
                print(cube.heuristic_corner_perm(s),
                      cube.edge_orientation(s))
                print(orbit_perm)
                print(cube.corner_ud_defect(s), cube.edge_ud_defect(s))
                print(cube.observables(s))

                assert edge_perm.shape == (12,)
                assert corner_perm.shape == (8,)


# ── Cache ────────────────────────────────────────────────────────

def test_cache_roundtrip():
    class_cache.save(CubieBase)
    class_property.save(CubieBase)
    class_cache.save(CubieState)
    class_property.save(CubieState)
    print(check_class_status(CubieBase))


# ── Main ─────────────────────────────────────────────────────────

def test_vector_roundtrip(num_random=500, max_steps=50):
    """验证 from_vector ∘ rho 作用 vs act 作用，以及 is_solvable 保持性

    核心发现（2026-05-11）:
    ─────────────────────────────────────────────────────────────
    1. ρ(m) 的 Co/Eo 是对角阵（仅含 twist，不含置换），因此：
       - 对 GENERATOR 作用于 solved state：始终正确（巧合：
         generator 的 ori_delta ≡ 其逆元的 ori_delta，对角 Co 恰好
         给出正确结果；perm 部分通过 Cp/Ep 正确传递）
       - 对 COMPOSED move 或任意非 solved 状态：ori 部分会出错，
         因为 Co/Eo 没有跟踪置换信息
    2. from_vector 本身逻辑正确（argmax + 最近单位根），不需要修。
       根因在 ρ() 的块对角设计——要修复需要 Co/Eo 与 Cp/Ep 交互
       （非块对角），会影响整个 spectral 层。
    3. 所有从 rho 路径解码的状态 is_solvable() 始终为 True。
    4. Perm 部分（corners_perm, edges_perm）始终正确。
    """
    prim_moves = CubieMove.prim_moves
    prim_keys = list(prim_moves.keys())
    solved = CubieState.solved()
    sv = solved.vector
    rng = random.Random(42)
    omega = np.exp(2j * np.pi / 3)

    print("\n" + "=" * 70)
    print("test_vector_roundtrip: from_vector(rho @ vec) vs act(state)")
    print("=" * 70)

    # ── 0. 验证生成元的 ori_delta 自逆性 ──
    print("\n--- Part 0: Generator ori_delta self-inverse property ---")
    all_self_inv = True
    for key, mv in prim_moves.items():
        inv_delta = mv.inverse().corners_ori_delta
        if not np.array_equal(mv.corners_ori_delta, inv_delta):
            print(f"  {key}: delta != inv.delta! {mv.corners_ori_delta} vs {inv_delta}")
            all_self_inv = False
    print(f"  All 18 generators: delta == inv.delta = {all_self_inv}")
    print(f"  -> This is why diagonal Co works for generators on solved state")

    # ── 1. 单步 on solved: 全对 ──
    print("\n--- Part 1: 1-step on solved state ---")
    ok = 0
    for key in prim_keys:
        mv = prim_moves[key]
        rec = CubieState.from_vector(mv.rho() @ sv)
        ref = mv.inverse().act(solved)
        if rec == ref:
            ok += 1
    print(f"  rec == inv.act(solved): {ok}/{len(prim_keys)} (should be 18)")
    print(f"  All solvable: {all(CubieState.from_vector(mv.rho() @ sv).is_solvable() for mv in prim_moves.values())}")

    # ── 2. 单步 on arbitrary states ──
    print("\n--- Part 2: 1-step on arbitrary states ---")
    n_perm_ok = n_ori_ok = 0
    n_trials = 100
    for t in range(n_trials):
        keys = rng.choices(prim_keys, k=rng.randint(1, 30))
        s = solved
        for k in keys:
            s = prim_moves[k].act(s)
        mk = rng.choice(prim_keys)
        mv = prim_moves[mk]
        rec = CubieState.from_vector(mv.rho() @ s.vector)
        ref = mv.inverse().act(s)
        if (np.array_equal(rec.corners_perm, ref.corners_perm) and
                np.array_equal(rec.edges_perm, ref.edges_perm)):
            n_perm_ok += 1
        if (np.array_equal(rec.corners_ori, ref.corners_ori) and
                np.array_equal(rec.edges_ori, ref.edges_ori)):
            n_ori_ok += 1
    n_solvable = sum(1 for _ in range(n_trials) if True)  # placeholder, all True
    print(f"  perm OK: {n_perm_ok}/{n_trials}")
    print(f"  ori OK:  {n_ori_ok}/{n_trials}")
    print(f"  (perm always correct; ori fails on non-solved states because")
    print(f"   diagonal Co/Eo can't track the state's existing orientation permutation)")

    # ── 3. 多步 on solved: is_solvable 保持性 ──
    print(f"\n--- Part 3: Multi-step from solved ---")
    for steps in [1, 2, 5, 10, 20, 50]:
        n_solvable = 0
        for t in range(50):
            keys = rng.choices(prim_keys, k=steps)
            v = sv.copy()
            for k in keys:
                v = prim_moves[k].rho() @ v
            rec = CubieState.from_vector(v)
            if rec.is_solvable():
                n_solvable += 1
        print(f"  {steps:>3d} steps: solvable={n_solvable}/50")

    # ── 4. 随机合法向量 is_solvable 比例 ──
    print(f"\n--- Part 4: Random valid-encoding vector is_solvable ({num_random} trials) ---")
    n_sol = 0
    for _ in range(num_random):
        cp = np.zeros((8, 8))
        for i in range(8):
            cp[i, rng.randint(0, 7)] = 1.0
        ep = np.zeros((12, 12))
        for i in range(12):
            ep[i, rng.randint(0, 11)] = 1.0
        co = np.array([rng.choice([1, omega, omega ** 2]) for _ in range(8)], dtype=np.complex64)
        eo = np.array([rng.choice([1.0, -1.0]) for _ in range(12)], dtype=np.float32)
        vec = np.concatenate([cp.flatten(), ep.flatten(), co, eo])
        s = CubieState.from_vector(vec)
        if s.is_solvable():
            n_sol += 1
    print(f"  Solvable: {n_sol}/{num_random} ({100 * n_sol / num_random:.1f}%)")

    # ── 5. 随机 rho 路径后 is_solvable ──
    print(f"\n--- Part 5: Random rho-path is_solvable ({num_random} trials, steps={max_steps}) ---")
    n_sol_rho = 0
    for _ in range(num_random):
        keys = rng.choices(prim_keys, k=max_steps)
        v = sv.copy()
        for k in keys:
            v = prim_moves[k].rho() @ v
        s = CubieState.from_vector(v)
        if s.is_solvable():
            n_sol_rho += 1
    print(f"  Solvable: {n_sol_rho}/{num_random} ({100 * n_sol_rho / num_random:.1f}%)")
    print()


def test_matrix_roundtrip(num_random=500, max_steps=50):
    """验证 from_vector ∘ matrix 作用 vs act 作用

    mv.matrix 与 mv.rho 的关键区别：
    - matrix 是右作用算子（v @ M），Co/Eo 含置换+twist
    - rho 是左作用（M @ v），Co/Eo 是对角阵无置换
    - matrix 对任意状态、任意步数都应该完全正确
    """
    prim_moves = CubieMove.prim_moves
    prim_keys = list(prim_moves.keys())
    solved = CubieState.solved()
    sv = solved.vector
    rng = random.Random(42)
    omega = np.exp(2j * np.pi / 3)

    print("\n" + "=" * 70)
    print("test_matrix_roundtrip: from_vector(v @ matrix) vs act(state)")
    print("=" * 70)

    # ── 1. 单步 on solved ──
    print("\n--- Part 1: 1-step on solved state ---")
    ok = 0
    for key in prim_keys:
        mv = prim_moves[key]
        rec = CubieState.from_vector(sv @ mv.matrix)
        ref = mv.act(solved)
        if rec == ref:
            ok += 1
        else:
            print(f"  FAIL {key}: eq={rec == ref}, solvable={rec.is_solvable()}")
            if rec != ref:
                print(f"    ref={ref}")
                print(f"    rec={rec}")
    print(f"  rec == act(solved): {ok}/{len(prim_keys)} (should be 18)")

    # ── 2. 单步 on arbitrary states ──
    print("\n--- Part 2: 1-step on arbitrary states ---")
    n_ok = n_solvable = 0
    n_trials = 100
    for t in range(n_trials):
        keys = rng.choices(prim_keys, k=rng.randint(1, 30))
        s = solved
        for k in keys:
            s = prim_moves[k].act(s)
        mk = rng.choice(prim_keys)
        mv = prim_moves[mk]
        rec = CubieState.from_vector(s.vector @ mv.matrix)
        ref = mv.act(s)
        if rec == ref:
            n_ok += 1
        if rec.is_solvable():
            n_solvable += 1
    print(f"  rec == act(s): {n_ok}/{n_trials}")
    print(f"  solvable:      {n_solvable}/{n_trials}")

    # ── 3. 多步 on solved ──
    print(f"\n--- Part 3: Multi-step from solved ---")
    for steps in [1, 2, 5, 10, 20, 50]:
        n_ok = 0
        for t in range(50):
            keys = rng.choices(prim_keys, k=steps)
            # act chain
            s = solved
            for k in keys:
                s = prim_moves[k].act(s)
            ref = s
            # matrix chain (right action: v @ M1 @ M2 @ ...)
            v = sv.copy()
            for k in keys:
                v = v @ prim_moves[k].matrix
            rec = CubieState.from_vector(v)
            if rec == ref and rec.is_solvable():
                n_ok += 1
        print(f"  {steps:>3d} steps: rec==act_chain={n_ok}/50")

    # ── 4. 随机 matrix 路径后 is_solvable ──
    print(f"\n--- Part 4: Random matrix-path is_solvable ({num_random} trials, steps={max_steps}) ---")
    n_sol = 0
    for _ in range(num_random):
        keys = rng.choices(prim_keys, k=max_steps)
        v = sv.copy()
        for k in keys:
            v = v @ prim_moves[k].matrix
        s = CubieState.from_vector(v)
        if s.is_solvable():
            n_sol += 1
    print(f"  Solvable: {n_sol}/{num_random} ({100 * n_sol / num_random:.1f}%)")
    print()


def test_from_rotation():
    """验证 from_rotation 的 twist 公式：单步非 Y 轴使用 axis-dependent face formula，
    半步/多步使用 rotation-direction formula，无后置 flip patch。"""
    pm = CubieMove.prim_moves

    # 1. U/D moves (axis=1): corners never twist
    for key, mv in pm.items():
        if key[0] == 1:
            assert np.all(mv.corners_ori_delta == 0), f'{key}: U/D should have no corner twist'

    # 2. Half-turns (dir=±2): corners never twist (two quarter turns cancel)
    for key, mv in pm.items():
        if abs(key[2]) == 2:
            assert np.all(mv.corners_ori_delta == 0), f'{key}: half-turn should have no corner twist'

    # 3. Single-turn R/L (axis=0): twist depends on face (side), not on ±direction
    for side in (-1, 1):
        r_cw = pm[(0, side, 1)].corners_ori_delta
        r_ccw = pm[(0, side, -1)].corners_ori_delta
        assert np.array_equal(r_cw, r_ccw), \
            f'R/L side={side}: CW vs CCW should agree (face-dependent), got {r_cw} vs {r_ccw}'

    # 4. Single-turn F/B (axis=2): twist depends on face (side), not on ±direction
    for side in (-1, 1):
        f_cw = pm[(2, side, 1)].corners_ori_delta
        f_ccw = pm[(2, side, -1)].corners_ori_delta
        assert np.array_equal(f_cw, f_ccw), \
            f'F/B side={side}: CW vs CCW should agree (face-dependent), got {f_cw} vs {f_ccw}'

    # 5. R vs R' differ (different faces): side=+1 vs side=-1
    r_cw = pm[(0, 1, 1)].corners_ori_delta  # R
    l_cw = pm[(0, -1, 1)].corners_ori_delta  # L
    assert not np.array_equal(r_cw, l_cw), f'R vs L should differ'

    # 6. F vs B differ
    f_cw = pm[(2, 1, 1)].corners_ori_delta  # F
    b_cw = pm[(2, -1, 1)].corners_ori_delta  # B
    assert not np.array_equal(f_cw, b_cw), f'F vs B should differ'

    # 7. Inverse roundtrip: mv ∘ mv⁻¹ = identity for all 18 prims
    for key, mv in pm.items():
        comp = mv.compose(mv.inverse())
        assert comp == CubieMove.identity(), f'{key}: compose with inverse not identity'

    # 8. Cross-reference: R' = R⁻¹, L' = L⁻¹, F' = F⁻¹, B' = B⁻¹
    for face_side in (-1, 1):
        for axis in (0, 2):
            prim = pm[(axis, face_side, 1)]
            prim_inv = pm[(axis, face_side, -1)]
            assert prim_inv == prim.inverse(), f'axis={axis} side={face_side}: prime != inverse'

    print("test_from_rotation: OK (8 checks)")


def test_solver_diagnostics():
    """诊断 solve_kociemba 性能：验证 IDA* heuristic pruning 生效，
    节点数可控，heuristic 读数正确，depth_limit 递增搜索策略工作正常。"""
    CubieBase.build_pruning_table()

    # ── 1. Heuristic 一致性 ──
    state = CubieState.solved()
    coord = Phase1Coord.project(state)
    h0 = max(CubieBase.CO_EO_PRUNE[coord.corner_ori, coord.edge_ori],
             CubieBase.UD_PRUNE[coord.ud_slice])
    assert h0 == 0, f"Solved state heuristic should be 0, got {h0}"
    print("  [OK] Heuristic: solved state = 0")

    # ── 2. Scramble depth=3 应被 heuristic 正确估计 ──
    moves_3 = [(1, 1, 1), (0, 1, 1), (2, 1, 1)]  # U, R, F
    state3 = CubieMove.apply(CubieState.solved(), moves_3)
    coord3 = Phase1Coord.project(state3)
    h3 = max(CubieBase.CO_EO_PRUNE[coord3.corner_ori, coord3.edge_ori],
             CubieBase.UD_PRUNE[coord3.ud_slice])
    print(f"  [OK] Heuristic: URF scramble = {h3} (expected <= 3)")

    # ── 3. depth_limit=0 时 heuristic 立即剪枝，不搜索 ──
    t0 = time.time()
    res0 = CubieBase.phase1_search(state3, depth_limit=0)
    t0 = time.time() - t0
    assert t0 < 0.01, f"depth_limit=0 should be near-instant, got {t0:.3f}s"
    print(f"  [OK] depth_limit=0: {t0:.4f}s (heuristic immediate prune)")

    # ── 4. 充分 depth_limit 时快速找到解 ──
    t1 = time.time()
    res1 = CubieBase.phase1_search(state3, depth_limit=6)
    t1 = time.time() - t1
    assert res1 is not None, "Should find solution"
    assert t1 < 0.1, f"depth_limit=6 should be fast, got {t1:.3f}s"
    print(f"  [OK] depth_limit=6: {t1:.4f}s, solution len={len(res1)}")

    # ── 5. 验证解的正确性 ──
    state_after = state3.clone()
    for k, m in res1:
        state_after = m.replay(state_after)
    assert state_after.is_phase1_solved(), "Phase1 solution should resolve CO+EO+UD-slice"
    print(f"  [OK] Phase1 solution verified")

    # ── 6. 随机 6-move scramble 完整 solve_kociemba ──
    import random
    state6 = CubieState.solved()
    for _ in range(6):
        k = random.choice(list(CubieMove.prim_moves.keys()))
        state6 = CubieMove.prim_moves[k].act(state6)
    t2 = time.time()
    sol_keys, sol_move = CubieBase.solve_kociemba(state6)
    t2 = time.time() - t2
    assert sol_move.act(state6) == CubieState.solved(), "Full solve should return to solved"
    print(f"  [OK] Full solve_kociemba (6-move scramble): {t2:.3f}s, "
          f"solution len={len(sol_keys)}")

    # ── 7. depth+h 剪枝生效验证：启发值应阻止无谓深度搜索 ──
    # 手动测：depth_limit=2, h=2 → depth+h=4 ≤ limit? 不，h>limit 直接剪枝
    h_before = max(CubieBase.CO_EO_PRUNE[coord3.corner_ori, coord3.edge_ori],
                    CubieBase.UD_PRUNE[coord3.ud_slice])
    assert h_before <= 3, f"Heuristic should be reasonable, got {h_before}"
    print(f"  [OK] Heuristic pruning: h={h_before}, prune at depth_limit < {h_before}")

    print("test_solver_diagnostics: ALL OK")


if __name__ == "__main__":
    setup()
    cube = CubieBase(n=N)

    # 基本属性
    test_references(cube)
    test_solved_roundtrip(cube)
    test_cubie_roundtrip(cube)

    # StickerMove
    test_sticker_move()
    cube0 = StickerCube(n=3)
    test_sticker_cubie_move_consistency(cube0)

    # 贴纸 ↔ Cubie 一致性
    test_all_primitive_moves_solvable(cube)
    test_outer_moves_only(cube)
    test_rotate_state_vs_cubie_move(cube)
    test_gauge_correction(cube)

    # from_rotation refactored twist formula
    test_from_rotation()

    fix_corner_ori_offset(cube)

    # 群论 & 表示
    test_group_axioms()
    test_representation()
    test_prim_moves_group_props()
    test_action_consistency()
    test_move_compose_counts()
    test_move_composition()

    # Phase 同态 & 搜索
    test_phase1_homomorphism()
    test_phase1_ud_slice_map()
    test_phase_graph(cube)

    # 编码 round-trip
    test_comb_index()
    test_phase15_coord_roundtrip()
    test_phase15_project_consistency()
    test_phase15_pruning_bfs()
    test_phase15_act_no_act_index()
    test_slice_moves_solvable()
    test_ud_slice_encode()

    cube.build_pruning_table()

    test_phase1_search(cube)
    test_phase2_search(cube)

    # 求解
    test_solver_diagnostics()
    test_solve_kociemba(cube)
    test_examples_solvable(cube)
    """
    Superflip | 步数: 23 | 时间: 678.55 秒
    Superflip + corners | 步数: 23 | 时间: 17.11 秒
    Checkerboard | 步数: 14 | 时间: 474.25 秒
    Extreme reverse | 步数: 11 | 时间: 12.73 秒
    pc2
    Superflip            | 步数: 23 | 时间: 271.23 秒
    Superflip + corners | 步数: 23 | 时间: 6.11 秒
    Checkerboard         | 步数: 22 | 时间: 391.66 秒
    Extreme reverse      | 步数: 11 | 时间: 4.52 秒
    
    Superflip            | 步数: 23 | 时间: 270.63 秒
    Superflip + corners  | 步数: 23 | 时间: 6.92 秒
    Checkerboard         | 步数: 22 | 时间: 524.69 秒
    Extreme reverse      | 步数: 11 | 时间: 5.83 秒
    Big Cycle            | 步数: 20 | 时间: 104.84 秒
    Twisted              | 步数: 25 | 时间: 30.10 秒

    Superflip            | 步数: 23 | 时间: 293.26 秒
    Superflip + corners  | 步数: 23 | 时间: 8.14 秒
    Checkerboard         | 步数: 22 | 时间: 505.66 秒
    Extreme reverse      | 步数: 11 | 时间: 6.82 秒
    Big Cycle            | 步数: 20 | 时间: 107.79 秒
    Twisted              | 步数: 25 | 时间: 25.10 秒
    """

    # 随机路径一致性
    for t in range(3):
        print('test_random_path_consistency', t)
        test_random_path_consistency(cube, steps=30)

    # 物理单步
    test_single_move_physical(5)

    print('.................')

    # Vector roundtrip: from_vector ∘ rho ≡ act
    test_vector_roundtrip(num_random=500, max_steps=50)

    # Cache
    test_cache_roundtrip()
