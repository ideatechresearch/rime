import numpy as np
import random, math, time
from rime.cube import CubeBase, ActionToken, StickerCube
from rime.cubie import CubieBase, CubieState, CubieMove, StickerMove
from rime.cubie import Phase1Coord, Phase1Action, Phase2Coord, Phase2Action, Phase15Coord, CubieExample
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
    # cube.build_phase15_pruning_by_idx()#Reachable: 2784/3360,mean:3.9569
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

    # 看一个 corner 的 3 个贴纸来源
    s_idx = cube.solved_idx.copy()
    s_idx1 = cube.rotate_state(s_idx, 0, 1, 1)
    corner = cube.corner_coords(cube.n)[1]
    print([s_idx1[f, r, c] for (f, r, c) in corner])


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


def test_phase15_move_consistency(cube):
    """Phase15 坐标 move act 一致性"""
    T = CubieMove.prim_moves[(0, -1, 2)]
    Tinv = T.inverse()

    i = 0
    s = cube.generate_cubie(20)
    idx = Phase15Coord.project(s).index
    for k, m in CubieMove.phase15_moves.items():
        s2, coord = m.act(s)
        idx2_true = coord.index
        idx2_fake = m.act_index(idx)
        if idx2_true != idx2_fake:
            print(f'{k},{idx2_true},{idx2_fake},{coord},{Phase15Coord.from_index(idx2_fake)}')
            i += 1
    print('!=', i)  # 0.10.12..18


def test_slice_moves_solvable():
    s = CubieState.solved()
    for k, m in CubieMove.slice_moves.items():
        s2 = m.act(s)
        print(f'{k},{s2.is_solvable()}')  # (0, 0, 2),(1, 0, 2) (2, 0, 2)True


def test_ud_slice_encode():
    for c in range(495):
        edges_perm = CubieState.decode_ud_slice(c)
        coord = CubieState.encode_ud_slice(edges_perm.tolist())
        assert coord == c, f'{c},{coord},{edges_perm}'


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

        print(CubieMove.basic_generators[i], k, mv.inverse().is_primitive())
        print(k, mv.compose(mv) == CubieMove.identity())

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
            assert phi.ud_slice_map[c] == out.ud_slice_coord()
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
    """Phase1 → Phase15 → Phase2 完整流程"""
    s0 = CubieState.solved()
    sampled_items = random.sample(list(CubieMove.prim_moves().items()), 9)
    m0 = CubieMove.identity()
    for x, m in sampled_items:
        print(x)
        m0 = m0.compose(m)
    print(m0)

    for _ in range(10):
        s0 = random.choice(list(CubieMove.phase1_moves().values())).replay(s0)

    moves_1 = CubieBase.phase1_search(s0, 15)
    phase1_state = s0.clone()
    phase11_state = s0.clone()
    if moves_1:
        phase1_state = CubieMove.apply(phase1_state, [x[0] for x in moves_1])
        for a, m in moves_1:
            phase11_state = m.replay(phase11_state)
        assert phase1_state == phase11_state
    else:
        print('no moves phase1')

    print(phase1_state.is_phase1_solved(), phase1_state.edges_perm)
    if not phase1_state.is_phase1_solved():
        print(phase1_state.ud_slice_coord())
        phase1_state = CubieBase.canonicalize_ud_slice(phase1_state)
    print(phase1_state.ud_slice_coord(), phase1_state.edges_perm)

    path15, _, cubie15 = CubieBase.solve_phase15(phase11_state, 8)
    if not cubie15.is_phase2_ready():
        print(path15, cubie15)
    coord_15 = Phase15Coord.project(cubie15)
    print(coord_15.observables(), coord_15)

    moves_2 = CubieBase.phase2_search(phase1_state, 20)
    phase2_state = phase1_state.clone()
    for a, m in moves_2:
        phase2_state = m.replay(phase2_state)

    _, phase22_state = CubieMove.act_moves(phase1_state, [x[1].cubie_move for x in moves_2])
    assert phase2_state == phase22_state

    coord_15 = Phase15Coord.project(phase2_state)
    print(coord_15.index, coord_15.heuristic(), coord_15)
    # 0 8.0 Phase15Coord(slice_perm=0, corner_coset=69, parity=0)
    print(coord_15.observables())  # [8. 0. 0. 8. 8.]
    print(phase2_state.ud_slice_coord())
    print(phase2_state)

    print(phase2_state.is_phase1_solved())
    assert phase2_state.corners_ori.sum() == 0
    assert phase2_state.edges_ori.sum() == 0
    print(phase2_state == CubieState.solved())
    print([a for a, _ in moves_1 + moves_2])
    print(len(moves_1 + moves_2))


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
        ("Superflip", CubieExample.superflip()),
        ("Superflip + corners", CubieExample.superflip_plus()),
        ("Checkerboard", CubieExample.checkerboard()),
        ("Extreme reverse", CubieExample.inversed()),
        ("Big Cycle", CubieExample.big_cycle()),
        ("Twisted", CubieExample.twisted()),
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
                print(cube.heuristic_corner_old(s), cube.heuristic_corner_perm(s),
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

if __name__ == "__main__":
    setup()
    cube = CubieBase(n=N)

    # 基本属性
    test_references(cube)
    test_solved_roundtrip(cube)

    # 编码 round-trip
    test_comb_index()
    test_phase15_move_consistency(cube)
    test_slice_moves_solvable()
    test_ud_slice_encode()

    # StickerMove
    test_sticker_move()
    cube0 = StickerCube(n=3)
    test_sticker_cubie_move_consistency(cube0)

    # 贴纸 ↔ Cubie 一致性
    test_all_primitive_moves_solvable(cube)
    test_outer_moves_only(cube)
    test_rotate_state_vs_cubie_move(cube)
    test_gauge_correction(cube)

    # 群论 & 表示
    test_group_axioms()
    test_representation()
    test_prim_moves_group_props()
    test_action_consistency()
    test_move_compose_counts()

    # Phase 同态 & 搜索
    test_phase1_homomorphism()
    test_phase1_ud_slice_map()
    test_phase_graph(cube)
    cube.build_pruning_table()
    test_phase1_search(cube)
    test_phase2_search(cube)

    # 求解
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

    # Cache
    test_cache_roundtrip()
