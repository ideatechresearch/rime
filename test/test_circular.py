"""
CircularBand 环形数据结构和 ABO 基因分型实验

实验分组：
  1. 基础操作（init / append / rotate / current）
  2. 动态缩放（expand / contract）
  3. 索引访问 & 切片
  4. 数据持久化（save / load）
  5. 容量限制（capacity）
  6. 完整功能演示
  7. 块操作（transpose / mirror / swap）
  8. 特殊字符 & 边界条件
  9. 导航模拟（浏览器历史）
  10. ABO 基因型 batch bands
  11. 方阵投影 & 编码解码

运行: python test/test_circular.py
"""

import os, itertools
from rime.circular import CircularBand
from rime.base import DATA_DIR


# ── 1. 基础操作 ─────────────────────────────────────────────────────

def test_basic_operations():
    """CircularBand 初始化、append、rotate、current"""
    band = CircularBand(["A", "B", "C"])
    print(band)  # [A] → B → C → [A]...

    band.append("D")
    print(band)  # A → [B] → C → D → [A]...

    band.rotate(2)
    print(band.current())  # D

    # 循环遍历
    print("Loop from current:")
    for item in band:
        print(item, end=" → ")
    print()


# ── 2. 动态缩放 ─────────────────────────────────────────────────────

def test_expand_contract():
    """expand 在指针后插入多个元素，contract 从指针处收缩"""
    band = CircularBand(["A", "B", "C"])
    band.append("D")
    band.rotate(1)
    print("expand 前:", band)
    band.expand(["X", "Y"])
    print("expand 后:", band)  # A → B → C → [D] → X → Y → [A]...

    band.contract(2)
    print("contract 2 后:", band)  # A → B → C → [D] → [A]...


# ── 3. 索引访问 & 切片 ──────────────────────────────────────────────

def test_index_and_slice():
    """正索引（从 cursor 环形）、负索引（从末尾）、切片"""
    band = CircularBand(["A", "B", "C", "D", "E"])
    band.rotate(2)
    print("初始 (cursor=2):", band)

    print(f"索引 0: {band[0]}")   # 当前元素 (C)
    print(f"索引 1: {band[1]}")   # 下一个元素 (D)
    print(f"索引 -1: {band[-1]}")  # 前一个元素 (B)

    print("band[:3]:", band[:3])    # [C, D, E]
    print("band[1:4]:", band[1:4])  # [D, E, A]


# ── 4. 数据持久化 ───────────────────────────────────────────────────

def test_persistence():
    """save / load 到 DATA_DIR"""
    band = CircularBand(["A", "B", "C", "D", "E"])
    save_path = os.path.join(DATA_DIR, "circular_data.pkl")
    band.save(save_path)
    loaded_band = CircularBand.load(save_path)
    print("加载后的数据:", loaded_band)


# ── 5. 容量限制 ─────────────────────────────────────────────────────

def test_capacity():
    """capacity 满时自动覆盖最近元素"""
    limited_band = CircularBand(["X", "Y", "Z"], capacity=3)
    print("初始状态:", limited_band)

    limited_band.append("A")
    print("添加 'A' 后:", limited_band)  # X → [A] → Z → [X]... (Max: 3)

    limited_band.expand(["B", "C"])
    print("扩展 ['B','C'] 后:", limited_band)  # B → C → [Z] → [B]... (Max: 3)


# ── 6. 完整功能演示 ─────────────────────────────────────────────────

def test_full_demo():
    """rotate / expand / to_list / 索引 / contract / current"""
    band = CircularBand(["Red", "Green", "Blue"], capacity=5)
    print("初始:", band)

    band.append("Yellow")
    print("添加 Yellow:", band)

    band.rotate(-1)
    print("左旋:", band)

    band.expand(["Cyan", "Magenta"])
    print("扩展 Cyan,Magenta:", band)

    print("转换为列表:", band.to_list())         # ['Magenta', 'Yellow', 'Green', 'Blue', 'Red']
    print("线性索引 [2]:", band[2])
    print("环形索引 [-1]:", band[-1])

    band.contract(2)
    print("收缩 2 个元素:", band)

    print("当前元素:", band.current())


# ── 7. 块操作 ────────────────────────────────────────────────────────

def test_block_operations():
    """transpose 块转置 / mirror 镜像 / swap 相邻交换"""
    data = CircularBand([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("原始数据:", data)  # [1] → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

    data.transpose(3)
    print("块转置后:", data)  # [1] → 4 → 7 → 2 → 5 → 8 → 3 → 6 → 9

    data.mirror()
    print("镜像反转:", data)  # 9 → 6 → 3 → 8 → 5 → 2 → 7 → 4 → [1] → [9]...

    data.swap()
    print("相邻交换:", data)  # [1] → 6 → 3 → 8 → 5 → 2 → 7 → 4 → 9 → [1]...


# ── 8. 特殊字符 & 边界条件 ─────────────────────────────────────────

def test_special_chars_and_bounds():
    """特殊字符（中文/emoji/\\xff）和空 band 边界"""
    band = CircularBand(["正常文本", "特殊\xff字符", "emoji😊"])
    print(band)

    # 边界条件
    empty = CircularBand()
    empty.remove()
    empty.contract(5)
    print("空 band 边界测试通过")

    # 指针稳定性：remove 后指针正确
    band2 = CircularBand(["X", "Y", "Z"])
    band2.rotate(1)
    band2.remove()
    print("当前元素:", band2.current())  # 指向 Z


# ── 9. 导航模拟 ─────────────────────────────────────────────────────

def test_navigation_simulation():
    """浏览器历史：append（前进）/ rotate -1（后退）"""
    history = CircularBand(capacity=50)
    history.append("homepage")
    history.append("about_page")
    history.append("contact_page")

    history.rotate(-1)
    print("返回上一页:", history.current())

    history.rotate(1)
    print("前进到下一页:", history.current())


# ── 10. ABO 基因型 batch bands ───────────────────────────────────────

def test_allele_bands():
    """按增量批次（8/16/24/...）构建多层 CircularBand"""
    from rime.allele import Allele
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    bands, total_processed = CircularBand.build_bands(genotypes_iter, max_batches=9, start_batch=8)

    for idx, band in enumerate(bands):
        size = (idx + 1) * 8
        print(f"Layer {idx + 1}: capacity={size}, filled={len(band)}")
        print(band.to_list(from_current=True)[:min(8, len(band))])

        matrix = band.to_matrix(block_size=4)
        for i, r in enumerate(matrix):
            cells = [str(x) if x is not None else '' for x in r]
            print(i, "  ".join([c for c in cells if c != '']))
        print("-" * 40)

    print("total processed:", total_processed)


# ── 11. 方阵投影 & 编码解码 ─────────────────────────────────────────

def test_square_projection_and_encoding():
    """to_square_projection → Allele.states_encode → states_decode"""
    from rime.allele import Allele
    genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
    bands, _ = CircularBand.build_bands(genotypes_iter, max_batches=9, start_batch=8)

    g = CircularBand.to_square_projection(bands, start_batch=8)
    print("投影方阵尺寸:", len(g))
    for i, b in enumerate(g):
        print(i, b)

    g[9][9] = ('O', 'O')
    m = []
    for b in g:
        m.extend(b)

    mapping = {g: i for i, g in enumerate(Allele.genotypes())}
    print(f"基因型数: {len(m)}, mapping 大小: {len(mapping)}")

    byte_data = Allele.states_encode(m, mapping)
    print(f"编码后大小: {len(byte_data)} 字节")

    m2 = Allele.states_decode(byte_data, len(m), mapping)
    print("解码后与原始一致:", m == m2)
    c_id = 9 * 19 + 9
    print(f"中心点 c_id={c_id}, value={m2[c_id]}")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═══ 1. 基础操作 ═══")
    test_basic_operations()

    print("\n═══ 2. 动态缩放 ═══")
    test_expand_contract()

    print("\n═══ 3. 索引访问 & 切片 ═══")
    test_index_and_slice()

    print("\n═══ 4. 数据持久化 ═══")
    test_persistence()

    print("\n═══ 5. 容量限制 ═══")
    test_capacity()

    print("\n═══ 6. 完整功能演示 ═══")
    test_full_demo()

    print("\n═══ 7. 块操作 ═══")
    test_block_operations()

    print("\n═══ 8. 特殊字符 & 边界条件 ═══")
    test_special_chars_and_bounds()

    print("\n═══ 9. 导航模拟 ═══")
    test_navigation_simulation()

    print("\n═══ 10. ABO 基因型 batch bands ═══")
    test_allele_bands()

    print("\n═══ 11. 方阵投影 & 编码解码 ═══")
    test_square_projection_and_encoding()
