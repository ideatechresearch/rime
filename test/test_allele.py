"""
Allele 等位基因系统 & BloodType 血型类实验

实验分组：
  1. 基础系统信息
  2. 基因型-表现型映射
  3. 量子态运算（张量积、纠缠）
  4. 频率分布（Hardy-Weinberg）
  5. 子代遗传分布
  6. 亲本可能表型
  7. BloodType 基础测试
  8. 输血相容性测试
  9. 遗传交叉测试
  10. Rh因子测试
  11. 概率映射表
  12. 输血兼容性映射
  13. 等位基因频率设置
  14. 向量映射表
  15. 缓存操作
  16. 常量重建

运行: python test/test_allele.py
"""

import itertools
from rime.allele import Allele, BloodType


# ── 1. 基础系统信息 ─────────────────────────────────────────────────────

def test_basic_system_info():
    """Allele 系统基本信息"""
    print(Allele.__doc__)
    print(Allele.system(), Allele.alleles())
    print('axes', Allele.get_axes_by_allele_vector(Allele.allele_vector_mapping()))


# ── 2. 基因型-表现型映射 ────────────────────────────────────────────────

def test_genotype_phenotype_mapping():
    """基因型<->表现型双向映射"""
    genotype_map = Allele.genotype_to_phenotype_mapping()
    print('genotype_to_phenotype', genotype_map)
    genotype_db = Allele.phenotype_to_genotypes_mapping()
    print('phenotype_to_genotypes', genotype_db)
    print('A', Allele.phenotype_to_genotypes('A'),
          Allele.allele_vector('O'),
          'AB:', Allele.genotype_to_phenotype('A', 'B'), Allele.genotype_to_phenotype('B', 'A'))


# ── 3. 量子态运算 ─────────────────────────────────────────────────────

def test_quantum_states():
    """张量积、纠缠态、状态转换"""
    print(Allele.vector_to_state((0, 0)))

    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('B'))
    print("Combined AB:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('B')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.genotype_state('A', 'B'))

    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('A'))
    print("Combined AA:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('A')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.genotype_state('A', 'A'))

    t1 = Allele.tensor_product(Allele.allele_state('A'), Allele.allele_state('O'))
    print("Combined AO:", Allele.combine_quantum(Allele.allele_vector('A'), Allele.allele_vector('O')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.original_states(t1),
          Allele.genotype_state('A', 'O'))

    t1 = Allele.tensor_product(Allele.allele_state('O'), Allele.allele_state('O'))
    print("Combined OO:", Allele.combine_quantum(Allele.allele_vector('O'), Allele.allele_vector('O')), t1,
          Allele.separate_product_state(t1), Allele.is_entangled(t1), Allele.original_states(t1),
          Allele.genotype_state('O', 'O'))

    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'B')))
    print(Allele.state_to_phenotype(Allele.genotype_state('A', 'O')))
    print(Allele.state_to_phenotype(Allele.genotype_state('O', 'O')))


# ── 4. 频率分布 ─────────────────────────────────────────────────────

def test_frequency_distribution():
    """Hardy-Weinberg 平衡下的基因型/表现型频率"""
    print('phenotype', Allele.generate_phenotype(), 'genotypes', Allele.genotype_iter(False))
    print('frq', Allele.genotype_freq())
    print(Allele.phenotype_freq())
    print('ab antigens', Allele.vector_to_antigens((1, 1)), 'antibodies',
          Allele.antigens_to_antibodies({'A'}), Allele.antigens_to_antibodies({'O'}),
          Allele.antigens_to_antibodies({'A', 'B'}))


# ── 5. 子代遗传分布 ─────────────────────────────────────────────────────

def test_child_distribution():
    """子代基因型/表现型分布"""
    a1 = Allele.generate_genotype()
    a2 = Allele.generate_genotype()
    p1 = Allele.allele_to_phenotype(*a1)
    p2 = Allele.allele_to_phenotype(*a2)
    print(a1, a2, 'then', Allele.child_genotype_distribution(a1, a2))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, True))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, False, False))
    print(p1, 'and', p2, 'then', Allele.child_phenotype_distribution(p1, p2, False, True))
    print(p1, 'and', p2, 'then', Allele.get_child_probability(p1, p2))


# ── 6. 亲本可能表型 ─────────────────────────────────────────────────────

def test_parent_phenotypes():
    """推断可能的亲本表型"""
    p1 = Allele.generate_phenotype()
    p2 = Allele.generate_phenotype()
    print(p1, 'c and p1', p2, 'then parent2', Allele.get_parent_phenotypes(p1, p2))
    print('parent_possible')
    print('A', Allele.get_parent_phenotypes('A'))
    print('O', Allele.get_parent_phenotypes('O'))
    print('AB', Allele.get_parent_phenotypes('AB'))


# ── 7. BloodType 基础测试 ───────────────────────────────────────────────

def test_bloodtype_basic():
    """BloodType 基础：创建、抗原、抗体"""
    bt_a = BloodType(alleles=('A', 'O'))
    bt_b = BloodType(phenotype='B')
    bt_ab = BloodType(alleles=('A', 'B'))
    bt_o = BloodType(phenotype='O')

    print(f"A型: {bt_a}, antigens={bt_a.antigens}, antibodies={bt_a.antibodies}")
    print(f"B型: {bt_b}, antigens={bt_b.antigens}, antibodies={bt_b.antibodies}")
    print(f"AB型: {bt_ab}, antigens={bt_ab.antigens}, antibodies={bt_ab.antibodies}")
    print(f"O型: {bt_o}, antigens={bt_o.antigens}, antibodies={bt_o.antibodies}")


# ── 8. 输血相容性测试 ─────────────────────────────────────────────────────

def test_transfusion():
    """输血相容性：can_donate_to"""
    bt_a = BloodType(alleles=('A', 'O'))
    bt_o = BloodType(phenotype='O')
    bt_ab = BloodType(alleles=('A', 'B'))
    bt_b = BloodType(phenotype='B')

    print("\n--- Transfusion Tests ---")
    print(f"O型能输给A型: {bt_o.can_donate_to(bt_a)}")  # True
    print(f"A型能输给O型: {bt_a.can_donate_to(bt_o)}")  # False
    print(f"AB型能输给B型: {bt_ab.can_donate_to(bt_b)}")  # False
    print(f"A型能输给AB型: {bt_a.can_donate_to(bt_ab)}")  # True


# ── 9. 遗传交叉测试 ─────────────────────────────────────────────────────

def test_inheritance():
    """BloodType 杂交：cross, child_distribution"""
    bt_a = BloodType(alleles=('A', 'O'))
    bt_b = BloodType(phenotype='B')
    bt_o = BloodType(phenotype='O')

    print("\n--- Inheritance Tests ---")
    child = bt_a.cross(bt_b)
    print(f"A型 x B型 -> 子代: {child}")
    print(f"A型 x B型 子代分布: {bt_a.child_distribution(bt_b)}")
    print(f"O型 x O型 子代分布: {bt_o.child_distribution(bt_o)}")


# ── 10. Rh因子测试 ─────────────────────────────────────────────────────

def test_rh_factor():
    """Rh阳性/阴性血型"""
    bt_rh_neg = BloodType(phenotype='A', rh_positive=False)
    bt_rh_pos = BloodType(phenotype='A', rh_positive=True)
    print(f"\nRh- A型: {bt_rh_neg}, antibodies={bt_rh_neg.antibodies}")
    print(f"Rh+ A型: {bt_rh_pos}, antibodies={bt_rh_pos.antibodies}")


# ── 11. 概率映射表 ─────────────────────────────────────────────────────

def test_probability_mappings():
    """子代概率映射表"""
    child_probability = {','.join(sorted(p)): Allele.get_child_probability(*p)
                         for p in itertools.product(Allele.phenotypes(), repeat=2)}
    print(child_probability)
    print('genotype_probs', Allele.genotype_probs_mapping())
    print('phenotype_probs', Allele.phenotype_probs_mapping())
    print('phenotype_probs_equal', Allele.phenotype_probs_equal_mapping())


# ── 12. 输血兼容性映射 ─────────────────────────────────────────────────────

def test_compatibility_mappings():
    """输血兼容性相关映射"""
    print('union', Allele.union_antigens(Allele.allele('A'), Allele.allele('O')))
    print('compatible O->A', Allele.is_compatible_phenotype('O', 'A'))
    print('genotype_transfusion', Allele.genotype_transfusion_mapping())
    print('allele_state', Allele.allele_state_mapping())


# ── 13. 等位基因频率设置 ─────────────────────────────────────────────────────

def test_allele_freq_setting():
    """设置等位基因频率"""
    Allele.set_allele_freq(freq={'A': 0.2, 'B': 0.1, 'O': 0.7})
    print(Allele.phenotype_freq(), Allele.genotype_freq(), Allele.allele_freq())
    print('AO:', Allele.genotype_to_phenotype(genotype='AO'))


# ── 14. 向量映射表 ─────────────────────────────────────────────────────

def test_vector_mappings():
    """向量<->抗原/基因型映射"""
    print('vector_to_antigen', Allele.vector_to_antigen_mapping())
    print(Allele.vector_to_antibody_mapping())
    print('genotype_to_vector_mapping', Allele.genotype_to_vector_mapping())
    print('phenotype_to_genotypes_mapping', Allele.phenotype_to_genotypes_mapping())
    print('phenotype_transfusion', Allele.phenotype_transfusion_mapping())
    print(Allele.phenotype_to_antigens('A'))
    print(Allele.phenotype_to_genotypes('A'))


# ── 15. 缓存操作 ─────────────────────────────────────────────────────

def test_cache_operations():
    """缓存的查看和重建"""
    print(Allele.generate_phenotype(size=30))
    print('generate_phenotype', Allele())
    print(list(Allele.genotype_combinations()))
    print(Allele._expressed_cache)
    print('vars', Allele.get_vars())
    print('cache', Allele.get_cache())


# ── 16. 常量重建 ─────────────────────────────────────────────────────

def test_rebuild_constants():
    """重建常量和不同轴系统"""
    print(dir(Allele.allele('B')), '\n', Allele.allele('O').__dict__)
    Allele.rebuild_constants()
    print(Allele.get_vars())
    print('repr', Allele.__repr__)

    # 使用不同轴重建
    Allele.rebuild_constants(axes=['X', 'Y', 'Z'], outer='')
    print(Allele.system(), Allele.alleles())
    print('axes', Allele.get_axes_by_allele_vector(Allele.allele_vector_mapping()))
    genotype_map = Allele.genotype_to_phenotype_mapping()
    print('genotype_to_phenotype', genotype_map)
    genotype_db = Allele.phenotype_to_genotypes_mapping()
    print('phenotype_to_genotypes', genotype_db)


# ── 主入口 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═══ 1. 基础系统信息 ═══")
    test_basic_system_info()

    print("\n═══ 2. 基因型-表现型映射 ═══")
    test_genotype_phenotype_mapping()

    print("\n═══ 3. 量子态运算 ═══")
    test_quantum_states()

    print("\n═══ 4. 频率分布 ═══")
    test_frequency_distribution()

    print("\n═══ 5. 子代遗传分布 ═══")
    test_child_distribution()

    print("\n═══ 6. 亲本可能表型 ═══")
    test_parent_phenotypes()

    print("\n═══ 7. BloodType 基础测试 ═══")
    test_bloodtype_basic()

    print("\n═══ 8. 输血相容性测试 ═══")
    test_transfusion()

    print("\n═══ 9. 遗传交叉测试 ═══")
    test_inheritance()

    print("\n═══ 10. Rh因子测试 ═══")
    test_rh_factor()

    print("\n═══ 11. 概率映射表 ═══")
    test_probability_mappings()

    print("\n═══ 12. 输血兼容性映射 ═══")
    test_compatibility_mappings()

    print("\n═══ 13. 等位基因频率设置 ═══")
    test_allele_freq_setting()

    print("\n═══ 14. 向量映射表 ═══")
    test_vector_mappings()

    print("\n═══ 15. 缓存操作 ═══")
    test_cache_operations()

    print("\n═══ 16. 常量重建 ═══")
    test_rebuild_constants()
