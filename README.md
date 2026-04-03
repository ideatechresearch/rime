# RIME - 多领域数学建模与计算框架

RIME 是一个跨领域的 Python 数学建模与计算框架，涵盖魔方求解双重建模（sticker-level + cubie-level）、遗传学计算、环形数据结构、金字塔神经网络等多个方向的算法实现与可视化。

## 项目结构

```
rime/
├── base.py          # 基础工具类：属性代理、类属性缓存装饰器、链式调用
├── allele.py        # 遗传学：ABO血型系统建模、基因型/表现型计算、群体遗传学
├── circular.py      # 环形数据结构、矩阵面操作、金字塔投影
├── pyramid.py       # 金字塔神经网络、旋转对称矩阵变换
├── cube.py          # 魔方建模：NxN魔方贴纸级状态表示与基础求解
├── cubie.py         # 魔方建模：块级群论建模、Kociemba两阶段算法
├── cubieoperator.py # 群表示论：魔方群的表示分析、Bose-Mesner代数验证
├── cubieworld.py    # 慢动力学：Phase-1转移算符谱分析、慢流形、群谐函数
├── cubedraw.py      # 魔方可视化：Pygame 3D渲染与交互
├── cubedrawgl.py    # 魔方可视化：OpenGL 3D渲染
├── cubeplot.py      # 数据可视化：训练曲线、角向低秩分析
├── cubelearn.py     # 学习模型：CubeEnv环境、RankingCritic、Phase15Critic
├── dice.py          # 骰子特征分析、游戏触发器规则
├── body.py          # 遗传进化：人类血型遗传、新颖性搜索算法
├── helpers.py       # 辅助工具：DBSCAN聚类、K-means、余弦相似度、softmax
└── option.py        # 配置选项：全局参数管理
```

## 主要模块

### 1. 魔方贴纸级系统 ([cube.py](rime/cube.py))

完整的 NxN 魔方数学建模与求解系统。

**核心特性：**
- 支持 3x3 到 NxN 阶魔方的完整状态表示
- 贴纸级 (sticker-level) 状态表示，保证物理可达性
- 基于群论的旋转操作与状态验证
- 多种求解算法：BFS、IDA*、十字求解、中心块修正
- 标准魔方记法解析：`U`, `U'`, `U2`, `Rw`, `2Rw2` 等

**核心类：**
- `CubeBase`: 魔方数学基础类，包含坐标系统、几何变换、群论约束
- `RubiksCube`: 魔方实现类，提供旋转、打乱、求解接口

```python
from rime.cube import RubiksCube

# 创建 3x3 魔方
cube = RubiksCube(n=3)

# 应用标准记法动作
cube.apply_move("R")
cube.apply_move("U'")

# 打乱并求解
scramble = cube.scramble(20)
cube.apply(scramble)
solution = cube.solve()
```

**数学特性：**
- 坐标系：右手笛卡尔坐标系，+X→R, +Y→U, +Z→F
- 群论约束：角块朝向 (Z₃)、边块朝向 (Z₂)、排列奇偶性
- 状态嵌入：20 维向量 (8 角块 + 12 边块)
- 启发式函数：角块置换、中心块误差评估

### 2. 魔方块级系统 ([cubie.py](rime/cubie.py))

基于群论的魔方块级建模与 Kociemba 两阶段算法实现。

**核心特性：**
- 块级状态表示：`CubieState` 包含角块/边块的置换与朝向
- 群论建模：`G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`，状态空间约 4.3×10¹⁹
- 角块 S₈ / 边块 S₁₂ 是置换群，ℤ₃⁷ / ℤ₂¹¹ 是朝向群，半直积表示方向和排列的半独立性
- 两阶段算法：Phase-1 (方向修正 + 边块分层) → Phase-2 (排列求解)
- 剪枝表：CO×EO (2187×2048)、UD-slice (495)、角块/边块排列 (40320)
- 同态投影：`CubieMove → Phase1Move/Phase2Move` 坐标映射
- 双向转换：`CubieState ↔ StickerArray` 贴纸-块级状态互转

**魔方块级状态流程图：**
```scss
CubieState (角块+边块)
        │
        │ to_sticker() → (6,n,n) 贴纸数组
        │                   ↓
        │             from_cubie() (双向转换)
        │                   ↑
        │ to_cubie() ←─────┘
        │
        │ Phase1_project → Phase1Coord (CO, EO, UD-slice)
        │             ↑
        │       Phase1Move φ
        │
        │ Phase1_search (CO=0, EO=0, UD-slice separated)
        │
        │ Phase2_project → Phase2Coord (corner_perm, edge_perm, slice_perm)
        │             ↑
        │       Phase2Move φ
        │
        └─ Phase2_search → Solved
```

**核心类：**
- `CubieState`: 块级状态，包含 8 角块 + 12 边块的置换与朝向
- `CubieMove`: 群元素，支持半直积作用、合成、逆元
- `SlowDynamics`: Phase-1 子群慢动力学模型，谱分解与慢流形分析
- `Phase1Coord`: Phase-1 坐标 (CO, EO, UD-slice)
- `Phase2Coord`: Phase-2 坐标 (角块/边块排列)
- `CubieBase`: 继承自 `CubeBase`，提供两阶段搜索接口
  - `to_cubie(state)`: 从贴纸数组转块级状态
  - `from_cubie(cubie)`: 从块级状态转贴纸数组


```python
from rime.cubie import CubieState, CubieMove, CubieBase, SlowDynamics

# 创建已解状态
state = CubieState.solved()

# 应用基本动作
move = CubieMove.from_rotation(axis=0, layer=1, direction=1)  # R
state = move.act(state)

# 两阶段求解
CubieBase.build_pruning_table()  # 构建剪枝表（首次运行）
moves, cubie_move, final_state = CubieBase.solve_kociemba(state)
print(f"Solution: {moves}")

# 慢动力学分析
model = SlowDynamics.from_phase1_generators(n_generators=18)
vec = state.to_rho()  # 转换为 228 维表示
z = model.project(vec)  # 投影到 100 维慢子空间
z_t = model.evolve(z, T=10)  # 慢子空间演化
x_t = model.reconstruct(z_t)  # 重构回原空间
distance = model.heuristic(vec, x_t)  # 计算慢子空间距离
```

**算法细节：**
- **Phase-1 目标**: CO=0, EO=0, UD-slice 分离（边块回到中层）
- **Phase-2 目标**: 角块/边块排列还原（在 G₁ = ⟨U,D,R²,L²,F²,B²⟩ 子群内）
- **剪枝策略**: 联合 CO×EO + UD-slice 启发式，限制搜索深度
- **群同态**: `φ: CubieMove → Phase1Move/Phase2Move` 满足 `φ(m₁∘m₂) = φ(m₁)∘φ(m₂)`
- **慢动力学**: Phase-1 转移算符谱分层，5 个有理特征值层 (1, 7/9, 2/3, 5/9, 1/3)

**数学基础：**
- 状态空间: |G| ≈ 4.3×10¹⁹
- 群结构: `G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`
- Phase-1 子群: `G₁ = ⟨U,D,R²,L²,F²,B²⟩`, |G|/|G₁| ≈ 1.95×10¹⁰
- 剪枝表大小: CO_EO (4.5M) + UD (495) + CO (40320) + EDGE (40320) + SLICE (24)
- **慢流形谱分解**: 228 维表示 → 5 个特征值层
  - λ=1: 24 维（守恒量）
  - λ=7/9: 44 维（真实慢模态）
  - λ=2/3: 32 维（次慢）
  - λ=5/9: 96 维（中速）
  - λ=1/3: 32 维（快衰减）

**剪枝表构建**
- CO×EO (2187×2048) → Phase1 方向搜索
- UD-slice (495) → Phase1 中层边
- 角块排列 (40320)、非 slice 边排列 (40320) → Phase2 搜索
- slice 内部排列 (24) → Phase2 局部自由度

> 首次构建剪枝表约耗时 10-20 秒（视 CPU 而定），数据会缓存到 `data/` 目录


### 3. 魔方可视化 ([cubedraw.py](rime/cubedraw.py))

基于 Pygame 的 3D 魔方可视化与交互系统。

**功能：**
- 3D 立方体实时渲染
- 局部旋转动画
- 鼠标拖拽交互（视角旋转、层转动）
- 自动播放打乱序列
- 2D 展开图显示

```python
from rime.cube import RubiksCube
from rime.cubedraw import RubiksCubeDraw

cube = RubiksCube(n=5)
app = RubiksCubeDraw(cube)
app.run()
```

**控制说明：**
- 左键拖拽：旋转视角
- 右键拖拽：推断并执行层转动
- `A`: 切换自动旋转
- `Space`: 暂停/恢复动画
- `P`: 执行单次随机动作
- `S`: 生成并播放打乱序列
- `R`: 重置魔方

### 3.1 OpenGL 可视化 ([cubedrawgl.py](rime/cubedrawgl.py))

基于 OpenGL 的 3D 魔方可视化系统。

**功能：**
- 硬件加速的 3D 渲染
- 平滑的旋转动画
- 实时交互控制

### 3.2 数据可视化 ([cubeplot.py](rime/cubeplot.py))

魔方学习与谱分析的可视化工具。

**功能：**
- 训练曲线绘制
- 角向低秩结构可视化
- 谱分析结果展示
- 相位空间投影

### 4. 群表示论系统 ([cubieoperator.py](rime/cubieoperator.py))

魔方群的表示论分析与 Bose-Mesner 代数验证。

**核心特性：**
- 块检测与分解：检测群表示的不可约子空间
- 同构分解：检查 Cayley 图是否等价于 association scheme
- 不变子空间验证：验证特征值对应的子空间在群生成元作用下的不变性
- 交换子维数计算：分析群表示的自由度
- 慢流形分析：Phase-1.5 状态空间的低维结构分析

**核心函数：**
- `detect_blocks()`: 检测群表示的块结构
- `verify_association_scheme()`: 验证 Bose-Mesner 代数
- `check_invariant_subspaces()`: 检查特征子空间的不变性
- `commutant_dimension()`: 计算交换子代数维数

**数学基础：**
- 魔方群 228 维表示分解为多个不可约表示
- Phase-1 生成元形成 5 个不变子空间（5 个有理特征值层）
- 慢流形 (λ ≥ 2/3) 捕捉局部搜索结构，100 维慢子空间
- 谱分层: λ ∈ {1, 7/9, 2/3, 5/9, 1/3}，对应维度 {24, 44, 32, 96, 32}
- 慢谱嵌入对 10 步以内状态距离相关性显著 (r ≈ 0.5)
- 快层谱半径 ≈ 5/9，20–23 步内充分衰减
- 准等距距离: d(x,y) = ||V_slowᵀ(x-y)||，误差 1.0059 ± 0.0871

### 5. 慢动力学模型 ([cubieworld.py](rime/cubieworld.py))

Phase-1 转移算符谱分析与慢流形建模。

**核心特性：**
- 228 维群表示的谱分层：5 个有理特征值层 {1, 7/9, 2/3, 5/9, 1/3}
- 慢子空间（λ ≥ 2/3）：100 维准不变子空间
- 快子空间（λ < 2/3）：128 维，谱半径 ≈ 5/9
- 双时间尺度系统：边块混合 ≈ 5 步，角块混合 ≈ 20 步
- 群谐函数：前 8 个模式精确谐（误差=0）

**核心类：**
- `SlowDynamics`: 慢动力学模型，支持投影、演化、重构、启发式距离
- 谱分解：`A = (1/|S|)∑ρ(s)` → Σ λ_i E_i
- 维度分布：守恒 24，慢模 44，次慢 32，中速 96，快速 32

```python
from rime.cubieworld import SlowDynamics, CubieMove, CubieState

# 构建慢动力学模型
model = SlowDynamics.from_phase1_generators(n_generators=18)

# 状态投影到慢子空间
state = CubieState.solved()
vec = state.to_rho()  # 228 维表示
z = model.project(vec)  # (100,) 慢子空间

# 慢子空间演化
z_t = model.evolve(z, T=10)
x_t = model.reconstruct(z_t)

# 计算慢子空间距离（启发式）
distance = model.heuristic(vec_a, vec_b)
```

**算法意义：**
- 准等距距离启发式：d(x,y) = ||V_slowᵀ(x-y)||，误差 1.0059 ± 0.0871
- 可用于 A*/IDA* 搜索、生成慢距离 scramble、低秩模拟
- 有效动力学维度 ≈ 5–6，由 rank-6 attention operator 控制

### 6. 魔方学习模型 ([cubelearn.py](rime/cubelearn.py))

贴纸-群论双层世界模型与神经网络学习系统。

**核心架构：**
```
[ 贴纸世界 / 连续 / 感知 ]
        ↓ observables
[ 中观物理量 / 势能 ]
        ↓ 学习
[ 群论世界 / 离散 / 搜索 ]
```

**核心类：**
- `CubeEnv`: 连接贴纸世界和群论世界的环境
- `RankingCritic`: 动作排序神经网络
- `Phase15Critic`: Phase-1.5 阶段评估网络
- `AngularLowRank`: 角向低秩结构建模 (rank=5 可精确拟合)
- `StructuredMoveLayer`: 结构化群表示层 (W_left ⊙ e_move ⊙ W_right)
- `LieMoveRepresentation`: 李代数参数化群表示 (ρ(g) = exp(Σ α_k A_k))

**核心发现：**
- Phase-1.5 状态空间的角向模式高度可压缩，rank=5 可精确拟合
- 40 维 cubie 空间上，每个 move 是精确线性映射
- 慢流形捕捉到宏观难度，但对远距离状态区分能力下降
- 群表示在线性可学，move norm 稳定 (≈ 6.3)

### 8. 遗传学系统 ([allele.py](rime/allele.py))

完整的 ABO 血型系统遗传学建模。

**核心类：**
- `AlleleBase`: 遗传学基础工具，向量映射、量子态表示
- `ABOSystem`: ABO 血型系统定义
- `Allele`: 等位基因与基因型操作
- `BloodType`: 血型类，支持输血相容性检查

```python
from rime.allele import Allele, BloodType

# 基因型转表现型
phenotype = Allele.genotype_to_phenotype('A', 'O')  # 'A'

# 子代概率计算
prob = Allele.get_child_probability('A', 'B')
# {'A': 0.1875, 'AB': 0.5625, 'B': 0.1875, 'O': 0.0625}

# 输血相容性
compatible = Allele.is_compatible_phenotype('O', 'A')  # True
```

**数据结构：**
- 等位基因向量：`A=(1,0)`, `B=(0,1)`, `O=(0,0)`
- 抗原/抗体映射：自动从向量推导
- 遗传概率矩阵：支持基因型和表现型两层概率
- 群体频率：Hardy-Weinberg 平衡计算

### 9. 环形数据结构 ([circular.py](rime/circular.py))

支持动态游标、容量限制、持久化的环形数据结构。

**核心类：**
- `CircularBand`: 环形缓冲区，支持旋转、转置、镜像等操作
- `MatrixFace`: 矩阵面操作，旋转对称、分块投影

```python
from rime.circular import CircularBand

band = CircularBand(['A', 'B', 'C'], capacity=5)
band.append('D')
band.rotate(1)  # 循环移动
band.transpose(4)  # 块转置
```

**金字塔投影：**
- 多层环形数据投影到方阵
- 支持 4 象限旋转对称切分
- 可逆变换：band ↔ matrix ↔ 3d blocks

### 10. 金字塔神经网络 ([pyramid.py](rime/pyramid.py))

基于递增环和旋转对称结构的混合神经网络架构。

**核心类：**
- `PYRAMID`: 金字塔数据结构，多层 band 管理
- `PyramidNN`: Transformer + GNN 混合架构
- `NextBandHead`: 层间预测头

```python
from rime.pyramid import PYRAMID, PyramidNN

pyramid = PYRAMID(max_layers=9)
pyramid.build(genotypes_iter)

nn = PyramidNN(dim=2, hidden=48)
outputs = nn.forward_pyramid(encoded_bands)
```

**特性：**
- 层内环状传播（邻域聚合）
- 层间交叉注意力
- 支持 band 编码与嵌入学习

### 11. 骰子特征系统 ([dice.py](rime/dice.py))

三骰子游戏特征分析与触发器系统。

**功能：**
- 骰子特征提取：顺子、三同、质数等
- 游戏触发器：优先级排序的效果系统
- 批量特征计算与规则匹配

```python
from rime.dice import dice_feature_game

features = dice_feature_game((4, 4, 4))
# ['四之恶', '三相之力']
```

**触发器示例：**
| 特征 | 条件 | 效果 |
|------|------|------|
| 三相之力 | 三枚相同 | 启用三种塔系 |
| 极限呈现 | 总和 > 16 | 极限表现 |
| 保底 | 1,2,3 顺子 | 稳定输出 |

### 12. 进化算法 ([body.py](rime/body.py))

遗传进化与新颖性搜索算法实现。

**核心类：**
- `Human`: 人类遗传模型，支持血型继承
- `NoveltySearch`: 新颖性搜索算法
- `Individual`: 进化个体

### 13. 辅助工具 ([helpers.py](rime/helpers.py))

常用机器学习与数据处理工具。

**功能：**
- `dbscan()`: DBSCAN 聚类算法实现
- `kmeans()`: K-means 聚类算法（支持 kmeans++ 初始化）
- `cosine_similarity()`: 余弦相似度计算
- `softmax()`: Softmax 函数

```python
from rime.helpers import dbscan, kmeans, cosine_similarity, softmax

# DBSCAN 聚类
labels = dbscan(X, eps=0.5, min_samples=5)

# K-means 聚类
centroids, labels = kmeans(X, k=3, init="kmeans++")

# 余弦相似度
sim = cosine_similarity(vectors_a, vectors_b)

# Softmax
probs = softmax(logits)
```

### 14. 配置管理 ([option.py](rime/option.py))

全局参数配置与管理。

## 依赖项

```bash
pip install numpy pygame scipy torch scikit-learn joblib matplotlib pandas sympy PyOpenGL
```

- `numpy`: 数值计算、数组操作
- `pygame`: 可视化渲染
- `scipy`: 科学计算（距离计算、统计）
- `torch`: 深度学习框架（神经网络模型）
- `scikit-learn`: 机器学习工具（PCA、TSNE 等）
- `joblib`: 数据序列化（缓存剪枝表、数据集）
- `matplotlib`: 数据可视化
- `pandas`: 数据处理与分析
- `sympy`: 符号计算
- `PyOpenGL`: OpenGL 3D 渲染
- `openai`: AI 模型集成
- `frozenlist`: 不可变列表数据结构

## 安装

### 方式一：从 requirements.txt 安装

```bash
pip install -r requirements.txt
```

### 方式二：逐个安装依赖

```bash
pip install numpy>=2.1.3
pip install pandas>=2.2.3
pip install scipy>=1.16.0
pip install torch>=2.10.0
pip install matplotlib>=3.10.3
pip install scikit-learn
pip install pygame==2.6.1
pip install PyOpenGL==3.1.10
pip install joblib==1.5.3
pip install sympy==1.14.0
pip install openai==1.63.2
```

### 可选：从 pyproject.toml 安装

```bash
pip install -e .
```

## 快速开始

### 魔方贴纸级操作

```bash
python -m rime.cube
python -m rime.cubedraw
```

### 慢动力学分析

```python
from rime.cubieworld import SlowDynamics, CubieMove, CubieState

# 构建慢动力学模型
model = SlowDynamics.from_phase1_generators(n_generators=18)

# 分析两个状态的距离
state_a = CubieState.solved()
state_b = CubieMove.from_rotation(0, 1, 1).act(state_a)

# 投影到慢子空间
vec_a = state_a.to_rho()
vec_b = state_b.to_rho()
z_a = model.project(vec_a)  # (100,) 慢子空间
z_b = model.project(vec_b)

# 计算慢子空间距离（启发式）
slow_distance = model.heuristic(vec_a, vec_b)
```

### 魔方块级求解

```python
from rime.cubie import CubieState, CubieMove, CubieBase

# 构建剪枝表（首次运行，数据会保存到 data/ 目录）
CubieBase.build_pruning_table()

# 打乱魔方
state = CubieState.solved()
for _ in range(10):
    state = random.choice(list(CubieMove.phase1_moves().values())).apply(state)

# 两阶段求解
moves, cubie_move, final_state = CubieBase.solve_kociemba(state)
print(f"Solution moves: {len(moves)}")
print(f"Final state solved: {final_state == CubieState.solved()}")
# 检查最终状态
assert final_state.is_solved()
```

### 遗传学计算

```python
from rime.allele import Allele

# 打印系统信息
print(Allele.system())  # 'ABO'
print(Allele.alleles())  # ('A', 'B', 'O')

# 计算子代概率
prob = Allele.get_child_probability('A', 'B')
print(f'A x B → {prob}')
```

### 金字塔投影

```python
from rime.pyramid import PYRAMID
from rime.allele import Allele

genotypes_iter = Allele.genotype_iter_by_freq(1000, 360)
pyramid = PYRAMID(max_layers=9)
pyramid.build(genotypes_iter)

# 获取 19x19 矩阵投影
matrix = pyramid.to_matrix(fill_center_with=('O', 'O'))
```

### 群表示分析

```python
from rime.cubieoperator import detect_blocks, verify_association_scheme
from rime.cubieworld import SlowDynamics
from rime.cubie import CubieMove, CubieState
import numpy as np

# 获取 Phase-1 生成元
generators = list(CubieMove.phase1_moves().values())

# 构建 228 维表示
# U, w = build_group_representation(generators)  # 假设有此函数
# blocks = detect_blocks(generators, U)

# 验证 Bose-Mesner 代数
# success, message, details = verify_association_scheme(A_micro, generators)

# 慢动力学模型
model = SlowDynamics.from_phase1_generators(n_generators=18)
state_a = CubieState.solved()
state_b = CubieMove.from_rotation(0, 1, 1).act(state_a)

vec_a = state_a.to_rho()
vec_b = state_b.to_rho()
z_a = model.project(vec_a)  # (100,) 慢子空间
z_b = model.project(vec_b)

# 慢子空间距离（可用作启发式）
slow_distance = model.heuristic(vec_a, vec_b)

# 慢子空间演化
z_t = model.evolve(z_a, T=10)
x_t = model.reconstruct(z_t)
```

### 魔方世界模型

```python
from rime.cubelearn import CubeEnv, Phase15Critic, train_ranking_critic_15
from rime.cubie import CubieBase

# 创建环境
env = CubeEnv(n=3)
env.build_pruning_table()

# 训练 Phase-1.5 critic
critic = train_ranking_critic_15(num_epochs=10, batch_size=32)

# 生成 Phase-1.5 数据集
dataset = env.generate_phase15_dataset(max_depth=16, num_starting_points=100, num_samples=20000)
```

## 数学模型

### 魔方贴纸级状态空间

- **状态表示**: `(6, n, n)` 数组，贴纸级编码
- **群约束**:
  - 角块朝向和 ≡ 0 (mod 3)
  - 边块朝向和 ≡ 0 (mod 2)
  - 排列奇偶性一致

### 魔方块级群结构

- **状态表示**: `CubieState = (corners_perm, corners_ori, edges_perm, edges_ori)`
- **群结构**: `G = (S₈ × S₁₂) ⋉ (ℤ₃⁷ × ℤ₂¹¹)`
  - 角块: 8! × 3⁷ = 88,179,840 种状态（受朝向约束）
  - 边块: 12! × 2¹¹ = 495,466,560,000 种状态（受翻转约束）
  - 总状态: |G| ≈ 4.3 × 10¹⁹
- **Phase-1 投影**: `(CO, EO, UD-slice)`，维度 2187 × 2048 × 495
- **Phase-2 投影**: `(corner_perm, edge_perm, slice_perm)`，维度 40320 × 40320 × 24
- **Kociemba 算法**: 分两阶段降维求解
  - Phase-1: 恢复方向 + 边块分层（搜索空间 ~10¹⁰）
  - Phase-2: 排列还原（搜索空间 ~10⁸）
- **慢动力学谱分解**: Phase-1 转移算符 A = (1/|S|)∑ρ(s) 的谱结构
  - 5 个有理特征值层: λ ∈ {1, 7/9, 2/3, 5/9, 1/3}
  - 对应维度: {24, 44, 32, 96, 32}，总计 228 维
  - 慢子空间 (λ ≥ 2/3): 100 维，准不变子空间
  - 快子空间 (λ < 2/3): 128 维，谱半径 ≈ 5/9，20–23 步内衰减
  - 转移算符: A = Σ λ_i E_i，E_i 为特征投影子
  - 慢演化: z_t = λ^t z₀，T=100 相对误差 < 6×10⁻⁷

### ABO 血型遗传

- **Hardy-Weinberg 平衡**:
  - P(AA) = p², P(AO) = 2pr
  - P(BB) = q², P(BO) = 2qr
  - P(AB) = 2pq
  - P(OO) = r²

### 金字塔投影

- **网格大小**: `2n + 1` (n 层)
- **元素总数**: `(2n + 1)² = 4n(n+1) + 1`
- **环形编码**: 第 i 层 `8i` 个元素

### 魔方群表示论

- **表示空间**: Phase-1 生成元在 228 维复空间上的表示
- **谱分解**: 5 个不变子空间，对应 5 个不可约表示
  - 快速模态 (λ < 1/3): 32 维
  - 中速模态 (1/3 ≤ λ < 2/3): 96 维
  - 慢速模态 (λ ≥ 2/3): 100 维
  - 守恒模态 (λ = 1): 24 维（群谐函数）
- **慢流形**: 捕捉局部搜索结构，对 10 步内状态距离相关性 r ≈ 0.5
- **角向低秩**: Phase-1.5 状态空间的角向变化 rank=5 可精确拟合
- **转移算符**: A = (1/|S|)∑ρ(s)，谱分解 A = Σ λ_i E_i
- **有效动力学**: 约 5–6 个宏观时间尺度，rank-6 attention operator
- **群谐性质**: 前 8 个模式精确谐（误差=0），λ=7/9 层准谐（误差≈0.17±0.444）

### 贴纸-块级双向转换

- **to_cubie()**: 从 `(6, n, n)` 贴纸数组转 `CubieState` (40 维)
  - 角块方向: 通过 roll 循环匹配计算
  - 边块翻转: 通过 roll 循环匹配计算
  - 验证: 满足群论约束 (朝向和为 0 mod n)
- **from_cubie()**: 从 `CubieState` 转 `(6, n, n)` 贴纸数组
  - 基于已解状态的角块/边块坐标映射
  - 朝向通过 roll 操作应用
  - 保证贴纸数组满足物理约束

## 项目特点

1. **跨学科融合**: 涵盖群论、遗传学、金字塔神经网络、游戏触发器、表示论、谱动力学、机器学习
2. **双重建模**: 贴纸级与块级魔方建模，双向可逆转换
3. **严格的数学基础**: 基于群论的魔方建模、基于孟德尔定律的遗传计算、Bose-Mesner 代数验证
4. **高效算法**: Kociemba 两阶段算法、剪枝表优化、群同态投影、慢流形分析
5. **谱动力学**: 228 维群表示的 5 层谱分解，100 维慢子空间准等距距离启发式
6. **神经网络支持**: Ranking Critic、Phase-1.5 评估网络、李代数群表示、结构化 Move 层
7. **可视化支持**: Pygame 3D 渲染、OpenGL 硬件加速、实时交互、训练曲线可视化
8. **扩展性设计**: 支持自定义阶魔方、血型系统、金字塔层数、群表示维度
9. **辅助工具**: DBSCAN、K-means、余弦相似度、softmax 等常用机器学习工具
10. **数据持久化**: 支持剪枝表缓存、类属性缓存、数据集持久化

## 许可证

本项目为学术研究项目，仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 最佳实践

### 首次运行剪枝表构建

首次使用魔方块级求解时，需要构建剪枝表（约耗时 10-20 秒）：

```python
from rime.cubie import CubieBase

# 构建剪枝表（数据会缓存到 data/ 目录）
CubieBase.build_pruning_table()
```

### 数据缓存

项目支持多种数据缓存机制：

- **剪枝表**: 自动缓存到 `data/` 目录
- **类属性缓存**: 使用 `@class_cache` 装饰器
- **数据集持久化**: 使用 `joblib.dump/load` 序列化

### 可视化选择

- **Pygame** ([cubedraw.py](rime/cubedraw.py)): 适合交互式魔方操作
- **OpenGL** ([cubedrawgl.py](rime/cubedrawgl.py)): 适合高性能 3D 渲染
- **数据绘图** ([cubeplot.py](rime/cubeplot.py)): 适合分析和结果展示

### 性能优化建议

1. 使用块级求解（[cubie.py](rime/cubie.py)）而非贴纸级（[cube.py](rime/cube.py)）以获得更好的性能
2. 利用慢动力学模型的启发式距离加速搜索
3. 启用剪枝表缓存以避免重复计算
4. 对于大规模数据集，使用 `joblib` 进行并行处理
