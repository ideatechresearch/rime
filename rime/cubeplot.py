import matplotlib.pyplot as plt
import numpy as np
import os
from rime.base import DATA_DIR


def draw_phase_graph(nodes, edges,
                     title: str = "Phase Schreier Graph (depth 2)",
                     figsize: tuple = (14, 12),
                     save_path: str = None
                     ):
    """
    可视化 Phase 的 Schreier 图

    nodes: {coord: depth}
    edges: [(src_coord, (axis, side, dir), dst_coord)]
    save_path: 如果提供，则保存为 PNG
    """
    import networkx as nx
    G = nx.DiGraph()  # 有向图

    # 添加节点（用 tuple key 作为节点名）
    for coord, depth in nodes.items():
        G.add_node(coord, depth=depth)

    max_d = max(nodes.values()) if nodes else 1
    depths = [data['depth'] for _, data in G.nodes(data=True)]  # [nodes[n] for n in G.nodes()]
    node_colors = [plt.cm.viridis(d / max_d) for d in depths]
    # degrees = dict(G.out_degree())
    node_sizes = [max(100, 1000 - 200 * d) for d in depths]  # [300 + 100 * degrees[n] for n in G.nodes()]
    # 边标签（动作）
    edge_labels = {}
    for src, label, dst in edges:
        G.add_edge(src, dst)
        edge_labels[(src, dst)] = f"{label}"  # label 是 (axis, side, dir)
    # 节点标签（简化显示，只显示 corner_coset 或完整 tuple）
    node_labels = {k: f"{k}" for k in G.nodes()}
    # 布局（spring 适合小图，kamada_kawai 更美观）
    pos = nx.kamada_kawai_layout(G, scale=2.5, center=(0, 0), dim=2)  # 或 nx.spring_layout(G, k=0.5, iterations=50)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="black")  # node_color="lightblue",
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=15)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)  # font_weight='bold'

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="blue")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", dpi=600, bbox_inches='tight', format='pdf')
        print(f"图已保存到 {base}.png 和 {base}.pdf")
    plt.show()
    return plt.gcf()


def draw_cycle_graph(perm, title="Cubie Permutation Cycles"):
    import networkx as nx
    G = nx.DiGraph()
    for i, j in enumerate(perm):
        G.add_edge(i, j)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    plt.title(title)
    plt.show()


def draw_phase15_heatmap(dist: np.ndarray, title="Phase 1.5 Distance Heatmap"):
    """
    dist: np.ndarray shape (3360,)，pure 距离表
    """
    import seaborn as sns
    N_SLICE = 24
    N_CORNER = 70
    N_PARITY = 2

    # 重塑成 (24, 70, 2)
    dist_3d = dist.reshape(N_SLICE, N_CORNER, N_PARITY)

    # 把 127 替换成 NaN，便于热图显示为灰色/白色
    dist_3d = np.where(dist_3d == 127, np.nan, dist_3d)

    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharey=True)

    for p in range(2):
        ax = axes[p]
        data = dist_3d[:, :, p]  # (24, 70)

        sns.heatmap(
            data,
            ax=ax,
            cmap='viridis_r',  # 低距离亮黄，高距离深紫
            cbar_kws={'label': 'Distance from solved'},
            annot=True,  # 显示数字（可选，节点少时好看）
            fmt=".0f",
            linewidths=0.3,
            linecolor='gray',
            square=True,
            mask=np.isnan(data),  # NaN 显示灰色
            vmin=0,
            vmax=np.nanmax(dist_3d)
        )

        ax.set_title(f"Parity = {p} (reachable: {np.sum(~np.isnan(data))})")
        ax.set_xlabel("Corner Coset (0 ~ 69)")
        ax.set_ylabel("Slice Perm (0 ~ 23)")

        # 加边框突出整体结构
        ax.add_patch(plt.Rectangle((0, 0), 70, 24, fill=False, edgecolor='black', lw=2))

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存高清版（推荐）
    plt.savefig(os.path.join(DATA_DIR, "phase15_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()


def draw_phase15_heatmap_parity_delta(dist: np.ndarray, title="Phase 1.5 Distance Heatmap (parity-delta)"):
    N_SLICE = 24
    N_CORNER = 70
    N_PARITY = 2

    # 重塑成 (24, 70, 2)
    dist_3d = dist.reshape(N_SLICE, N_CORNER, N_PARITY)
    print(np.bincount(dist))

    # 把 127 替换成 NaN，便于热图显示为灰色/白色
    dist_3d = np.where(dist_3d == 127, np.nan, dist_3d)
    p_delta = dist_3d[:, :, 1] - dist_3d[:, :, 0]  # (24, 70)

    dist_2d = np.where(p_delta == 0, np.nan, p_delta)
    plt.figure(figsize=(18, 8))
    plt.imshow(dist_2d, cmap='bwr', aspect='auto')
    plt.colorbar(label='Distance')
    plt.xlabel("Corner Coset (0 ~ 69)")
    plt.ylabel("Slice Perm (0~23)")
    plt.title(title)
    plt.savefig(os.path.join(DATA_DIR, "phase15_heatmap_parity_delta.png"), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_angular_lowrank(pred_np, true_np, mask_np, rank=5, save_prefix=os.path.join(DATA_DIR, "angular_lowrank")):
    """
    完整可视化 AngularLowRank 训练结果

    输出：
    1. 角向基函数 (SVD正交化)
    2. 距离层 × rank 模态贡献热图
    3. 原始幅度散点图
    4. 残差分布

    参数：
        pred_np -> model: 训练好的 AngularLowRank
        true_np -> M_torch: 原始观测矩阵 (未中心化)
        mask_np -> mask: 观测点 mask (bool tensor)
        rank: 显示前几个rank
        save_prefix: 如果不为None，将自动保存图片
    """
    n_layers, n_corners = pred_np.shape

    # ============================================================
    # 1️⃣ SVD 正交化角向基
    # ============================================================

    U_svd, S_svd, Vt_svd = np.linalg.svd(pred_np, full_matrices=False)

    V_modes = Vt_svd[:rank]  # (rank, n_corners)
    radial_modes = U_svd[:, :rank] * S_svd[:rank]

    explained_ratio = S_svd ** 2 / np.sum(S_svd ** 2)

    plt.figure(figsize=(12, 6))
    # for k in range(rank):
    #     plt.plot(V_modes[k], label=f"mode {k+1} ({explained_ratio[k]*100:.1f}%)", alpha=0.8)
    for k in range(rank):
        plt.scatter(np.arange(V_modes.shape[1]), V_modes[k],
                    label=f"mode {k + 1} ({explained_ratio[k] * 100:.1f}%)", alpha=0.6)

    plt.xlabel("Corner Coset Index")
    plt.ylabel("Angular Weight")
    plt.title(f"Learned Angular Basis (SVD orthogonalized, rank={rank})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_angular_basis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plt.figure(figsize=(12,6))
    # plt.imshow(V_modes, aspect='auto', cmap='bwr', interpolation='nearest')
    # plt.colorbar(label="Amplitude")
    # plt.xlabel("Corner Coset Index")
    # plt.ylabel("Mode")
    # plt.title("Angular Basis Heatmap")
    # plt.show()
    # ============================================================
    # 2️⃣ 距离层贡献热图
    # ============================================================

    plt.figure(figsize=(12, 6))
    im = plt.imshow(radial_modes[:, :rank], aspect='auto', cmap='coolwarm')
    plt.colorbar(im, label="Layer contribution")
    plt.xlabel("Angular Rank Mode")
    plt.ylabel("Distance Layer Index")
    plt.title("Radial Contribution to Angular Modes")
    plt.xticks(np.arange(rank), [f"r{k + 1}" for k in range(rank)])
    plt.yticks(np.arange(n_layers))
    if save_prefix:
        plt.savefig(f"{save_prefix}_radial_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 3️⃣ 原始幅度散点图
    # ============================================================

    true_vals = true_np[mask_np]  # observed_true
    pred_vals = pred_np[mask_np]  # observed_pred

    plt.figure(figsize=(8, 8))
    plt.scatter(true_vals, pred_vals, s=5, alpha=0.6)
    # pred_orig = model.U @ model.V.T  # + model.bias
    # plt.scatter(true_np, pred_orig.detach().numpy(), s=5, alpha=0.6)

    min_v = min(true_vals.min(), pred_vals.min())
    max_v = max(true_vals.max(), pred_vals.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1)

    plt.xlabel("True Observed")
    plt.ylabel("Predicted")
    plt.title("Fit on Observed Points (Original Scale)")
    plt.grid(True, alpha=0.3)

    if save_prefix:
        plt.savefig(f"{save_prefix}_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 4️⃣ 残差分布
    # ============================================================

    residual = pred_vals - true_vals

    plt.figure(figsize=(12, 6))
    plt.hist(residual, bins=50, density=True)
    plt.xlabel("Residual (pred - true)")
    plt.title("Residual Distribution (Observed Points)")
    plt.grid(True, alpha=0.3)

    if save_prefix:
        plt.savefig(f"{save_prefix}_residual.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 残差 vs True 值（检查是否随真值变化）
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 2)
    plt.scatter(true_vals, residual, s=10, alpha=0.6, edgecolor='none')
    plt.axhline(0, color='gray', ls='--', lw=1.5)
    plt.xlabel("True Observed")
    plt.ylabel("Residual (pred - true)")
    plt.title("Residual vs True Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_scatter_residual.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ============================================================
    # 误差统计
    # ============================================================
    rel_err = np.linalg.norm(residual) / np.linalg.norm(true_vals)
    print(f"\nRelative error (observed) ≈ {rel_err:.6f}")
    print("Explained variance by rank:")
    for k in range(rank):
        print(f"  mode {k + 1}: {explained_ratio[k] * 100:.2f}%")


def draw_coeo_pixel_full(prune: np.ndarray, title="CO-EO Prune Pixel Map"):
    """
    prune: (2187, 2048) 的距离表
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        prune,
        cmap='viridis_r',  # 黄低 → 紫高
        interpolation='nearest',  # 像素风格
        aspect='auto'
    )

    plt.colorbar(im, ax=ax, label='Distance to solved')
    ax.set_title(title)
    ax.set_xlabel("Edge Orientation (0 ~ 2047)")
    ax.set_ylabel("Corner Orientation (0 ~ 2186)")
    ax.grid(False)

    plt.tight_layout()

    # 保存高清
    plt.savefig(os.path.join(DATA_DIR, "co_eo_pixel_full.png"), dpi=300, bbox_inches='tight')
    plt.show()


def draw_coeo_prune(coeo_prune: np.ndarray, title="Corner-Edge Orientation Prune Table"):
    """
    coeo_prune: shape (2187, 2048)，你的 co_eo 距离表
    """
    import seaborn as sns
    # 计算每个 corner_ori 的平均 eo 距离
    avg_per_co = np.mean(coeo_prune, axis=1)  # (2187,)
    avg_per_eo = np.mean(coeo_prune, axis=0)  # (2048,)

    fig = plt.figure(figsize=(16, 10))

    # 子图 1: corner_ori 的平均距离曲线
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(avg_per_co, color='royalblue', linewidth=2)
    ax1.set_title("Average Distance per Corner Orientation")
    ax1.set_xlabel("Corner Orientation Index (0 ~ 2186)")
    ax1.set_ylabel("Avg EO Distance")
    ax1.grid(True, alpha=0.3)

    # 子图 2: edge_ori 的平均距离曲线
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(avg_per_eo, color='darkorange', linewidth=2)
    ax2.set_title("Average Distance per Edge Orientation")
    ax2.set_xlabel("Edge Orientation Index (0 ~ 2047)")
    ax2.set_ylabel("Avg CO Distance")
    ax2.grid(True, alpha=0.3)

    # 子图 3: 随机选 4 个 corner_ori 的 eo 距离热图
    ax3 = fig.add_subplot(2, 2, 3)
    sample_co = np.random.choice(2187, 4, replace=False)
    sample_data = coeo_prune[sample_co, :]
    sns.heatmap(sample_data, ax=ax3, cmap='viridis_r', cbar_kws={'label': 'Distance'})
    ax3.set_title(f"Sample CO slices (indices: {sample_co})")
    ax3.set_xlabel("Edge Orientation (0 ~ 2047)")
    ax3.set_ylabel("Sample Corner Ori")

    # 子图 4: 整体统计
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(coeo_prune.flatten(), bins=20, color='teal', edgecolor='black')
    ax4.set_title("Distance Distribution (all 2187×2048 values)")
    ax4.set_xlabel("Distance")
    ax4.set_ylabel("Frequency")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # 保存高清
    plt.savefig(os.path.join(DATA_DIR, "co_eo_prune_overview.png"), dpi=300, bbox_inches='tight')
    plt.show()


def draw_training_curves(losses, accuracies, loss_label='MSE', acc_label='Accuracy',
                         title="Training Curves (Loss & Accuracy)"):
    """
    绘制训练过程中的损失和准确率曲线（双Y轴）

    参数:
        losses: list or array, 每个epoch的损失值
        accuracies: list or array, 每个epoch的准确率
        loss_label: str, 损失曲线的图例标签
        acc_label: str, 准确率曲线的图例标签
    """
    epochs = range(1, len(losses) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 左轴：损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color, marker='o', linestyle='-', label=loss_label)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 右轴：准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracies, color=color, marker='s', linestyle='-', label=acc_label)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例（合并两个轴的图例）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)  # , loc='upper right'

    plt.title(title)
    fig.tight_layout()  # 防止标签重叠
    plt.savefig(os.path.join(DATA_DIR, f"{title}.png"), dpi=300, bbox_inches='tight')
    plt.show()


def draw_error_histogram(values, title="Error Distribution", xlabel="Error",
                         save_name=None, n_bins=100):
    """群谐函数误差等复数值的直方图，自动拆分实/虚部"""
    plt.figure(figsize=(10, 6))
    plt.hist(np.real(values), bins=n_bins, density=True, alpha=0.7,
             color='skyblue', edgecolor='black', label='Real part')
    plt.hist(np.imag(values), bins=n_bins, density=True, alpha=0.7,
             color='salmon', edgecolor='black', label='Imag part')
    plt.axvline(0, color='red', ls='--', label='ideal = 0')
    plt.xlabel(xlabel)
    plt.ylabel('density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_slow_coordinates(T_steps, Z, n_dims=3, title='Slow subspace evolution',
                          save_name=None):
    """慢坐标随时间步的衰减曲线"""
    plt.figure(figsize=(12, 8))
    for i in range(min(n_dims, Z.shape[1])):
        plt.plot(T_steps, Z[:, i], label=f'slow dim {i}')
    plt.xlabel('time step')
    plt.ylabel('slow coordinates')
    plt.title(title)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()


def draw_gram_matrix(G, title='Gram Matrix', xlabel='Mode index', ylabel='Mode index',
                     cmap='hot', save_name=None):
    """Gram 矩阵热图"""
    plt.figure(figsize=(12, 12))
    plt.imshow(np.real(G), cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Real part')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_annealing(norm_discrete, norm_continuous, Tf, title='分离时间尺度退火下的状态收敛',
                   save_name=None):
    """离散 / 连续退火范数收敛对比"""
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(norm_discrete)) * Tf, norm_discrete,
             label='离散退火 (每 Tf 增 β)')
    plt.plot(np.arange(len(norm_continuous)), norm_continuous,
             label='连续退火 β(t)')
    plt.xlabel('步数 t')
    plt.ylabel('状态范数 ||x_t||')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_coeo_distribution(prune: np.ndarray, title="CO-EO Prune Distance Distribution",
                           save_name=None):
    """CO-EO 剪枝距离分布直方图 + 累积分布"""
    flat = prune.flatten()
    flat = flat[flat < 127]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(flat, bins=12, color='cornflowerblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("Frequency")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    sorted_flat = np.sort(flat)
    cdf = np.arange(1, len(sorted_flat) + 1) / len(sorted_flat)
    ax2.plot(sorted_flat, cdf, color='darkred', linewidth=2, label='CDF')
    ax2.set_ylabel("Cumulative Probability")
    ax2.legend(loc='upper left')

    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_coeo_slice_heatmaps(prune: np.ndarray, num_samples=6,
                              title="Random CO Slices of EO Prune Table", save_name=None):
    """随机采样 corner_ori 行，展示 eo 距离热图"""
    import seaborn as sns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    sample_indices = np.random.choice(prune.shape[0], num_samples, replace=False)

    for i, co_idx in enumerate(sample_indices):
        row = prune[co_idx, :]
        row = np.where(row == 127, np.nan, row)

        sns.heatmap(
            row.reshape(1, -1),
            ax=axes[i],
            cmap='viridis_r',
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            linewidths=0
        )
        axes[i].set_title(f"CO index = {co_idx}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_phase15_parity_delta_analysis(dist: np.ndarray, save_prefix=None):
    """
    Phase 1.5 parity-delta 分析：条件均值/方差 + log-log 幂律检验

    dist: shape (3360,) pure 距离表
    """
    from scipy.stats import linregress

    N_SLICE, N_CORNER, N_PARITY = 24, 70, 2
    dist_3d = dist.reshape(N_SLICE, N_CORNER, N_PARITY)
    dist_3d = np.where(dist_3d == 127, np.nan, dist_3d)

    p_delta = dist_3d[:, :, 1] - dist_3d[:, :, 0]
    dist_flat = dist_3d[:, :, 0].flatten()
    delta_flat = p_delta.flatten()

    # 去掉 NaN
    valid = ~(np.isnan(dist_flat) | np.isnan(delta_flat))
    dist_flat = dist_flat[valid]
    delta_flat = delta_flat[valid]
    unique_d = np.sort(np.unique(dist_flat))

    means = np.array([np.mean(delta_flat[dist_flat == d]) for d in unique_d])
    vars_ = np.array([np.var(delta_flat[dist_flat == d]) for d in unique_d])

    # 1. 均值 & 方差
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    ax1.plot(unique_d, means, 'o-', color='royalblue', linewidth=1.5)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel("Distance (parity=0)")
    ax1.set_ylabel("Mean p_delta")
    ax1.set_title("Conditional Mean per Distance Layer")
    ax1.grid(True, alpha=0.3)

    ax2.plot(unique_d, vars_, 'o-', color='darkorange', linewidth=1.5)
    ax2.set_xlabel("Distance (parity=0)")
    ax2.set_ylabel("Variance of p_delta")
    ax2.set_title("Conditional Variance per Distance Layer")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Mean and Variance of Parity Delta vs Distance", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_mean_var.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 均值+方差 双Y轴
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(unique_d, means, 'o-', color='royalblue', label='Mean p_delta')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax1.set_xlabel("Distance (parity=0)")
    ax1.set_ylabel("Mean p_delta", color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(unique_d, vars_, 's-', color='darkorange', label='Variance')
    ax2.set_ylabel("Variance of p_delta", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_dual_axis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Log-log 幂律检验
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, label in zip(axs, [means, vars_], ["Mean", "Variance"]):
        pos = (unique_d > 0) & (data > 0)
        if pos.sum() > 5:
            log_d = np.log10(unique_d[pos])
            log_data = np.log10(data[pos])
            ax.loglog(unique_d[pos], data[pos], 'o-', label=label)
            slope, intercept, r_value, _, _ = linregress(log_d, log_data)
            ax.plot(unique_d[pos], 10 ** (intercept + slope * log_d), '--r',
                    label=f"slope={slope:.2f}, R²={r_value**2:.2f}")
            print(f"{label} log-log slope: {slope:.2f}, R²: {r_value**2:.2f}")
        ax.set_xlabel("Distance (log scale)")
        ax.set_ylabel(f"{label} (log scale)")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
    plt.suptitle("Log-Log: Check for Power-Law")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_loglog.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. parity delta 分布比例 per distance
    neg_ratio, zero_ratio, pos_ratio = [], [], []
    for d in unique_d:
        vals = delta_flat[dist_flat == d]
        neg_ratio.append(np.mean(vals == -1))
        zero_ratio.append(np.mean(vals == 0))
        pos_ratio.append(np.mean(vals == 1))

    plt.figure()
    plt.plot(unique_d, neg_ratio, label="-1")
    plt.plot(unique_d, zero_ratio, label="0")
    plt.plot(unique_d, pos_ratio, label="+1")
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Ratio")
    plt.title("Parity Delta Distribution per Distance")
    if save_prefix:
        plt.savefig(f"{save_prefix}_ratio.png", dpi=300, bbox_inches='tight')
    plt.show()


def draw_phase15_angular_svd(dist: np.ndarray, n_modes=5, save_prefix=None):
    """
    Phase 1.5 距离矩阵角向 SVD 分解可视化

    dist: shape (3360,) pure 距离表
    输出：角向主成分、模态贡献热图、奇异值谱、累计解释方差
    """
    N_SLICE, N_CORNER, N_PARITY = 24, 70, 2
    dist_3d = dist.reshape(N_SLICE, N_CORNER, N_PARITY)
    dist_3d = np.where(dist_3d == 127, np.nan, dist_3d)

    p_delta = dist_3d[:, :, 1] - dist_3d[:, :, 0]

    # 构造距离层 × corner 矩阵
    D = sorted(np.unique(dist_3d[:, :, 0][~np.isnan(dist_3d[:, :, 0])]))
    M_raw = np.zeros((len(D), N_CORNER))
    for i, d in enumerate(D):
        mask = dist_3d[:, :, 0] == d
        block = np.where(mask, p_delta, np.nan)
        M_raw[i] = np.nanmean(block, axis=(0, 1))

    # 去掉全 NaN 列
    valid_cols = ~np.isnan(M_raw).all(axis=0)
    M = M_raw[:, valid_cols]
    n_valid = M.shape[1]

    # 中心化
    row_means = np.nanmean(M, axis=1, keepdims=True)
    centered = M - row_means
    centered = np.nan_to_num(centered, nan=0.0)

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_ratio = S ** 2 / np.sum(S ** 2)

    # 1. 角向主成分
    n_show = min(n_modes, Vt.shape[0])
    plt.figure(figsize=(12, 7))
    for k in range(n_show):
        plt.plot(range(n_valid), Vt[k, :], label=f"PC{k+1} (σ={S[k]:.3f})", linewidth=1.5)
    plt.xlabel("Filtered Corner Index")
    plt.ylabel("Weight")
    plt.title("Angular Modes (clean SVD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_angular_modes.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 模态贡献热图
    n_rank = min(n_modes, len(S))
    modes_layers = U[:, :n_rank] * S[:n_rank]
    plt.figure(figsize=(10, 6))
    im = plt.imshow(modes_layers, aspect='auto', cmap='coolwarm')
    plt.colorbar(im, label='Mode amplitude (U*S)')
    plt.xlabel("Angular mode index (rank)")
    plt.ylabel("Radial layer index")
    plt.title("Layer Contribution to Angular Modes")
    plt.yticks(np.arange(len(D)))
    plt.xticks(np.arange(n_rank), [f"rank {k+1}" for k in range(n_rank)])
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_radial_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 奇异值谱
    plt.figure(figsize=(8, 4))
    plt.plot(S, marker='o')
    plt.title("Singular Value Spectrum")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_sv_spectrum.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 累计解释方差
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(S) + 1), np.cumsum(explained_ratio), marker='o')
    plt.xlabel("Rank mode")
    plt.ylabel("Cumulative explained variance")
    plt.title("Angular Mode Explained Variance")
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_cumvar.png", dpi=300, bbox_inches='tight')
    plt.show()


def draw_slow_manifold_3d(z_list: np.ndarray, depths: np.ndarray,
                          title="Slow Manifold Projection of Random Rubik States",
                          save_name=None):
    """
    慢坐标 3D 投影 (PCA → 3D scatter)

    z_list: (n_samples, dim) 慢坐标数组
    depths: (n_samples,) 对应的 scramble 深度
    """
    from sklearn.decomposition import PCA

    Z_real = np.real(z_list)
    pca = PCA(n_components=3)
    Z_3d = pca.fit_transform(Z_real)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Z_3d[:, 0], Z_3d[:, 1], Z_3d[:, 2],
                         c=depths, cmap='viridis', s=8, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter, label='Random Walk Length')
    plt.title(title)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


def draw_state_geometry_2d(X: np.ndarray, depths: np.ndarray, method='pca',
                           title=None, colorbar_label='Distance from solved',
                           save_name=None, **kwargs):
    """
    状态空间 2D 降维可视化 (PCA / t-SNE / UMAP)

    X: (n_samples, dim) 状态向量
    depths: (n_samples,) 对应深度
    method: 'pca', 'tsne', 'umap'
    """
    method = method.lower()
    if title is None:
        titles = {'pca': "Rubik's Cube State Geometry (PCA)",
                  'tsne': "Rubik's Cube State Geometry (t-SNE)",
                  'umap': "Rubik's Cube State Geometry (UMAP)"}
        title = titles.get(method, f"State Geometry ({method})")

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = kwargs.get('perplexity', 30)
        reducer = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1, verbose=1)
        X_2d = reducer.fit_transform(X)
    elif method == 'umap':
        import umap as umap_mod
        reducer = umap_mod.UMAP(n_components=2, random_state=42, n_jobs=-1, verbose=True)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    plt.figure(figsize=(16, 12))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=depths, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()
    return X_2d


def draw_state_geometry_3d(X: np.ndarray, depths: np.ndarray,
                           title="3D PCA of Rubik States", colorbar_label='Depth from solved',
                           save_name=None):
    """状态空间 3D PCA 可视化"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3, svd_solver='randomized', random_state=42)
    X_3d = pca.fit_transform(X)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"累计方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                         c=depths, cmap='viridis', s=3, alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter, label=colorbar_label)
    plt.title(title)
    if save_name:
        plt.savefig(os.path.join(DATA_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    from rime.cubie import CubieBase, CubieState, Phase1Coord, Phase2Coord, Phase0Coord

    # ── 1. Phase Schreier 图 ──
    for Coord, name in [(Phase0Coord, "Phase 0"), (Phase1Coord, "Phase 1"), (Phase2Coord, "Phase 2")]:
        for depth in [2, 3]:
            nodes, edges = CubieBase.build_phase_graph(Coord.solved(), max_depth=depth)
            draw_phase_graph(nodes, edges,
                             title=f"{name} Schreier Graph (depth {depth})",
                             save_path=os.path.join(DATA_DIR, f"{name} Schreier Graph_{depth}"))

    # ── 2. Phase 1.5 距离热图 ──
    phase15_dist = CubieBase.build_phase15_pruning()
    draw_phase15_heatmap(phase15_dist)
    draw_phase15_heatmap_parity_delta(phase15_dist)

    # ── 3. Phase 1.5 parity-delta 分析 ──
    draw_phase15_parity_delta_analysis(
        phase15_dist, save_prefix=os.path.join(DATA_DIR, "phase15_parity_delta"))

    # ── 4. Phase 1.5 角向 SVD ──
    draw_phase15_angular_svd(
        phase15_dist, save_prefix=os.path.join(DATA_DIR, "phase15_angular_svd"))

    # ── 5. CO-EO 剪枝表可视化 ──
    coeo = CubieBase.cubie_distance()
    if coeo is not None and coeo.ndim == 2:
        draw_coeo_pixel_full(coeo)
        draw_coeo_prune(coeo)
        draw_coeo_distribution(coeo)
        draw_coeo_slice_heatmaps(coeo)

    # ── 6. 状态空间降维可视化 ──
    dataset = CubieBase.generate_phase15_dataset(max_depth=10, num_starting_points=20, num_samples=5000)
    X = np.array([d[2].embedding() for d in dataset])
    depths = np.array([d[6] for d in dataset])

    # 6a. PCA 2D
    draw_state_geometry_2d(X, depths, method='pca',
                           save_name="phase15_pca_State Geometry.png")

    # 6b. PCA 3D
    draw_state_geometry_3d(X, depths, save_name="phase15_pca3d_State Geometry.png")

    # 6c. t-SNE
    draw_state_geometry_2d(X, depths, method='tsne', perplexity=30,
                           title="TSNE of Rubik States",
                           colorbar_label='Depth from Phase1 solved',
                           save_name="phase15_tsne_State Geometry.png")

    # 6d. 慢流形 3D 投影
    from rime.cubieworld import SlowDynamics
    model = SlowDynamics()
    model.load()
    z_solved = model.project(CubieState.solved().vec)
    z_list, d_list = [], []
    for _ in range(5000):
        d = np.random.randint(0, 40)
        state = CubieBase.generate_cubie(length=d)
        z_list.append(model.project(state.vec))
        d_list.append(d)
    Z = np.real(np.array(z_list) - z_solved)
    draw_slow_manifold_3d(Z, np.array(d_list),
                          save_name="slow_manifold_pca3d.png")
