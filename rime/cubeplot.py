import matplotlib.pyplot as plt
import numpy as np


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
    plt.savefig("data/phase15_heatmap.png", dpi=300, bbox_inches='tight')
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
    plt.savefig("data/phase15_heatmap_parity_delta.png", dpi=300, bbox_inches='tight')
    plt.show()


def visualize_angular_lowrank(pred_np, true_np, mask_np, rank=5, save_prefix="data/angular_lowrank"):
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
    plt.savefig("data/co_eo_pixel_full.png", dpi=300, bbox_inches='tight')
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
    plt.savefig("data/co_eo_prune_overview.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"data/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()
