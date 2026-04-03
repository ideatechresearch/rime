import numpy as np
from scipy.linalg import sqrtm, logm


def dbscan(X, eps=0.5, min_samples=5):
    n = X.shape[0]
    labels = np.full(n, -1)  # -1 表示噪声
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # 预计算距离矩阵
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    def region_query(i):
        return np.where(dist_matrix[i] <= eps)[0]

    def expand_cluster(i, neighbors):
        nonlocal cluster_id
        labels[i] = cluster_id

        j = 0
        while j < len(neighbors):
            pt = neighbors[j]

            if not visited[pt]:
                visited[pt] = True
                pt_neighbors = region_query(pt)

                if len(pt_neighbors) >= min_samples:
                    neighbors = np.concatenate([neighbors, pt_neighbors])

            if labels[pt] == -1:
                labels[pt] = cluster_id

            j += 1

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1  # 噪声
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels


def kmeans(X, k, max_iter=100, tol=1e-4, init="kmeans++", seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # 1. 初始化中心（随机选k个点）
    if init == "kmeans++":
        centroids = []

        # 1. 随机选第一个中心
        idx = rng.integers(n)
        centroids.append(X[idx])

        for _ in range(1, k):
            # 2. 计算每个点到最近中心的距离平方
            dists = np.min(
                np.linalg.norm(X[:, None, :] - np.array(centroids)[None, :, :], axis=2) ** 2,
                axis=1
            )

            # 3. 按距离加权采样
            probs = dists / np.sum(dists)
            idx = rng.choice(n, p=probs)

            centroids.append(X[idx])

        centroids = np.array(centroids)

    else:  # "random":
        centroids = X[rng.choice(n, k, replace=False)]

    for _ in range(max_iter):
        # 2. 计算距离 (n, k)
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

        # 3. 分配标签
        labels = np.argmin(distances, axis=1)

        # 4. 更新中心
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # 5. 收敛判断
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels


# from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity(ndarr1, ndarr2):
    ndarr1 = np.atleast_2d(ndarr1)
    ndarr2 = np.atleast_2d(ndarr2)
    denominator = np.outer(np.linalg.norm(ndarr1, axis=1), np.linalg.norm(ndarr2, axis=1))
    dot_product = np.dot(ndarr1, ndarr2.T)  # np.einsum('ik,jk->ij', ndarr1, ndarr2)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(denominator != 0, dot_product / denominator, 0)
    return similarity

#from sklearn.metrics.pairwise import cosine_distances  
def cosine_distance(a, b):
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# from scipy.special import softmax
def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 对每行减去最大值
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fidelity(rho, sigma):
    """
    计算两个密度矩阵的保真度 F(ρ, σ) = [Tr √(√ρ σ √ρ)]²
    重叠程度，越接近1越相似
    """
    # 加小正则化防止数值问题

    sqrt_rho = sqrtm(rho)  # matrix_sqrt(rho)
    middle = sqrt_rho @ sigma @ sqrt_rho.conj().T
    sqrt_middle = sqrtm(middle)
    fid = np.real(np.trace(sqrt_middle)) ** 2
    return np.clip(fid, 0.0, 1.0)


def matrix_log(A, eps=1e-10):
    """Hermitian 矩阵的对数（仅适用于正半定矩阵）"""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)  # 避免 log(0)
    log_eigvals = np.log(eigvals)
    return eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T


def matrix_sqrt(A, tol=1e-10):
    """用特征值分解实现 Hermitian 矩阵的平方根（仅适用于正半定 Hermitian 矩阵）"""
    eigvals, eigvecs = np.linalg.eigh(A)  # 密度矩阵是 Hermitian 的
    # 数值稳定性处理：负的极小特征值置0
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eigvals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T


def quantum_cross_entropy(rho, sigma, eps=1e-10):
    """适合一个分布相对于另一个分布的预测误差，不对称"""
    sigma_reg = sigma + eps * np.eye(sigma.shape[0], dtype=complex)
    log_sigma = logm(sigma_reg)  # matrix_log
    cross_ent = -np.real(np.trace(rho @ log_sigma))
    return max(cross_ent, 0.0)  # 不对称


def rho_sigreg(rho, num_projections=32, lambda_reg=0.08):
    """
    强制 latent embedding 的随机投影接近各向同性高斯分布，从而防止 collapse（所有表示坍缩到同一个点或极端纯态）
    LeCun SIGReg 适配版：对 rho_b 施加轻量正则，防止过度纯态或过度混沌
    lambda_reg: 正则强度（建议 0.05~0.12）
    """
    dim = rho.shape[0]
    total = 0.0
    for _ in range(num_projections):
        # 随机投影方向（单位向量）
        v = np.random.randn(dim) + 1j * np.random.randn(dim)
        v /= np.linalg.norm(v) + 1e-12

        proj = np.real(np.vdot(v, rho @ v))  # 计算二次型 <v| rho_b |v> （实数）

        # 希望 proj 接近标准正态分布的统计特性（均值0，方差1）
        # 用简单的平方惩罚（接近 LeCun 的 sketched Gaussian 思想）
        total += (proj - 0.5) ** 2 + 0.1 * (proj ** 2 - 1.0) ** 2

    return lambda_reg * (total / num_projections)
