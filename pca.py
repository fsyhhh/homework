import numpy as np
def pca(X, k):  # k是要降到的维数
    # 求每一列均值
    a, b = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(b)])
    # 中心化
    norm_X = X - mean
    # 协方差矩阵
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # 特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(b)]
    # 排序
    eig_pairs.sort(reverse=True)
    # 找k个最大的特征向量
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # 降维
    data = np.dot(norm_X, np.transpose(feature))
    return data


X = np.array([[-1, 1, 1, 4], [-2, -1, 2, 5], [-3, -2, 3, 6], [1, 1, 2, 7], [2, 1, 1, 8], [3, 2, 2, 9]])

print(pca(X, 2))
