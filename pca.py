import pickle
from scipy.sparse.linalg import svds
import numpy as np
import typing
from node2vec import Node2Vec
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

def load_pkl(file_path):
    """
    读取pkl文件
    Args:
        file_path: pkl文件路径
    Returns:
        pkl文件中的数据
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def svd_decompose(A, k: int):
    """
    对矩阵A进行SVD分解并降维
    Args:
        A: 输入矩阵
        k: 降维后的维度
    Returns:
        降维后的矩阵
    """
    # 检查k值是否合法
    if k <= 0:
        raise ValueError("k必须为正整数")
    if k > min(A.shape):
        raise ValueError("k不能大于矩阵的最小维度")
    
    # 进行SVD分解
    u, s, v = svds(A, k=k)
    
    # 计算降维结果
    X_2d = u @ np.diag(s) @ v
    return X_2d

def convert_to_G(A):
    """
    将稀疏矩阵转化为图
    Args:
        A: 输入矩阵
    Returns:
        X_2d:降维后的矩阵 
    """
    G = nx.from_scipy_sparse_array(A)

    # 使用Node2Vec
    node2vec = Node2Vec(G, dimensions=2, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # 获取2D嵌入
    X_2d = np.array([model.wv[str(i)] for i in range(A.shape[0])])
    return X_2d

if __name__=="__main__":
    file_path=r"/home/hyy/codes/CGKR/My_Diff/RecDiff-main/datasets/ciao/dataset.pkl"
    data=load_pkl(file_path)
    ciao_A=data["trust"].astype(np.float32)
    ciao_A = ciao_A.tocsr()
    k = 2
    X_2d=convert_to_G(ciao_A)
    
    # 计算向量
    vectors = np.zeros_like(X_2d)
    ciao_A_array = ciao_A.toarray()
    
    for i in range(ciao_A.shape[0]):
        neighbors = np.nonzero(ciao_A_array[i])[0]
        if len(neighbors) > 0:
            weights = ciao_A_array[i, neighbors]
            vectors[i] = np.sum(weights[:, np.newaxis] * (X_2d[neighbors] - X_2d[i]), axis=0) / weights.sum()
    
    # 向量场可视化
    vector_lengths = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / (vector_lengths[:, np.newaxis] + 1e-8)
    colors = cm.viridis(vector_lengths / vector_lengths.max())

    plt.figure(figsize=(12, 12))
    plt.quiver(
        X_2d[:, 0], X_2d[:, 1], 
        normalized_vectors[:, 0], normalized_vectors[:, 1], 
        vector_lengths,
        angles='xy', scale_units='xy', scale=0.5, 
        cmap='viridis', width=0.005, alpha=0.8
    )

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='red', s=10, alpha=0.6, fonlabel="Data Points")

    plt.colorbar(label="Vector Magnitude")
    plt.title("Improved Vector Field Visualization", fontsize=25)
    plt.xlabel("Dimension 1", fontsize=25)
    plt.ylabel("Dimension 2", fontsize=25)
    plt.legend(loc='upper right', fontsize=20)
    plt.grid(alpha=0.3)

    plt.savefig('vector_field_visualization_ciao.png', dpi=300, bbox_inches='tight')
    plt.show()