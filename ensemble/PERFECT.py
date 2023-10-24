import numpy as np
import scipy.stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from datasets import *
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_cluster(labels: np.array, m, n) -> np.array:
    """
    Returns indices for cluster n in m-th partition
    :param labels: matrix of size N x M (labels[i, j] = label of i-th data point in j-th partition)
    :param n: number of cluster
    :param m: number of partition
    :return: indices for cluster n in m-th partition
    """
    partition = labels[m, :]
    return np.where(partition == n)[0]


def removal_get_cluster(labels: np.array, m, n, removal) -> np.array:
    """
    Returns indices for cluster n in m-th partition
    :param labels: matrix of size N x M (labels[i, j] = label of i-th data point in j-th partition)
    :param n: number of cluster
    :param m: number of partition
    :return: indices for cluster n in m-th partition
    """
    partition = labels[m, :]
    removal_partition = partition.copy()
    removal_partition[removal[0]] = removal_partition[removal[1]] = -1
    return np.where(removal_partition == n)[0]


def p(c1: np.array, c2: np.array) -> float:
    """
    Calculates the intersection of two clusters
    :param c1: indices of objects clustered as c1
    :param c2: indices of objects clustered as c1
    :return: ratio of shared objects in two clusters
    """
    if len(set(c1)) == 0:
        return 1
    return len(set(c1) & (set(c2))) / len(set(c1))


def entropy(lst):
    """
    Alias for scipy entropy calculation
    """
    # if sum(lst) == 0:
    #     return 0
    return scipy.stats.entropy(lst, base=2)


def partition_entropy(labels, m_partition, n_cluster, target_partition):
    """
    Calculates uncertainty (entropy) of given cluster w.r.t. the given target_partition
    :param m_partition:
    :param n_cluster:
    :param target_partition:
    :param labels:
    :return:
    """
    # probs = [p(get_cluster(labels, m_partition, n_cluster), get_cluster(labels, target_partition, i))
    #          for i in range(len(np.unique(labels[target_partition, :])))]
    probs = [p(cluster_belong[str(m_partition) + ',' + str(n_cluster)],
               cluster_belong[str(target_partition) + ',' + str(i)])
             for i in range(len(np.unique(labels[target_partition, :])))]
    return entropy(probs)


def removal_partition_entropy(labels, m_partition, n_cluster, target_partition, removal):
    # removal_target_partition = [i for i in labels[target_partition, :] if
    #                             i != labels[target_partition][removal[0]] and i != labels[target_partition][removal[1]]]
    # probs = [p(removal_get_cluster(labels, m_partition, n_cluster, removal),
    #            removal_get_cluster(labels, target_partition, i, removal))
    #          for i in range(len(np.unique(removal_target_partition)))]
    # return entropy(probs)
    probs = [p(removal_get_cluster(labels, m_partition, n_cluster, removal),
               removal_get_cluster(labels, target_partition, i, removal))
             for i in range(len(np.unique(labels[target_partition, :])))]
    return entropy(probs)


def cluster_uncertainty(labels, m_partition, n_cluster):
    """
    Calculates cluster uncertainty w.r.t the whole ensemble of partitions
    :param labels:
    :param m_partition:
    :param n_cluster:
    :return:
    """
    entropies = [partition_entropy(labels, m_partition, n_cluster, m)
                 for m in range(len(labels))]
    return sum(entropies)


def removal_cluster_uncertainty(labels, m_partition, n_cluster, removal):
    entropies = [removal_partition_entropy(labels, m_partition, n_cluster, m, removal)
                 for m in range(len(labels))]
    return sum(entropies)


def eci(labels, m_partition, n_cluster, removal, theta=0.5) -> np.float32:
    H = cluster_uncertainty(labels, m_partition, n_cluster)
    h = removal_cluster_uncertainty(labels, m_partition, n_cluster, removal)
    # removal_labels =
    x = np.exp((-H + h) / (theta * len(labels)))
    return x


def LWCA(labels: np.array, theta):
    N = labels.shape[1]
    ca = np.zeros((N, N))
    for m, partition in enumerate(tqdm(labels)):
        for i, l1 in enumerate(partition):
            for j, l2 in enumerate(partition):
                if l1 == l2:
                    # ca[i][j] += 1 * eci(labels, m, l1, [i, j], theta)
                    ca[i][j] += 1
    return 1 / len(labels) * ca


cluster_belong = {}


def LWEA(labels: np.array, k, theta):
    for i in range(len(labels)):
        klass = labels[i].max()
        for j in range(0, klass+1):
            cluster_belong[str(i) + ',' + str(j)] = get_cluster(labels, i, j)
    similarity_matrix = LWCA(labels, theta)
    X = np.ones(similarity_matrix.shape) - similarity_matrix

    # model = AgglomerativeClustering(n_clusters=k).fit(X)
    # return model.labels_
    S = similarity_matrix
    D = np.diag(np.sum(S, axis=1))
    # 计算拉普拉斯矩阵
    L = D - S

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(L)
    eigvals = np.abs(eigvals)
    eigvecs = np.abs(eigvecs)

    # 取出前k个特征向量
    idx = eigvals.argsort()[:k]
    U = eigvecs[:, idx]

    # 对U进行归一化处理
    norm = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / norm

    # 对U_norm进行k-means聚类
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(U_norm)


if __name__ == '__main__':
    # label_true = iris[4]
    # base_clusters = []
    # for k in range(2, 10):
    #     kmeans = KMeans(n_clusters=k, n_init=k)
    #     kmeans.fit(iris)
    #     base_clusters.append(kmeans.predict(iris))
    # base_clusters = np.array(base_clusters)
    #
    # # c00 = get_cluster(labels, m=0, n=0)
    # # c01 = get_cluster(labels, 0, 1)
    # # print(partition_entropy(labels, 0, n_cluster=0, target_partition=1))
    # # print(cluster_uncertainty(labels, 0, 1))
    # #
    # # for i in range(3):
    # #     print()
    # #     for j in range(3):
    # #         H = cluster_uncertainty(labels, i, j)
    # #         print(H, eci(labels, i, j))
    #
    # ca = LWCA(base_clusters)
    # # print("diag", np.diag(ca))
    # # print(ca)
    # # print(LWEA(labels, 3))
    # nmi_score = normalized_mutual_info_score(label_true, LWEA(base_clusters, 3), average_method='arithmetic')
    # print("nmi: ", nmi_score)
    # labels = np.array([
    #     [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 2],
    #     [0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    # ])

    # c00 = get_cluster(labels, m=0, n=0)
    # c01 = get_cluster(labels, 0, 1)
    # print(partition_entropy(labels, 0, n_cluster=0, target_partition=1))
    # print(cluster_uncertainty(labels, 0, 1))

    # for i in range(3):
    #     print()
    #     for j in range(3):
    #         H = cluster_uncertainty(labels, i, j)
    #         print(H, eci(labels, i, j))

    labels = np.array([
        [0, 0, 1, 1, 0, 2, 0],
        [0, 0, 0, 0, 1, 2, 2]
    ])
    # ca = LWCA(labels)
    # print("diag", np.diag(ca))
    # print(ca)
    # print(LWEA(labels, 3))
    # print(2 * entropy([2.0 / 3, 0, 1.0 / 3]))
    # print(2 * entropy([1.0 / 4, 0, 3.0 / 4]))
    # print(entropy([3.0 / 3, 1.0 / 3, 0]) + entropy([1, 0, 0]))
    # print(entropy([2.0 / 5, 2.0 / 5, 1.0/5]) + entropy([3.0/5, 2.0/5, 0]))
    # print(2*entropy([1.0 / 4,0, 3.0 / 4]))
    # print(entropy([1, 0, 0])*2)
    print(np.exp(-(entropy([2.0 / 5, 2.0 / 5, 1.0 / 5]) + entropy([3.0 / 5, 2.0 / 5, 0]) - entropy(
        [1.0 / 3, 1.0 / 3, 1.0 / 3]) - entropy([1.0 / 3, 2.0 / 3, 0])) / (0.5 * 3)))
    print(entropy([2.0 / 5, 2.0 / 5, 1.0 / 5]) + entropy([3.0 / 5, 2.0 / 5, 0]) - entropy(
        [1.0 / 3, 1.0 / 3, 1.0 / 3]) - entropy([1.0 / 3, 2.0 / 3, 0]))
    print()
    print(entropy([1.0 / 4, 1.0 / 4, 1.0 / 6]) + entropy([1.0 / 3, 1.0 / 2, 0]))
