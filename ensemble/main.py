from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score

from ensemble import Voting
from ensemble.ClusterEnsemble import cluster_ensembles
from cluster_method import *
from EAC import EAC
import LWEA
import PERFECT
from datasets import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 载入数据
    for i, df in enumerate(dataset_list):
        # for it in range(0, 10):
        df = df.sample(min(2000, len(df)))
        print(name_list[i])
        data_all = np.array(df)
        label_true = data_all[:, 0]
        data = data_all[:, 1:]

        # EAC
        clustering = kmeans_cluster  # 选择K-means聚类方法
        para_list = np.arange(2, 8, 1)  # K-means的超参列表

        result_dict = {}
        base_clusters = []
        # for k in para_list:
        #     result = KMeans(n_clusters=k).fit_predict(data)
        #     result_dict.setdefault(k, result)
        #     base_clusters.append(result)
        #

        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)
        for _ in range(0, 10):
            k = np.random.randint(2, np.sqrt(len(df)))
            # k = 3
            result = KMeans(n_clusters=k).fit_predict(data)
            result_dict.setdefault(k, result)
            base_clusters.append(result)

        base_clusters = np.array(base_clusters)

        label_pred = Voting.LWEA(base_clusters, cluster_list[i])
        print("Voting: ")
        print("CA: ", metrics.accuracy_score(label_true, label_pred))
        print("NMI: ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))

        label_pred = LWEA.LWEA(base_clusters, cluster_list[i])
        # print("lwea: ", normalized_mutual_info_score(label, label_pred, average_method='arithmetic'))
        # print("lwea: ：", metrics.silhouette_score(data, label_pred))
        print("LWEA: ")
        print("CA: ", metrics.accuracy_score(label_true, label_pred))
        print("NMI: ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))

        eac = EAC(clustering)

        label_pred = eac.ensemble_fit(data, result_dict,
                                  method='single',
                                  if_plot=0)
        # print(result)
        # print(label_true)
        print("EAC")
        print("CA: ", metrics.accuracy_score(label_true, label_pred))
        print("NMI: ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))
        # print("ARI: ", metrics.adjusted_rand_score(label_true, label_pred))
        # print("eac 轮廓系数：", metrics.silhouette_score(data, label_pred))
        #

        # hgpa mcla hbgf cspa nmf
        label_pred = cluster_ensembles(data, base_clusters, solver='all', verbose=True, label_true=label_true)
        # nmi_score = normalized_mutual_info_score(label, label_pred, average_method='arithmetic')
        # print("nmi: ：", metrics.silhouette_score(data, label_pred))
        # print("nmi: ", nmi_score)

        # lwea
        # ca = LWEA.LWCA(base_clusters)
        # print("diag", np.diag(ca))
        # print(ca)
        # print(LWEA(labels, 3))

        # print("ARI: ", metrics.adjusted_rand_score(label_true, label_pred))

        # perfect
        # ca = PERFECT.LWCA(base_clusters)
        # print("diag", np.diag(ca))
        # print(ca)
        # print(LWEA(labels, 3))

        # print("perfect: ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))
        label_pred = PERFECT.LWEA(base_clusters, cluster_list[i], 0.5)
        # print("perfect: ：", metrics.silhouette_score(data, label_pred))
        print("PERFECT: ")
        print("CA: ", metrics.accuracy_score(label_true, label_pred))
        print("NMI: ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))
        # print("ARI: ", metrics.adjusted_rand_score(label_true, label_pred))

        # for theta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]:
        #     label_pred = PERFECT.LWEA(base_clusters, cluster_list[i], theta)
        #     print("CA perfect(theta=" + str(theta) + "): ", metrics.accuracy_score(label_true, label_pred))
        #     print("NMI perfect(theta=" + str(theta) + "): ", normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))
        # label_pred = PERFECT.LWEA(base_clusters, cluster_list[i], 0.5)
        # print("CA perfect: ", metrics.accuracy_score(label_true, label_pred))
        # print("NMI perfect: ",normalized_mutual_info_score(label_true, label_pred, average_method='arithmetic'))
