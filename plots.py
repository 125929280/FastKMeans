import time
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# lof
def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1, novelty=True)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -clf.decision_function(predict.iloc[:, :-1])
    return predict


def plot_lof(result, threshold):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > threshold].index,
                result[result['local outlier factor'] > threshold]['local outlier factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local outlier factor'] <= threshold].index,
                result[result['local outlier factor'] <= threshold]['local outlier factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(threshold, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()


def lof(data, predict=None, k=5, threshold=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, threshold)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor'] > threshold].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= threshold].sort_values(by='local outlier factor')
    return outliers, inliers


# load datasets
# dataset_list = [pd.DataFrame(datasets.load_iris().data, columns=datasets.load_iris().feature_names),
#                 pd.DataFrame(datasets.load_wine().data, columns=datasets.load_wine().feature_names),
#                 pd.DataFrame(datasets.load_breast_cancer().data, columns=datasets.load_breast_cancer().feature_names),
#                 pd.DataFrame(datasets.load_digits().data, columns=datasets.load_digits().feature_names)
#                 ]
dataset_list = []
name_list = []
cluster_list = []
class_list = []

# Iris
iris = pd.read_csv('uci/iris.data', header=None)
iris = iris.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
class_list.append(4)
# class_list.append(iris[4])
# iris = iris.drop(4, axis=1)
dataset_list.append(iris)
cluster_list.append(3)
name_list.append('Iris')

# lof = LocalOutlierFactor(n_neighbors = 10)
# error = lof.fit_predict(iris_df.values)
# iris_lof = pd.concat([iris_df, pd.DataFrame(error, columns=['error'])], axis = 1)
# iris_lof = iris_lof[iris_lof['error'] == 1]
# print(iris_lof)
# dataset_list = [arrhythmia]
scaler = StandardScaler()
shape = ["d", "^", "o"]


def draw(x_axis, y_axis, pred, cluster):
    for i in range(0, cluster):
        x = []
        y = []
        for j in range(0, len(pred)):
            if pred[j] == i:
                x.append(x_axis[j])
                y.append(y_axis[j])
        plt.scatter(x, y, marker=shape[i])
    plt.show()


for idx, dataset in enumerate(dataset_list):
    dataset = dataset.astype(float)
    values = [dataset.max(), dataset.min()]
    # print(column_min)
    # for noise_alpha in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    # noise_size = int(0.3 * len(dataset))
    # noise = []
    # for i in range(0, noise_size):
    #     noise.append([])
    #     for j in range(0, len(values[0])):
    #         # print(i, j)
    #         noise[i].append(values[np.random.randint(0,1)][j])
    # df = pd.concat([dataset, pd.DataFrame(noise, columns=dataset.columns)])

    df = dataset
    outliers, inliers = lof(df, k=10, threshold=0)
    df_true = df[class_list[idx]]
    df = df.drop(class_list[idx], axis=1)
    inliers_true = inliers[class_list[idx]]
    inliers = inliers.drop(class_list[idx], axis=1)

    # dataset = scaler.fit_transform(dataset)

    print(name_list[idx])
    start_time = time.time()
    kmeans = KMeans(n_clusters=cluster_list[idx], init='random')
    kmeans.fit(df)
    dataset_pred = kmeans.predict(df)
    print("数据集大小: ", len(df))
    print("ARI: ", metrics.adjusted_rand_score(df_true, dataset_pred))
    print("轮廓系数：", metrics.silhouette_score(df, dataset_pred, metric='euclidean'))
    print("Time:", time.time() - start_time)


    inliers = inliers.drop(['k distances', 'local outlier factor'], axis=1)
    # print(inliers)
    start_time = time.time()
    kmeans = KMeans(n_clusters=cluster_list[idx])
    kmeans.fit(inliers)
    inliers_pred = kmeans.predict(inliers)
    print("数据集大小: ", len(inliers))
    print("ARI: ", metrics.adjusted_rand_score(inliers_true, inliers_pred))
    print("轮廓系数：", metrics.silhouette_score(inliers, inliers_pred))
    print("Time:", time.time() - start_time)

    # draw(df[0].to_list(), df[2].to_list(), dataset_pred, 3)
    draw(inliers[0].to_list(), inliers[2].to_list(), inliers_pred, 3)

# iris = pd.concat([iris_lof, pd.DataFrame(pred, columns=['pred'])], axis = 1)
# print(iris)
# print("离群点数量：", np.sum(pred == -1))
# factor = lof.negative_outlier_factor_
# radius = (factor.max()-factor)/(factor.max()-factor.min())
# iris.plot(kind = "scatter",x= "sepal width (cm)",y = "petal width (cm)", c = "r",figsize = (10,6),label = "data")
# plt.scatter(iris["sepal width (cm)"], iris["petal width (cm)"], s = 800 * radius, edgecolors="k",facecolors="none", label="LOF score")
# plt.legend()
# # plt.grid()
# plt.show()
