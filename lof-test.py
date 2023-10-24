import glob
import time
import datetime
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
dataset_list = []
name_list = []
cluster_list = []
class_list = []

# # Iris
# iris = pd.read_csv('uci/iris.data', header=None)
# iris = iris.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# class_list.append(4)
# # class_list.append(iris[4])
# # iris = iris.drop(4, axis=1)
# dataset_list.append(iris)
# cluster_list.append(3)
# name_list.append('Iris')

# # Wine
# wine = pd.read_csv('uci/wine.data', header=None)
# class_list.append(0)
# # class_list.append(wine[0])
# # wine = wine.drop(0, axis=1)
# dataset_list.append(wine)
# cluster_list.append(3)
# wine.columns = [i for i in range(0, wine.shape[1])]
# name_list.append('Wine')
#
# # Breast Cancer
# breast = pd.read_csv('uci/wdbc.data', header=None)
# breast = breast.replace({'M': 0, 'B': 1})
# class_list.append(0)
# # class_list.append(breast[1])
# breast = breast.drop(0, axis=1)
# dataset_list.append(breast)
# cluster_list.append(2)
# breast.columns = [i for i in range(0, breast.shape[1])]
# name_list.append('Breast')
#
# # Optical Recognition of Handwritten Digits
# digits = pd.read_csv('uci/optdigits.tra', header=None)
# class_list.append(64)
# # class_list.append(digits[64])
# # digits = digits.drop(64, axis=1)
# dataset_list.append(digits)
# cluster_list.append(10)
# name_list.append('Digits')

# Blood Transfusion Service Center
transfusion = pd.read_csv('uci/transfusion.data', header=None)
class_list.append(4)
# class_list.append(transfusion['whetherhe/shedonatedbloodinMarch2007'])
# transfusion = transfusion.drop('whetherhe/shedonatedbloodinMarch2007', axis=1)
dataset_list.append(transfusion)
cluster_list.append(2)
name_list.append('Transfusion')

# # Arrhythmia
# arrhythmia = pd.read_csv('uci/arrhythmia.data', header=None)
# arrhythmia = arrhythmia.replace({'?': 0})
# class_list.append(279)
# # class_list.append(arrhythmia[279])
# # arrhythmia = arrhythmia.drop(279, axis=1)
# dataset_list.append(arrhythmia)
# cluster_list.append(16)
# name_list.append('Arrhythmia')
#
# # Congressional Voting Records
# votes = pd.read_csv('uci/house-votes-84.data', header=None)
# votes = votes.replace({'republican': 0, 'democrat': 1, 'y': 1, 'n': 0, '?': -1})
# class_list.append(0)
# # class_list.append(votes[0])
# # votes = votes.drop(0, axis=1)
# dataset_list.append(votes)
# cluster_list.append(2)
# # votes.columns = [i for i in range(0, votes.shape[1])]
# name_list.append('Votes')
#
# # Ionosphere
# ionosphere = pd.read_csv('uci/ionosphere.data', header=None)
# ionosphere = ionosphere.replace({'g': 0, 'b': 1})
# class_list.append(34)
# dataset_list.append(ionosphere)
# cluster_list.append(2)
# name_list.append('Ionosphere')
#
# # Mushroom
# mushroom = pd.read_csv('uci/agaricus-lepiota.data', header=None)
# mushroom[0] = mushroom[0].replace({'p': 'a', 'e':'b'})
# mushroom = mushroom.applymap(lambda x: ord(x) - 97)
# class_list.append(0)
# dataset_list.append(mushroom)
# cluster_list.append(2)
# name_list.append('Mushroom')

# # Covertype
# covertype = pd.read_csv('uci/covtype.data', header=None)
# class_list.append(54)
# dataset_list.append(covertype)
# cluster_list.append(7)
# name_list.append('Covtype')

# # 云资源用户数据集
# csv_files_path = './rebate/*/*.csv'  # 匹配csv文件路径
# csv_files_list = glob.glob(csv_files_path)  # 获取所有匹配到的文件路径
# headers = ['客户id', '分销关系关联时间', '产品id', '实例id', '实例首购时间', '支付时间', '原价', '应付金额', '实付金额',
#            '评级业绩', '消费账号数', '佣金比例', '佣金金额', '佣金金额']
# df = pd.concat([pd.read_csv(file_path, low_memory=False)[headers] for file_path in csv_files_list], axis=0,
#                ignore_index=True)
# df = df.sample(10000)
# # df.head()
# # print(df.shape)
# # print(df.dtype)
# # print(df.describe().T)
# df = df.fillna(method='ffill')
# # null_all = df.isnull().sum()
# # print(null_all)
#
# for header in ['原价', '应付金额', '实付金额', '评级业绩', '消费账号数']:
#     df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))
#
# load_time = [datetime.datetime.strptime(load_time.split()[0], '%Y-%m-%d') for load_time in
#              df['分销关系关联时间']]
# first_time = [datetime.datetime.strptime(first_time.split()[0], '%Y-%m-%d') for first_time in
#               df['实例首购时间']]
# pay_time = [datetime.datetime.strptime(pay_time.split()[0], '%Y-%m-%d') for pay_time in df['支付时间']]
# df['L'] = [(pay_time[i] - load_time[i]).days for i in range(0, len(pay_time))]
# df['R'] = [(pay_time[i] - first_time[i]).days for i in range(0, len(pay_time))]
# # df['F'] = [order_count[customer] for customer in df['客户id']]
# # df['M'] = [payment_sum[customer] for customer in df['客户id']]
# # df['C'] = df['佣金比例']
# # features = ['客户id', 'L', 'R', 'F', 'M', 'C']
# # df = df[features]
#
# l_dict = df.groupby('客户id')['L'].mean().to_dict()
# r_dict = df.groupby('客户id')['R'].mean().to_dict()
# f_dict = df['客户id'].value_counts().to_dict()
# m_dict = df.groupby('客户id')['实付金额'].sum().to_dict()
# c_dict = df.groupby('客户id')['佣金比例'].mean().to_dict()
# data = []
# for k, v in l_dict.items():
#     data.append([v, r_dict[k], f_dict[k], m_dict[k], c_dict[k]])
#
# df = pd.DataFrame(data, columns=['l', 'r', 'f', 'm', 'c'])
#
# dataset_list.append(df)
# cluster_list.append(4)
# name_list.append('云资源用户数据集')
#
# dataset_list = [df]
# cluster_list.append(4)

#
# x = [i for i in range(1, 11)]
# y = []
# for i in x:
#   model = KMeans(n_clusters=i)
#   model.fit(df)
#   y.append(model.inertia_)
# plt.plot(x, y)
# plt.show()

scaler = StandardScaler()
for idx, dataset in enumerate(dataset_list):
    dataset = dataset.astype(float)
    # values = [dataset.max(), dataset.min()]
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
    if name_list[idx] != '云资源用户数据集':
        df_true = df[class_list[idx]]
        df = df.drop(class_list[idx], axis=1)
        inliers_true = inliers[class_list[idx]]
        inliers = inliers.drop(class_list[idx], axis=1)


    print(name_list[idx])
    start_time = time.time()
    kmeans = KMeans(n_clusters=cluster_list[idx])
    df = scaler.fit_transform(df)
    kmeans.fit(df)
    dataset_pred = kmeans.predict(df)
    print("数据集大小: ", len(df))
    if name_list[idx] != '云资源用户数据集':
        print("ARI: ", metrics.adjusted_rand_score(df_true, dataset_pred))
    print("轮廓系数：", metrics.silhouette_score(df, dataset_pred, metric='euclidean'))
    print("sse: ", kmeans.inertia_)
    print("Time:", time.time() - start_time)

    inliers = inliers.drop(['k distances', 'local outlier factor'], axis=1)
    # print(inliers)
    start_time = time.time()
    inliers = scaler.fit_transform(inliers)
    kmeans.fit(inliers)
    inliers_pred = kmeans.predict(inliers)
    print("数据集大小: ", len(inliers))
    if name_list[idx] != '云资源用户数据集':
        print("ARI: ", metrics.adjusted_rand_score(inliers_true, inliers_pred))
    print("轮廓系数：", metrics.silhouette_score(inliers, inliers_pred))
    print("sse: ", kmeans.inertia_)
    print("Time:", time.time() - start_time)