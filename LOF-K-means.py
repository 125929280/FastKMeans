import glob
import time
import datetime
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
from sklearn import metrics
import warnings

from sklearn.preprocessing import StandardScaler

from ensemble.datasets import *

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
# dataset_list = []
# name_list = []
# cluster_list = []

# Iris
# iris = pd.read_csv('uci/iris.data', header=None)
# iris = iris.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# iris = pd.concat([iris[4], iris.drop(4, axis=1)], axis=1)
# # class_list.append(iris[4])
# # iris = iris.drop(4, axis=1)
# dataset_list.append(iris)
# cluster_list.append(3)
# name_list.append('Iris')
#
# # Wine
# wine = pd.read_csv('uci/wine.data', header=None)
# # class_list.append(wine[0])
# # wine = wine.drop(0, axis=1)
# dataset_list.append(wine)
# cluster_list.append(3)
# name_list.append('Wine')
#
# # Breast Cancer
# breast = pd.read_csv('uci/wdbc.data', header=None)
# breast = breast.replace({'M': 0, 'B': 1})
# breast = pd.concat([breast[1], breast.drop(1, axis=1)], axis=1)
# # class_list.append(breast[1])
# breast = breast.drop(0, axis=1)
# dataset_list.append(breast)
# cluster_list.append(2)
# name_list.append('Breast')
#
# # Optical Recognition of Handwritten Digits
# digits = pd.read_csv('uci/optdigits.tra', header=None)
# digits = pd.concat([digits[64], digits.drop(64, axis=1)], axis=1)
# # class_list.append(digits[64])
# # digits = digits.drop(64, axis=1)
# dataset_list.append(digits)
# cluster_list.append(10)
# name_list.append('Digits')
#
# # Blood Transfusion Service Center
# transfusion = pd.read_csv('uci/transfusion.data', header=None)
# transfusion = pd.concat([transfusion[4], transfusion.drop(4, axis=1)], axis=1)
# # class_list.append(transfusion['whetherhe/shedonatedbloodinMarch2007'])
# # transfusion = transfusion.drop('whetherhe/shedonatedbloodinMarch2007', axis=1)
# dataset_list.append(transfusion)
# cluster_list.append(2)
# name_list.append('Transfusion')
#
# # Arrhythmia
# arrhythmia = pd.read_csv('uci/arrhythmia.data', header=None)
# arrhythmia = arrhythmia.replace({'?': 0})
# arrhythmia = pd.concat([arrhythmia[279], arrhythmia.drop(279, axis=1)], axis=1)
# # class_list.append(arrhythmia[279])
# # arrhythmia = arrhythmia.drop(279, axis=1)
# dataset_list.append(arrhythmia)
# cluster_list.append(16)
# name_list.append('Arrhythmia')
#
# # Congressional Voting Records
# votes = pd.read_csv('uci/house-votes-84.data', header=None)
# votes = votes.replace({'republican': 0, 'democrat': 1, 'y': 1, 'n': 0, '?': 2})
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
# ionosphere = pd.concat([ionosphere[34], ionosphere.drop(34, axis=1)], axis=1)
# dataset_list.append(ionosphere)
# cluster_list.append(2)
# name_list.append('Ionosphere')
#
# # Mushroom
# mushroom = pd.read_csv('uci/agaricus-lepiota.data', header=None)
# mushroom[0] = mushroom[0].replace({'p': 'a'})
# mushroom = mushroom.applymap(lambda x: ord(x) - 97)
# dataset_list.append(mushroom)
# cluster_list.append(2)
# name_list.append('Mushroom')

# # Covertype
# covertype = pd.read_csv('uci/covtype.data', header=None)
# covertype = pd.concat([covertype[54], covertype.drop(54, axis=1)], axis=1)
# dataset_list.append(covertype)
# cluster_list.append(7)
# name_list.append('Covtype')

# # 云资源用户数据集
# csv_files_path = './rebate/*/*.csv'  # 匹配csv文件路径
# csv_files_list = glob.glob(csv_files_path)  # 获取所有匹配到的文件路径
# # headers = ['客户id', '分销关系关联时间', '产品id', '实例id', '实例首购时间', '支付时间', '原价', '应付金额', '实付金额',
# #            '评级业绩', '消费账号数', '佣金比例', '佣金金额', '佣金金额']
# df = pd.concat([pd.read_csv(file_path, low_memory=False) for file_path in csv_files_list], axis=0,
#                ignore_index=True)
# df.columns = ['customer_id',
#               'login',
#               'customer_name',
#               'relate_time',
#               'order_id',
#               'product_id',
#               'product_name',
#               'instance_id',
#               'instance_purchase_time',
#               'order_type',
#               'duration',
#               'pay_time',
#               'price',
#               'payable_amount',
#               'payment',
#               'project_id',
#               'is_performance',
#               'performance',
#               'consumption_accounts',
#               'no_rebate_reason',
#               'contract_rebate_ratio',
#               'rebate_ratio',
#               'rebate_amount',
#               'is_partner',
#               'positive_order',
#               'customer_type',
#               'certification_type',
#               'enroll_time',
#               'history_consumption'
#               ]
# headers = ['price', 'payable_amount', 'payment']
# df = df.sample(10000)
# # df.head()
# # print(df.shape)
# # print(df.dtype)
# # print(df.describe().T)
# df = df.fillna(method='ffill')
# # null_all = df.isnull().sum()
# # print(null_all)
#
# for header in ['price', 'payable_amount', 'payment']:
#     df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))
#
# # load_time = [datetime.datetime.strptime(load_time.split()[0], '%Y-%m-%d') for load_time in
# #              df['relate_time']]
# # first_time = [datetime.datetime.strptime(first_time.split()[0], '%Y-%m-%d') for first_time in
# #               df['instance_purchase_time']]
# # pay_time = [datetime.datetime.strptime(pay_time.split()[0], '%Y-%m-%d') for pay_time in df['pay_time']]
# # df['L'] = [(pay_time[i] - load_time[i]).days for i in range(0, len(pay_time))]
# # df['R'] = [(pay_time[i] - first_time[i]).days for i in range(0, len(pay_time))]
# # df['F'] = [order_count[customer] for customer in df['客户id']]
# # df['M'] = [payment_sum[customer] for customer in df['客户id']]
# # df['C'] = df['佣金比例']
# # features = ['客户id', 'L', 'R', 'F', 'M', 'C']
# # df = df[features]
#
# # l_dict = df.groupby('customer_id')['L'].mean().to_dict()
# # r_dict = df.groupby('customer_id')['R'].mean().to_dict()
# # f_dict = df['customer_id'].value_counts().to_dict()
# # m_dict = df.groupby('customer_id')['payment'].sum().to_dict()
# # c_dict = df.groupby('customer_id')['rebate_ratio'].mean().to_dict()
#
# l1 = df.groupby('customer_id')[['pay_time']].max().reset_index()
# l2 = df.groupby('customer_id')[['instance_purchase_time']].min().reset_index()
# l = pd.merge(l1, l2, on='customer_id', how='inner')
# l['L'] = (pd.to_datetime(l['pay_time']) - pd.to_datetime(l['instance_purchase_time'])).dt.days
# # print('l')
# # print(l)
#
# r = df.groupby(['customer_id'])['pay_time'].max().reset_index(name='R')
# r['R'] = (pd.to_datetime(r['R']) - pd.to_datetime('1999-07-28')).dt.days
#
# # print('r')
# # print(r)
# f = df.groupby(['customer_id', 'pay_time']).size().reset_index(name='F')
# f = f.groupby(['customer_id'])[['F']].sum().reset_index()
# # print('f')
# # print(f)
# m_sum = df.groupby(['customer_id'])['payment'].sum().reset_index(name='total_amount')
# # print('m_sum')
# # print(m_sum)
# f_m_sum = pd.merge(f, m_sum, on='customer_id', how='inner')
# f_m_sum['M'] = f_m_sum['total_amount'] / f_m_sum['F']
# # print('f_m_sum')
# # print(f_m_sum)
#
# c = df.groupby(['customer_id'])['rebate_ratio'].mean().reset_index(name='C')
#
# import sqlite3
#
# con = sqlite3.connect(':memory:')
# l.to_sql('l', con)
# r.to_sql('r', con)
# f.to_sql('f', con)
# f_m_sum.to_sql('m', con)
# c.to_sql('c', con)
#
# lrfmc = pd.read_sql_query(
#     'select     \
#         l.customer_id     \
#         , l.L     \
#         , r.R     \
#         , f.F     \
#         , m.M     \
#         , c.C     \
#     from l     \
#     inner join r     \
#         on l.customer_id = r.customer_id     \
#     inner join f     \
#         on l.customer_id = f.customer_id     \
#     inner join m     \
#         on l.customer_id = m.customer_id     \
#     inner join c     \
#         on l.customer_id = c.customer_id     ',
#     con
# )
# print(lrfmc)

# data = []
# for k, v in l_dict.items():
#     data.append([v, r_dict[k], f_dict[k], m_dict[k], c_dict[k]])
#
# df = pd.DataFrame(data, columns=['l', 'r', 'f', 'm', 'c'])

# dataset_list.append(lrfmc)
# cluster_list.append(4)
# name_list.append('云资源用户数据集')

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
    df.columns = [i for i in range(0, df.shape[1])]
    # print(df.describe())
    outliers, inliers = lof(df, k=10, threshold=0)
    if name_list[idx] != '云资源用户数据集':
        df_true = df[0]
        df = df.drop(0, axis=1)
        inliers_true = inliers[0]
        inliers = inliers.drop(0, axis=1)

    df = scaler.fit_transform(df)

    print(name_list[idx])
    start_time = time.time()
    kmeans = KMeans(n_clusters=cluster_list[idx])
    kmeans.fit(df)
    df_pred = kmeans.predict(df)
    print("数据集大小: ", len(df))
    if name_list[idx] != '云资源用户数据集':
        print("CA: ", metrics.accuracy_score(df_true, df_pred))
        print("NMI: ", normalized_mutual_info_score(df_true, df_pred, average_method='arithmetic'))
        # print("ARI: ", metrics.adjusted_rand_score(df_true, df_pred))
    print("轮廓系数：", metrics.silhouette_score(df, df_pred, metric='euclidean'))
    print("sse: ", kmeans.inertia_)
    print("Time:", time.time() - start_time)

    inliers = inliers.drop(['k distances', 'local outlier factor'], axis=1)
    inliers = scaler.fit_transform(inliers)
    # print(inliers)
    start_time = time.time()
    kmeans.fit(inliers)
    inliers_pred = kmeans.predict(inliers)
    print("数据集大小: ", len(inliers))
    if name_list[idx] != '云资源用户数据集':
        print("CA: ", metrics.accuracy_score(inliers_true, inliers_pred))
        print("NMI: ", normalized_mutual_info_score(inliers_true, inliers_pred, average_method='arithmetic'))
        # print("ARI: ", metrics.adjusted_rand_score(inliers_true, inliers_pred))
    print("轮廓系数：", metrics.silhouette_score(inliers, inliers_pred))
    print("sse: ", kmeans.inertia_)
    print("Time:", time.time() - start_time)
    print(len(inliers_pred))
