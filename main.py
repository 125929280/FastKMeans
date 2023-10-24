import glob
import jieba.analyse
import pandas as pd
import numpy as np
import scipy
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


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


# 云资源用户数据集
csv_files_path = '/Users/zz/PycharmProjects/fuck-kmeans/rebate/*/*.csv'  # 匹配csv文件路径
csv_files_list = glob.glob(csv_files_path)  # 获取所有匹配到的文件路径
# headers = ['客户id', '分销关系关联时间', '产品id', '实例id', '实例首购时间', '支付时间', '原价', '应付金额', '实付金额',
#            '评级业绩', '消费账号数', '佣金比例', '佣金金额', '佣金金额']
df = pd.concat([pd.read_csv(file_path, low_memory=False) for file_path in csv_files_list], axis=0,
               ignore_index=True)
df = df.sample(1000)
df.columns = ['customer_id',
              'login',
              'customer_name',
              'relate_time',
              'order_id',
              'product_id',
              'product_name',
              'instance_id',
              'instance_purchase_time',
              'order_type',
              'duration',
              'pay_time',
              'price',
              'payable_amount',
              'payment',
              'project_id',
              'is_performance',
              'performance',
              'consumption_accounts',
              'no_rebate_reason',
              'contract_rebate_ratio',
              'rebate_ratio',
              'rebate_amount',
              'is_partner',
              'positive_order',
              'customer_type',
              'certification_type',
              'enroll_time',
              'history_consumption'
              ]
text = ''
# for header in ['customer_name', 'product_name', 'order_type', 'no_rebate_reason']:
#     for i in df[header]:
#         if type(i) == str:
#             text += i + '\n'
# for header in ['customer_name', 'product_name', 'order_type', 'no_rebate_reason']:
#     text += '\n'.join([i for i in df[header] if type(i) == str])
df['text'] = df['customer_name'].map(str) + "\n" + df['product_name'].map(str) + "\n" + df['order_type'].map(str) + (
    "\n" + df['no_rebate_reason'].map(str) if type(df['no_rebate_reason']) == str else '')
text += '\n'.join([i for i in df['text']])
print(len(text))
print(jieba.analyse.extract_tags(text, topK=20, withWeight=True))
# df = df.sample(10000)
df = df.fillna(method='ffill')

for header in ['price', 'payable_amount', 'payment']:
    df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))
l1 = df.groupby('customer_id')[['pay_time']].max().reset_index()
l2 = df.groupby('customer_id')[['instance_purchase_time']].min().reset_index()
l = pd.merge(l1, l2, on='customer_id', how='inner')
l['L'] = (pd.to_datetime(l['pay_time']) - pd.to_datetime(l['instance_purchase_time'])).dt.days

r = df.groupby(['customer_id'])['pay_time'].max().reset_index(name='R')
r['R'] = (pd.to_datetime(r['R']) - pd.to_datetime('1999-07-28')).dt.days

f = df.groupby(['customer_id', 'pay_time']).size().reset_index(name='F')
f = f.groupby(['customer_id'])[['F']].sum().reset_index()

m_sum = df.groupby(['customer_id'])['payment'].sum().reset_index(name='total_amount')

f_m_sum = pd.merge(f, m_sum, on='customer_id', how='inner')
f_m_sum['M'] = f_m_sum['total_amount'] / f_m_sum['F']

c = df.groupby(['customer_id'])['rebate_ratio'].mean().reset_index(name='C')

import sqlite3

con = sqlite3.connect(':memory:')
l.to_sql('l', con)
r.to_sql('r', con)
f.to_sql('f', con)
f_m_sum.to_sql('m', con)
c.to_sql('c', con)

lrfmc = pd.read_sql_query(
    'select     \
          l.customer_id     \
        , l.L     \
        , r.R     \
        , f.F     \
        , m.M     \
        , c.C     \
    from l     \
    inner join r     \
        on l.customer_id = r.customer_id     \
    inner join f     \
        on l.customer_id = f.customer_id     \
    inner join m     \
        on l.customer_id = m.customer_id     \
    inner join c     \
        on l.customer_id = c.customer_id     ',
    con
)


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
    probs = [p(get_cluster(labels, m_partition, n_cluster), get_cluster(labels, target_partition, i))
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


def LWEA(labels: np.array, k, theta):
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


lrfmc = lrfmc.astype(float)
lrfmc.columns = [i for i in range(0, lrfmc.shape[1])]
outliers, inliers = lof(lrfmc, k=10, threshold=0)

inliers = inliers.drop(['k distances', 'local outlier factor'], axis=1)
inliers = StandardScaler().fit_transform(inliers)

base_clusters = []
for _ in range(0, 10):
    k = np.random.randint(2, np.sqrt(len(inliers)))
    # k = 2
    result = KMeans(n_clusters=k).fit_predict(inliers)
    base_clusters.append(result)
base_clusters = np.array(base_clusters)
pred = LWEA(base_clusters, 4, 0.5)
# print("数据集大小: ", len(inliers))
# print("预测结果大小：", len(pred))
# print(pred)
print("轮廓系数：", metrics.silhouette_score(inliers, pred))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

tf_vectorizer = TfidfVectorizer()
# tf_vectorizer = TfidfVectorizer(ngram_range=(2,2)) #2元词袋
X = tf_vectorizer.fit_transform(df['text'])
# print(tf_vectorizer.get_feature_names_out())
print(X.shape)
data1 = {'word': tf_vectorizer.get_feature_names_out(),
         'tfidf': X.toarray().sum(axis=0).tolist()}
df1 = pd.DataFrame(data1).sort_values(by="tfidf", ascending=False, ignore_index=True)
print(df1.head(20))

n_topics = 10  # 分为10类
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100,
                                learning_method='batch',
                                learning_offset=100,
                                #                                 doc_topic_prior=0.1,
                                #                                 topic_word_prior=0.01,
                                random_state=0)
lda.fit(X)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    tword2 = []
    tword3 = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_pro = [str(round(topic[i], 3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  # (round(topic[i],3))
        tword.append(topic_w)
        tword2.append(topic_pro)
        print(" ".join(topic_w))
        print(" ".join(topic_pro))
        print(' ')
        word_pro = dict(zip(topic_w, topic_pro))
        tword3.append(word_pro)
    return tword3


n_top_words = 20
feature_names = tf_vectorizer.get_feature_names_out()
word_pro = print_top_words(lda, feature_names, n_top_words)

# 输出每篇文章对应主题
topics = lda.transform(X)
topic = np.argmax(topics, axis=1)
df['topic'] = topic
# df.to_excel("data_topic.xlsx",index=False)
print(topics.shape)
print(topics[0])

import random  # 定义随机生成颜色函数


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#" + ''.join([random.choice(colorArr) for i in range(6)])
    return color


[randomcolor() for i in range(3)]

from collections import Counter
from wordcloud import WordCloud
from matplotlib import colors, pyplot as plt


# from imageio import imread    #形状设置
# mask = imread('爱心.png')

def generate_wordcloud(tup):
    color_list = [randomcolor() for i in range(10)]  # 随机生成10个颜色
    wordcloud = WordCloud(background_color='white', font_path='SimHei.ttf',  # mask = mask, #形状设置
                          max_words=20, max_font_size=50, random_state=42,
                          colormap=colors.ListedColormap(color_list)  # 颜色
                          ).generate(str(tup))
    return wordcloud


dis_cols = 4  # 一行几个
dis_rows = 3
dis_wordnum = 20
plt.figure(figsize=(5 * dis_cols, 5 * dis_rows), dpi=128)
kind = len(df['topic'].unique())

for i in range(kind):
    ax = plt.subplot(dis_rows, dis_cols, i + 1)
    most10 = [(k, float(v)) for k, v in word_pro[i].items()][:dis_wordnum]  # 高频词
    ax.imshow(generate_wordcloud(most10), interpolation="bilinear")
    ax.axis('off')
    # ax.set_title("第{}类话题 前{}词汇".format(i, dis_wordnum), fontsize=30)
plt.tight_layout()
plt.show()
