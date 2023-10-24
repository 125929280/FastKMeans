import glob

import pandas as pd
dataset_list = []
name_list = []
cluster_list = []
#
# # Iris
# iris = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/iris.data', header=None)
# iris = iris.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# iris = pd.concat([iris[4], iris.drop(4, axis=1)], axis=1)
# # class_list.append(iris[4])
# # iris = iris.drop(4, axis=1)
# dataset_list.append(iris)
# name_list.append('Iris')
# cluster_list.append(3)
#
# # Wine
# wine = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/wine.data', header=None)
# # class_list.append(wine[0])
# # wine = wine.drop(0, axis=1)
# dataset_list.append(wine)
# name_list.append('Wine')
# cluster_list.append(3)

# # Breast Cancer
# breast = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/wdbc.data', header=None)
# breast = breast.replace({'M': 0, 'B': 1})
# # class_list.append(breast[1])
# breast = breast.drop(0, axis=1)
# dataset_list.append(breast)
# name_list.append('Breast')
# breast.columns = [i for i in range(0, breast.shape[1])]
# cluster_list.append(2)
#
# # Optical Recognition of Handwritten Digits
# digits = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/optdigits.tra', header=None)
# digits = pd.concat([digits[64], digits.drop(64, axis=1)], axis=1)
# # class_list.append(digits[64])
# # digits = digits.drop(64, axis=1)
# dataset_list.append(digits)
# name_list.append('Digits')
# cluster_list.append(10)
#
# # Blood Transfusion Service Center
# transfusion = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/transfusion.data', header=None)
# transfusion = pd.concat([transfusion[4], transfusion.drop(4, axis=1)], axis=1)
# # class_list.append(transfusion['whetherhe/shedonatedbloodinMarch2007'])
# # transfusion = transfusion.drop('whetherhe/shedonatedbloodinMarch2007', axis=1)
# dataset_list.append(transfusion)
# name_list.append('Transfusion')
# cluster_list.append(2)
#
# # Arrhythmia
# arrhythmia = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/arrhythmia.data', header=None)
# arrhythmia = arrhythmia.replace({'?': 0})
# arrhythmia = arrhythmia.astype(float)
# arrhythmia = pd.concat([arrhythmia[279], arrhythmia.drop(279, axis=1)], axis=1)
# # class_list.append(arrhythmia[279])
# # arrhythmia = arrhythmia.drop(279, axis=1)
# dataset_list.append(arrhythmia)
# name_list.append('Arrhythmia')
# cluster_list.append(16)
#
# # Congressional Voting Records
# votes = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/house-votes-84.data', header=None)
# votes = votes.replace({'republican': 0, 'democrat': 1, 'y': 1, 'n': 0, '?': -1})
# # class_list.append(votes[0])
# # votes = votes.drop(0, axis=1)
# dataset_list.append(votes)
# # votes.columns = [i for i in range(0, votes.shape[1])]
# name_list.append('Votes')
# cluster_list.append(2)
#
# # Ionosphere
# ionosphere = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/ionosphere.data', header=None)
# ionosphere = ionosphere.replace({'g': 0, 'b': 1})
# ionosphere = pd.concat([ionosphere[34], ionosphere.drop(34, axis=1)], axis=1)
# dataset_list.append(ionosphere)
# name_list.append('Ionosphere')
# cluster_list.append(2)
#
# # Mushroom
# mushroom = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/agaricus-lepiota.data', header=None)
# mushroom = pd.DataFrame.map(mushroom, lambda x: ord(x) - 97)
# # mushroom = mushroom.applymap(lambda x: ord(x) - 97)
# dataset_list.append(mushroom)
# name_list.append('Mushroom')
# cluster_list.append(2)
#
# # # Covertype
# # covertype = pd.read_csv('/Users/zz/PycharmProjects/fuck-kmeans/uci/covtype.data', header=None)
# # covertype = pd.concat([covertype[54], covertype.drop(54, axis=1)], axis=1)
# # dataset_list.append(covertype)
# # name_list.append('Covtype')
# # cluster_list.append(7)
# #
# # # # 云资源用户数据集
# # # csv_files_path = '/Users/zz/PycharmProjects/fuck-kmeans/rebate/*/*.csv'  # 匹配csv文件路径
# # # csv_files_list = glob.glob(csv_files_path)  # 获取所有匹配到的文件路径
# # # # headers = ['客户id', '分销关系关联时间', '产品id', '实例id', '实例首购时间', '支付时间', '原价', '应付金额', '实付金额',
# # # #            '评级业绩', '消费账号数', '佣金比例', '佣金金额', '佣金金额']
# # # df = pd.concat([pd.read_csv(file_path, low_memory=False) for file_path in csv_files_list], axis=0,
# # #                ignore_index=True)
# # # df.columns = ['customer_id',
# # #               'login',
# # #               'customer_name',
# # #               'relate_time',
# # #               'order_id',
# # #               'product_id',
# # #               'product_name',
# # #               'instance_id',
# # #               'instance_purchase_time',
# # #               'order_type',
# # #               'duration',
# # #               'pay_time',
# # #               'price',
# # #               'payable_amount',
# # #               'payment',
# # #               'project_id',
# # #               'is_performance',
# # #               'performance',
# # #               'consumption_accounts',
# # #               'no_rebate_reason',
# # #               'contract_rebate_ratio',
# # #               'rebate_ratio',
# # #               'rebate_amount',
# # #               'is_partner',
# # #               'positive_order',
# # #               'customer_type',
# # #               'certification_type',
# # #               'enroll_time',
# # #               'history_consumption'
# # #               ]
# # # headers = ['price', 'payable_amount', 'payment']
# # # # df = df.sample(10000)
# # # df = df.fillna(method='ffill')
# # #
# # # for header in ['price', 'payable_amount', 'payment']:
# # #     df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))
# # # l1 = df.groupby('customer_id')[['pay_time']].max().reset_index()
# # # l2 = df.groupby('customer_id')[['instance_purchase_time']].min().reset_index()
# # # l = pd.merge(l1, l2, on='customer_id', how='inner')
# # # l['L'] = (pd.to_datetime(l['pay_time']) - pd.to_datetime(l['instance_purchase_time'])).dt.days
# # #
# # # r = df.groupby(['customer_id'])['pay_time'].max().reset_index(name='R')
# # # r['R'] = (pd.to_datetime(r['R']) - pd.to_datetime('1999-07-28')).dt.days
# # #
# # # f = df.groupby(['customer_id', 'pay_time']).size().reset_index(name='F')
# # # f = f.groupby(['customer_id'])[['F']].sum().reset_index()
# # #
# # # m_sum = df.groupby(['customer_id'])['payment'].sum().reset_index(name='total_amount')
# # #
# # # f_m_sum = pd.merge(f, m_sum, on='customer_id', how='inner')
# # # f_m_sum['M'] = f_m_sum['total_amount'] / f_m_sum['F']
# # #
# # # c = df.groupby(['customer_id'])['rebate_ratio'].mean().reset_index(name='C')
# # #
# # # import sqlite3
# # #
# # # con = sqlite3.connect(':memory:')
# # # l.to_sql('l', con)
# # # r.to_sql('r', con)
# # # f.to_sql('f', con)
# # # f_m_sum.to_sql('m', con)
# # # c.to_sql('c', con)
# # #
# # # lrfmc = pd.read_sql_query(
# # #     'select     \
# # #           l.customer_id     \
# # #         , l.L     \
# # #         , r.R     \
# # #         , f.F     \
# # #         , m.M     \
# # #         , c.C     \
# # #     from l     \
# # #     inner join r     \
# # #         on l.customer_id = r.customer_id     \
# # #     inner join f     \
# # #         on l.customer_id = f.customer_id     \
# # #     inner join m     \
# # #         on l.customer_id = m.customer_id     \
# # #     inner join c     \
# # #         on l.customer_id = c.customer_id     ',
# # #     con
# # # )
# # #
# # # dataset_list.append(lrfmc)
# # # cluster_list.append(4)
# # # name_list.append('云资源用户数据集')
# #
# # for df in dataset_list:
# #     df.columns = [i for i in range(0, df.shape[1])]
# #
# # # lrfmc.columns = ['id', 'L', 'R', 'F', 'M', 'C']
# # # labels = [1, 2, 3, 4]
# # # l_bins = [0, 1100, 1600, 2100, 100000000]
# # # lrfmc['l_score'] = pd.cut(lrfmc['L'], bins=l_bins, labels=labels, right=False)
# # # r_bins = [0, 8100, 8300, 8500, 100000000]
# # # lrfmc['r_score'] = pd.cut(lrfmc['R'], bins=r_bins, labels=labels, right=False)
# # # f_bins = [0, 62, 292, 1289, 100000000]
# # # lrfmc['f_score'] = pd.cut(lrfmc['F'], bins=f_bins, labels=labels, right=False)
# # # m_bins = [0, 2, 15, 57, 100000000]
# # # lrfmc['m_score'] = pd.cut(lrfmc['M'], bins=m_bins, labels=labels, right=False)
# # # c_bins = [0, 0.1, 0.13, 0.14, 1]
# # # lrfmc['c_score'] = pd.cut(lrfmc['C'], bins=c_bins, labels=labels, right=False)
# # # print(lrfmc)
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from scipy.stats import norm
# # #
# # #
# # # def norm_comparision_plot(data, figsize=(12, 10), color="#099DD9",
# # #                           ax=None, surround=True, grid=True):
# # #     """
# # #         function: 传入 DataFrame 指定行，绘制其概率分布曲线与正态分布曲线(比较)
# # #         color: 默认为标准天蓝  #F79420:浅橙  ‘green’：直接绿色(透明度自动匹配)
# # #         ggplot 经典三原色：'#F77B72'：浅红, '#7885CB'：浅紫, '#4CB5AB'：浅绿
# # #         ax=None: 默认无需绘制子图的效果；  surround：sns.despine 的经典组合，
# # #                                              默认开启，需要显式关闭
# # #         grid：是否添加网格线，默认开启，需显式关闭
# # #         """
# # #     plt.figure(figsize=figsize)  # 设置图片大小
# # #     # fit=norm: 同等条件下的正态曲线(默认黑色线)；lw-line width 线宽
# # #     sns.distplot(data, fit=norm, color=color, kde_kws={"color": color, "lw": 3}, ax=ax)
# # #     (mu, sigma) = norm.fit(data)  # 求同等条件下正态分布的 mu 和 sigma
# # #
# # #     # 添加图例：使用格式化输入，loc='best' 表示自动将图例放到最合适的位置
# # #     plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'. \
# # #                format(mu, sigma)], loc='best')
# # #     plt.ylabel('Frequency')
# # #     plt.title("Distribution")
# # #     if surround:
# # #         # trim=True-隐藏上面跟右边的边框线，left=True-隐藏左边的边框线
# # #         # offset：偏移量，x 轴向下偏移，更加美观
# # #         sns.despine(trim=True, left=True, offset=10)
# # #     if grid:
# # #         plt.grid(True)  # 添加网格线
# # #     plt.show()
# # # print(lrfmc)
# # # for i in ['L', 'R', 'F', 'M', 'C']:
# # #     norm_comparision_plot(lrfmc[i], figsize=(8, 6))
# # #
# # #
# # # print(lrfmc['F'].value_counts(normalize=True, ascending=True))
# #
from scipy.io import arff
# rings, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/rings.arff')
# rings = pd.DataFrame(rings)
# rings = rings.astype(float)
# rings.columns = [i for i in range(0, rings.shape[1])]
# rings = pd.concat([rings[2], rings.drop(2, axis=1)], axis=1)
# dataset_list.append(rings)
# cluster_list.append(3)
# name_list.append('Rings')
#
# spiral, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/spiral.arff')
# spiral = pd.DataFrame(spiral)
# spiral = spiral.astype(float)
# spiral.columns = [i for i in range(0, spiral.shape[1])]
# spiral = pd.concat([spiral[2], spiral.drop(2, axis=1)], axis=1)
# dataset_list.append(spiral)
# cluster_list.append(3)
# name_list.append('spiral')
#
# flame, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/flame.arff')
# flame = pd.DataFrame(flame)
# flame = flame.astype(float)
# flame.columns = [i for i in range(0, flame.shape[1])]
# flame = pd.concat([flame[2], flame.drop(2, axis=1)], axis=1)
# dataset_list.append(flame)
# cluster_list.append(2)
# name_list.append('flame')
#
# blobs, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/blobs.arff')
# blobs = pd.DataFrame(blobs)
# blobs = blobs.astype(float)
# blobs.columns = [i for i in range(0, blobs.shape[1])]
# blobs = pd.concat([blobs[2], blobs.drop(2, axis=1)], axis=1)
# dataset_list.append(blobs)
# cluster_list.append(3)
# name_list.append('blobs')
#
# # banana, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/banana.arff')
# # banana = pd.DataFrame(banana)
# # banana['class'] = banana['class'].apply(lambda x : str(x).split()[1][0])
# # banana = banana.astype(float)
# # banana.columns = [i for i in range(0, banana.shape[1])]
# # banana = pd.concat([banana[2], banana.drop(2, axis=1)], axis=1)
# # dataset_list.append(banana)
# # cluster_list.append(2)
# # name_list.append('banana')
#
# # cassini, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/cassini.arff')
# # cassini = pd.DataFrame(cassini)
# # cassini = cassini.astype(float)
# # cassini.columns = [i for i in range(0, cassini.shape[1])]
# # cassini = pd.concat([cassini[2], cassini.drop(2, axis=1)], axis=1)
# # dataset_list.append(cassini)
# # cluster_list.append(3)
# # name_list.append('cassini')
#
atom, _ = arff.loadarff('/Users/zz/PycharmProjects/fuck-kmeans/artificial/atom.arff')
atom = pd.DataFrame(atom)
atom = atom.astype(float)
atom.columns = [i for i in range(0, atom.shape[1])]
atom = pd.concat([atom[3], atom.drop(2, axis=1)], axis=1)
dataset_list.append(atom)
cluster_list.append(3)
name_list.append('atom')