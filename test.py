import random
import time
import glob
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from main import NK, LK

# load all csv files
csv_files_path = './rebate/*/*.csv'  # 匹配csv文件路径
csv_files_list = glob.glob(csv_files_path)  # 获取所有匹配到的文件路径
headers = ['客户id', '分销关系关联时间', '产品id', '实例id', '实例首购时间', '支付时间', '原价', '应付金额', '实付金额',
           '评级业绩', '消费账号数', '佣金比例', '佣金金额', '佣金金额']
df = pd.concat([pd.read_csv(file_path, low_memory=False)[headers] for file_path in csv_files_list], axis=0,
               ignore_index=True)
# df = df.sample(100000)
df.head()
print(df.shape)
# print(df.dtype)
# print(df.describe().T)
df = df.fillna(method='ffill')
# null_all = df.isnull().sum()
# print(null_all)

for header in ['原价', '应付金额', '实付金额', '评级业绩', '消费账号数']:
    df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))

load_time = [datetime.datetime.strptime(load_time.split()[0], '%Y-%m-%d') for load_time in
             df['分销关系关联时间']]
first_time = [datetime.datetime.strptime(first_time.split()[0], '%Y-%m-%d') for first_time in
              df['实例首购时间']]
pay_time = [datetime.datetime.strptime(pay_time.split()[0], '%Y-%m-%d') for pay_time in df['支付时间']]
df['L'] = [(pay_time[i] - load_time[i]).days for i in range(0, len(pay_time))]
df['R'] = [(pay_time[i] - first_time[i]).days for i in range(0, len(pay_time))]
# df['F'] = [order_count[customer] for customer in df['客户id']]
# df['M'] = [payment_sum[customer] for customer in df['客户id']]
# df['C'] = df['佣金比例']
# features = ['客户id', 'L', 'R', 'F', 'M', 'C']
# df = df[features]

l_dict = df.groupby('客户id').mean()['L'].to_dict()
r_dict = df.groupby('客户id').mean()['R'].to_dict()
f_dict = df['客户id'].value_counts().to_dict()
m_dict = df.groupby('客户id').sum()['实付金额'].to_dict()
c_dict = df.groupby('客户id').mean()['佣金比例'].to_dict()
data = []
for k, v in l_dict.items():
    data.append([v, r_dict[k], f_dict[k], m_dict[k], c_dict[k]])
    # data_dict[k]['id'] = k
    # data_dict[k]['l'] = v
    # data_dict[k]['r'] = r_dict[k]
    # data_dict[k]['f'] = f_dict[k]
    # data_dict[k]['m'] = m_dict[k]
    # data_dict[k]['c'] = c_dict[k]

df = pd.DataFrame(data, columns=['l', 'r', 'f', 'm', 'c'])
# df.columns = ['id', 'l', 'r', 'f', 'm', 'c']
print(df.shape)
# df.columns = features

# x = [10000, 20000, 50000, 100000, 200000, 300000]
# alpha = [0, 0.01, 0.02, 0.05, 0.1]
# y = [[0 for i in range(0, len(x))] for i in range(0, len(alpha))]
# iterations = 1
# for iteration in range(0, iterations):
#     for i, size in enumerate(x):
#         df_sample = df.sample(size)
#         scaler = StandardScaler()
#         df_sample = scaler.fit_transform(df_sample)
#         # print("Training Linear Kmeans with k="+str(i))
#
#         for iter, a in enumerate(alpha):
#             start_time = time.time()
#             # print("alpha: ", a)
#             n_kmeans = LK(k=3, alpha=a)
#             n_kmeans.fit(df_sample)
#             y[iter][i] += n_kmeans.iteration
#             # print("Time Taken for Linear Kmeans:", time.time() - start_time)
#
# plot_type = ['^', 'd', 's', 'o', '*']
# for iter, a in enumerate(alpha):
#     plt.plot(x, [y_val/iterations for y_val in y[iter]], plot_type[iter]+'-')
# # plt.plot(x,y_001)
# plt.show()

x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
y = []
scaler = StandardScaler()
df = scaler.fit_transform(df)
print(df)
for i in range(0, len(x)):
    n_kmeans = NK(k=x[i])
    n_kmeans.fit(df)
    y.append(n_kmeans.sse/x[i])
plt.plot(x, y)
plt.show()
