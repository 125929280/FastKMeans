import pandas as pd
from sklearn.preprocessing import StandardScaler

from main import NK

df = pd.read_csv('./rebate/2021-07/杭州驻云-2021年7月.csv', header='infer')
headers = ['原价', '应付金额', '实付金额', '评级业绩', '消费账号数',
          '佣金比例', '佣金金额']
for header in headers:
    df[header] = df[header].apply(lambda x: float(str(x).replace(",", "")))
df.fillna(0, inplace=True)
X = df[headers].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
n_kmeans = NK(10)
n_kmeans.fit(X)
