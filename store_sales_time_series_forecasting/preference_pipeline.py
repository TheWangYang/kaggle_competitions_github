# 时间序列预测任务
# --------------------导入需要的包------------------
import numpy as np
import pandas as pd
import os
import gc
import warnings
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# -----------------------------配置---------------------------
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')


# ----------------------导入数据集--------------------------
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
stores = pd.read_csv('./data/stores.csv')
# 按照store_nbr和date属性进行排序
transactions = pd.read_csv('./data/transactions.csv').sort_values(['store_nbr', 'date'])


# ---------------------设置数据集中时间为日期格式--------------------
train['date'] = pd.to_datetime(train.date)
test['date'] = pd.to_datetime(test.date)
transactions['date'] = pd.to_datetime(transactions['date'])


# -------------------将数据集中的数据类型转换为需要的数据类型--------------------
train.onpromotion = train.onpromotion.astype('float16')
train.sales = train.sales.astype('float32')
stores.cluster = stores.cluster.astype('int8')

# print(train.head())


# --------------------------对数据集的处理-------------------------
# print(transactions.head())


# 使用transactions数据集生成新的数据特征
# 查看总销售额和交易之间的斯皮尔曼相关性检验
temp = pd.merge(train.groupby(['date', 'store_nbr']).sales.sum().reset_index(),
                transactions, how="left")


print("Spearman Correlation between Total Sales and Transactions : {:.4f}".
      format(temp.corr("spearman").sales.loc["transactions"]))

px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions',
        color='store_nbr', title="Transactions")


# 设置








