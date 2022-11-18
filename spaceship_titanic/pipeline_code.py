# 导入需要的库
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.stats import norm, skew

# 导入封装好的模型算法
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# 定义evaluate函数用来评价在test集上的表现
def evaluate_on_train(predictions, y_train):
    acc = 0
    for i in range(len(y_train)):
        if predictions[i] == y_train[i]:
            acc += 1
    return acc / len(y_train)


# 定义忽略警告函数
def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

# 限制floats类型的数据输出为小数点后3位
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# 导入数据
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# 观察数据
# print("[INFO] train dataset: {}".format(train.head(5)))
# print("[INFO] test_code dataset: {}".format(test_code.head(5)))


# 得到训练数据和测试数据的总数
# print("[INFO] train size is : {}".format(train.shape[0]))
# print("[INFO] test_code size is : {}".format(test_code.shape[0]))

# [INFO] train size is : 8693
# [INFO] test_code size is : 4277

# 根据上述结果展示可以看到训练数据集占比为2/3，测试数据集占比为1/3

# 保存train和test中的Id列
train_Id = train["PassengerId"]
test_Id = test["PassengerId"]

# 由于Id对训练数据没有影响，因此去除掉Id列
train.drop("PassengerId", axis=1, inplace=True)
test.drop("PassengerId", axis=1, inplace=True)

# -----------------------------数据预处理------------------------------------
# 得到train和test数据集的长度
ntrain = train.shape[0]
ntest = test.shape[0]

# 得到train中的标签
y_train = train["Transported"]
y_train = y_train.astype(int)  # 将bool转换为int类型，从而实现false/true到0/1的转换

# 合并train和test数据集
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop("Transported", axis=1, inplace=True)


# 查看数据的丢失率
# all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
# # 得到前30个数据丢失率对应的属性
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
#
# # 得到丢失数据情况字典，便于后续展示
# missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
#
# print(missing_data)


# 查看所有属性对label的影响
# 展示所有的属性对Transported的影响

# corrmat = train.corr()
# plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()


# ----------------------对all_data中需要作为features使用的数据进行填充---------------------
# 1.对字符类型和Bool类型的使用None进行填充，对数字类型使用0进行填充
# all_data["ShoppingMall"] = all_data["ShoppingMall"].fillna(0)
# all_data["VIP"] = all_data["VIP"].fillna("None")
# all_data["HomePlanet"] = all_data["HomePlanet"].fillna("None")
# all_data["CryoSleep"] = all_data["CryoSleep"].fillna("None")
# all_data["VRDeck"] = all_data["VRDeck"].fillna(0)
# all_data["FoodCourt"] = all_data["FoodCourt"].fillna(0)
# all_data["RoomService"] = all_data["RoomService"].fillna(0)
# all_data["Age"] = all_data["Age"].fillna(0)


# 参考链接：https://www.kaggle.com/code/viktortaran/space-titanic-0-80336-from-neophyte
# 2.使用上述参考链接填充模式代码
all_data["CryoSleep"] = all_data["CryoSleep"].fillna("Unknown")

# 增加相加的属性
all_data1 = all_data
all_data1["Expenses"] = all_data1.iloc[:, 7: 12].sum(axis=1)


# 设置CroySleep属性为新属性和老属性的综合结果
def CroySleep_fill(x, y):
    if x == "Unknown" and y == 0:
        return True
    elif x == "Unknown" and y != 0:
        return False
    else:
        return x


# 给CryoSleep属性重新设置属性值
all_data1["CryoSleep"] = all_data1[["CryoSleep", "Expenses"]].apply(lambda row: CroySleep_fill(row["CryoSleep"],
                                                                                               row["Expenses"]), axis=1)

# 重新设置CryoSleep属性，设置为False
all_data["CryoSleep"] = all_data["CryoSleep"].astype('bool')

# print(all_data["CryoSleep"].describe())

# 对新生成的属性Expense的处理
# 得到生成新属性的原始属性的Mean值
RoomService_mean = all_data.loc[all_data.CryoSleep == 0].RoomService.mean()
FoodCourt_mean = all_data.loc[all_data.CryoSleep == 0].FoodCourt.mean()
ShoppingMall_mean = all_data.loc[all_data.CryoSleep == 0].ShoppingMall.mean()
Spa_mean = all_data.loc[all_data.CryoSleep == 0].Spa.mean()
VRDeck_mean = all_data.loc[all_data.CryoSleep == 0].VRDeck.mean()

# 对7-12列的属性，进行空缺值填充
all_data.iloc[:, 7: 12] = all_data.iloc[:, 7: 12].fillna("Unknown")

# print("before, null values is :\n{}".format(all_data.isnull().sum()))


# 设置填充的函数
def fill_in(x, y, mean):
    if x == True and y == "Unknown":
        return 0
    elif x == False and y == "Unknown":
        return mean
    else:
        return y


# all_data['RoomService'] = all_data[['CryoSleep', 'RoomService']].apply(
#     lambda row: fill_in(row['CryoSleep'], row['RoomService'], RoomService_mean), axis=1)

# 使用mean中值填充
all_data['RoomService'] = all_data['RoomService'].fillna(RoomService_mean)

# print("RoomService null values size is : {}".format(all_data['RoomService'].isnull().sum()))

all_data['FoodCourt'] = all_data[['CryoSleep', 'FoodCourt']].apply(
    lambda row: fill_in(row['CryoSleep'], row['FoodCourt'], FoodCourt_mean), axis=1)
all_data['ShoppingMall'] = all_data[['CryoSleep', 'ShoppingMall']].apply(
    lambda row: fill_in(row['CryoSleep'], row['ShoppingMall'], ShoppingMall_mean), axis=1)
all_data['Spa'] = all_data[['CryoSleep', 'Spa']].apply(
    lambda row: fill_in(row['CryoSleep'], row['Spa'], Spa_mean), axis=1)
all_data['VRDeck'] = all_data[['CryoSleep', 'VRDeck']].apply(
    lambda row: fill_in(row['CryoSleep'], row['VRDeck'], VRDeck_mean), axis=1)

# 使用“Unknown Unknown”对Name属性进行填充
all_data.Name = all_data.CryoSleep.fillna("Unknown Unknown")

# 填充HomePlanet和Destination
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
object_cols = ['HomePlanet', 'Destination']
all_data.HomePlanet = all_data.HomePlanet.fillna('Earth')
all_data.Destination = all_data.Destination.fillna('PSO J318.5-22')
all_data[object_cols] = ordinal_encoder.fit_transform(all_data[object_cols])

# 填充Age属性
Age_mean_adult = all_data.loc[all_data.Expenses > 0].Age.mean()
Age_mean_kids = all_data.loc[all_data.Expenses == 0].Age.mean()
Age_mean_all = all_data.Age.mean()


# 定义填充年龄的函数
def fill_in_age(x):
    if x > 0:
        return Age_mean_adult
    elif x == 0:
        return Age_mean_kids
    else:
        return


# 直接使用age_mean_adult数值进行填充
all_data["Age"] = all_data["Age"].fillna(Age_mean_adult)

# 填充VIP
all_data["VIP"] = all_data["VIP"].fillna(0)

# 填充Cabin属性
all_data.Cabin.str.split("/", expand=True)

# 得到两个新属性
all_data['Cabin_Deck'] = all_data.Cabin.str.split("/", expand=True)[0]
all_data['Cabin_Side'] = all_data.Cabin.str.split("/", expand=True)[2]


# 填充得到的关于Cabin属性值
all_data['Cabin_Deck'] = all_data['Cabin_Deck'].fillna("F")
all_data['Cabin_Side'] = all_data['Cabin_Side'].fillna("P")

object_cols_1 = ['Cabin_Side', 'Cabin_Deck']
all_data[object_cols_1] = ordinal_encoder.fit_transform(all_data[object_cols_1])

# print("after, null values is :\n{}".format(all_data.isnull().sum()))


# --------------------------进行更多的特征工程-----------------------------
# -----------Skewed features，倾斜特征-----------
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# # print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew': skewed_feats})
# # skewness.head(10)
# # （高度）偏斜特征的 Box Cox 变换
# skewness = skewness[abs(skewness) > 0.75]
# # print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     # all_data[feat] += 1
#     all_data[feat] = boxcox1p(all_data[feat], lam)




# --------------------------得到需要的特征-------------------------
# 选择其中对Transported标签影响较大的属性
features = ['HomePlanet','CryoSleep', 'Destination',
            'Age','VIP','RoomService',
            'FoodCourt','ShoppingMall',
            'Spa','VRDeck','Expenses',
            'Cabin_Deck','Cabin_Side']

# 原来使用的属性：["HomePlanet", "CryoSleep", "Age", "VRDeck", "VIP", "ShoppingMall", "FoodCourt", "RoomService"]

# 得到独热编码
all_data = pd.get_dummies(all_data[features])

# 得到训练集和测试集
# 得到新的训练集和测试集
new_X_train = all_data[:ntrain]
new_X_test = all_data[ntrain:]
# print("[INFO] train features shape is: {}, test_code features shape is: {}".format(new_X_train.shape, new_X_test.shape))


# ---------------------------初始化模型算法-------------------------

# ----------------------使用随机森林来实现分类-------------------
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# rfc.fit(new_X_train, y_train)
# predictions = rfc.predict(new_X_test)


# --------------------------使用LR算法--------------------
lr = LogisticRegression(max_iter=2000)
# lr.fit(new_X_train, y_train)
# predictions = lr.predict(new_X_test)


# # -----------------使用kNN最近邻算法--------------------
knn = KNeighborsClassifier()
# knn.fit(new_X_train, y_train)
# predictions = knn.predict(new_X_test)



# -----------------使用决策树算法---------------
dt = tree.DecisionTreeClassifier(random_state=1)
# dt.fit(new_X_train, y_train)
# predictions = dt.predict(new_X_test)



# ---------------使用SVC算法-----------
svc = SVC(probability=True)
svc.fit(new_X_train, y_train)
predictions = svc.predict(new_X_test)


# 使用XGBClassifier算法
xgb = XGBClassifier(random_state=1)
# xgb拟合数据，在train上训练
# xgb.fit(new_X_train, y_train)
# predictions = xgb.predict(new_X_train)
# 使用xgb算法在test数据集上进行测试
# predictions = xgb.predict(new_X_test)


# 使用投票策略集成模型
voting_classifier = \
    VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('rfc', rfc), ('svc', svc), ('xgb', xgb)], voting='soft')
# 模型拟合训练样本
# voting_classifier.fit(new_X_train, y_train)
# 查看在train dataset上的train_acc
# predictions = voting_classifier.predict(X_train)
# 得到在test数据集上的结果
# predictions = voting_classifier.predict(new_X_test)


# 得到train_acc
# train_acc = evaluate_on_train(predictions, y_train)
# print("[INFO] train_acc is {}".format(train_acc))


new_predictions = []
# 将predictions转换为true或false
for i, label in enumerate(predictions):
    if label == 1:
        new_predictions.append(True)
    else:
        new_predictions.append(False)

# print("[INFO] After transfer, predictions is : {}".format(new_predictions))

# 处理结果
output = pd.DataFrame({'PassengerId': test_Id, 'Transported': new_predictions})
output.to_csv("./submission/202208011904_SVC_classifier_submit.csv", index=False)
print("[INFO] Submission was successfully saved!")
