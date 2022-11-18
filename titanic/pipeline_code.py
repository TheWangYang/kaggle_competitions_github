from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np


# 定义evaluate函数用来评价在test集上的表现
def evaluate_on_train(predictions, train_data):
    acc = 0
    for i in range(len(train_data)):
        if predictions[i] == train_data.Survived[i]:
            acc += 1
    return acc / len(train_data)


# 导入数据
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test_code.csv')


# 数据处理
# 填充缺失值
# 缺失值填充，Age列缺失的值，按中位数填充
# train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
#
# # 把机器学习不能处理的字符值转换成机器学习可以处理的数值
# train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
# train_data.loc[train_data["Sex"] == "female", "Sex"] = 1
#
# # 通过统计三个登船地点人数最多的填充缺失值
# train_data["Embarked"] = train_data["Embarked"].fillna("S")
#
# # 字符处理
# train_data.loc[train_data["Embarked"] == "S", "Embarked"] = 0
# train_data.loc[train_data["Embarked"] == "C", "Embarked"] = 1
# train_data.loc[train_data["Embarked"] == "Q", "Embarked"] = 2


'''
----------------使用不同种类代码进行测试------------------
'''
# 使用官方代码进行测试
# 得到训练数据的标签
y_train = train_data["Survived"]

# 选择需要使用的特征
features = ["Pclass", "Sex", "SibSp", "Parch"]
# 得到训练x特征
X_train = pd.get_dummies(train_data[features])
# 得到测试x特征
X_test = pd.get_dummies(test_data[features])

# ----------------------使用随机森林来实现分类----------------
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# --------------------------使用LR算法--------------------
lr = LogisticRegression(max_iter=2000)


# -----------------使用kNN最近邻算法--------------------
knn = KNeighborsClassifier()

# -----------------使用决策树算法---------------
dt = tree.DecisionTreeClassifier(random_state=1)

# ---------------使用SVC算法-----------
svc = SVC(probability=True)

# 使用XGBClassifier算法
xgb = XGBClassifier(random_state=1)

# 使用投票策略集成模型

voting_classifier = VotingClassifier(estimators=[('lr', lr), ('knn', knn), ('rfc', rfc),
                                       ('svc', svc), ('xgb', xgb)], voting='soft')

# 模型拟合训练样本
voting_classifier.fit(X_train, y_train)

# 查看在train dataset上的train_acc
# predictions = voting_classifier.predict(X_train)

# 得到在test数据集上的结果
predictions = voting_classifier.predict(X_test)

# 得到train_acc
# train_acc = evaluate_on_train(predictions, train_data)
# print("[INFO] train_acc is {}".format(train_acc))


# 处理结果
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv("./submission/202207291542_vote_classifier_submit.csv", index=False)
print("[INFO] Submission was successfully saved!")

# [INFO]the RF's result is 0.77511
# [INFO]the DT's result is 0.76555
# [INFO]the SVC's result is


# ---------------使用数据预处理等标准pipeline进行预测--------------------
# 数据处理，第1种
# 合并训练和测试数据为 all_data，用Na填充测试数据的Survived属性
# train_data['train_test'] = 1
# test_data['train_test'] = 0
# test_data['Survived'] = np.NaN
# # 将test_data和train_data进行合并
# all_data = pd.concat([train_data, test_data])
#
#
# # 对训练和测试数据进行特征转换（进行特征工程时使用的属性）
# all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
# all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
# all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
#
# # 删除Embarked中含缺失值的两条数据
# all_data.dropna(subset=['Embarked'], inplace=True)
#
# # 用均值填充Age Fare中的缺失值
# all_data.Age = all_data.Age.fillna(train_data.Age.mean())
# all_data.Fare = all_data.Fare.fillna(train_data.Fare.mean())
#
# # 对Fare属性进行log函数转换，转换后的值分布接近正态分布
# all_data['norm_fare'] = np.log(all_data.Fare+1)
# all_data['norm_fare'].hist()
# all_data.Pclass = all_data.Pclass.astype(str)
#
# # created dummy variables from categories (also can use OneHotEncoder)
# all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp',
# 'Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','train_test']])
#
# # Split to train test_code again
# X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
# X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)
#
# y_train = all_data[all_data.train_test == 1].Survived
#
# # Scale Data
# scale = StandardScaler()
# all_dummies_scaled = all_dummies.copy()
# all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']] = \
#     scale.fit_transform(all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
# # .fit_transform 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
#
# X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis=1)
# X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)
#
# y_train = all_data[all_data.train_test == 1].Survived


# 1.naive bayes
# 使用朴素贝叶斯网络
# gnb = GaussianNB()
# cv = cross_val_score(gnb, X_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())
# [0.67977528 0.6741573  0.71910112 0.73595506 0.79096045]
# 0.7199898432044689

# 1.使用LR回归
# logistic regression
# lr = LogisticRegression(max_iter=2000)
# cv = cross_val_score(lr, X_train_scaled, y_train, cv=5)
# print(cv)
# print(cv.mean())
#[0.78651685 0.80337079 0.78089888 0.79775281 0.82485876]
#0.7986796165809688










