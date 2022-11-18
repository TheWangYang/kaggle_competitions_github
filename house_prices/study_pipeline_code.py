# Reference link : https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard
# 导入需要的库
# import some necessary libraries
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew
import numpy as np

# 限制floats类型的数据输出为小数点后3位
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

from subprocess import check_output
# 检查目录下是否存在文件
# print(check_output(["train.csv", "./data"]).decode("utf8"))


# 导入训练数据和测试数据
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test_code.csv')


# 展示训练数据的前5行
# print(train.head(5))
# print(test_code.head(5))

# 检查样本数据和特征
# print("The train data size before dropping Id features is : {}".format(train.shape))
# print("The test_code data size before dropping Id features is : {}".format(test_code.shape))


# 保存'Id'列表
train_data_Id = train['Id']
test_data_Id = test['Id']

# 扔掉id列
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# 检查目前数据长度
# print("\nThe train data size after dropping Id feature is : {} ".format(train_data.shape))
# print("The test_code data size after dropping Id feature is : {} ".format(test_data.shape))


# # ------------------------数据预处理------------------------
# # 绘制GrLivArea和SalePrice的关系
# fig, ax = plt.subplots()
# ax.scatter(x=train_data['GrLivArea'], y=train_data['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()


# 可以放心地删除属性GrLivArea>4000且<300000的数据
train = train.drop(train[(train['GrLivArea'] > 4000) &
                         (train['SalePrice'] < 300000)].index)

# # 删除之后重新检查数据分布
# fig, ax = plt.subplots()
# ax.scatter(train_data['GrLivArea'], train_data['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()


# # ---------------对SalePrice变量进行分析--------------
# sns.distplot(train_data['SalePrice'], fit=norm)
#
# # 得到拟合参数
# (mu, sigma) = norm.fit(train_data['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# # 绘制分布曲线
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')
# plt.ylabel('Frequency')
# plt.xlabel('SalePrice distribution')
#
#
# # 得到QQ-plot
# fig = plt.figure()
# res = stats.probplot(train_data['SalePrice'], plot=plt)
# plt.show()
#
# # --------------对SalePrice进行Log变换---------------
train['SalePrice'] = np.log1p(train['SalePrice'])


# 检查新的分布
# sns.distplot(train['SalePrice'], fit=norm)


# 得到norm拟合参数
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu  = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# 绘制分布曲线
# plt.legend(['Normal dist. ($\mu=$ {:/2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#            loc='best')


# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')


# 得到QQ-plot
# fig = plt.figure()
# # res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()


# -------展示了目标变量SalePrice之后，进行特征工程--------


# 合并train_data和test_data
ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
# print("all_data size is : {}".format(all_data.shape))


# 查看数据的丢失率
# all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
# # 得到前30个数据丢失率对应的属性
# all_data_na = \
#     all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
#
# # 得到丢失数据情况字典，便于后续展示
# missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
# print(missing_data.head(20))


# # 展示丢失率在前30的属性
# f, ax = plt.subplots(figsize=(15, 21))
# plt.xticks(rotation='90')
# sns.barplot(x=all_data_na.index, y=all_data_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by features', fontsize=15)
# plt.show()


# 展示所有的属性对SalePrice的影响
# corrmat = train.corr()
# plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()


# --------------对有缺失值的属性进行插入-----------------------
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# 对于GarageType,GarageFinish,GarageQual,GarageCond四个属性来说，使用None进行填充
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna("None")


# 对'GarageYrBlt', 'GarageArea', 'GarageCars'属性使用0填充
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
#             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# 除去Utilities属性
all_data = all_data.drop(['Utilities'], axis=1)


# 该属性中的None被设置为"典型"
all_data["Functional"] = all_data["Functional"].fillna("Typ")


all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# # 查看是否有缺失值存在
# all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
# missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
# print(missing_data.head())


# -----------------实施更多的特征工程-----------------
# 转换一些用于真正分类的变量
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


# 更改OverallCond变量为可分类变量
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# Year和month sold被转换为可分类变量
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# 标签编码一些可能在其排序集中包含信息的分类变量
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape
# print('Shape all_data: {}'.format(all_data.shape))


# -----------------------增加一个更重要的特征--------------------
all_data['TotalSF'] = \
    all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# -----------Skewed features，倾斜特征-----------
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
# skewness.head(10)


# （高度）偏斜特征的 Box Cox 变换
skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    # all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)


# 获取虚拟分类特征
all_data = pd.get_dummies(all_data)
# print(all_data.shape)


# 得到新的训练集和测试集
new_train_data = all_data[:ntrain]
new_test_data = all_data[ntrain:]


# ----------------------建模------------------------
# 导入相关的库
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# 定义交叉验证策略
# 定义n折
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(new_train_data.values)
    rmse = np.sqrt(-cross_val_score(model, new_train_data.values, y_train,
                                    scoring="neg_mean_squared_error", cv=kf))
    return rmse


# -----------------------基础模型pipeline-----------------------
# lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))


# elastic net regression
ENet = make_pipeline(RobustScaler(),
                     ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))


# KRR
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# GBoost
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

#
# # 基础模型得分scores
# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
# score = rmsle_cv(ENet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
# score = rmsle_cv(GBoost)
# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
# score = rmsle_cv(model_xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# 堆叠模型

# # 1.简单地平均models分数
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#
#
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#
#
#         for model in self.models_:
#             model.fit(X, y)
#
#         return self
#
#
#
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models_
#         ])
#
#         return np.mean(predictions, axis=1)
#
#
# averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
#
#
# score = rmsle_cv(averaged_models)
# print("[INFO] Averaging base models score : {:.4f} ({:.4f}) \n".
#       format(score.mean(), score.std()))


# 2.较复杂堆叠模型
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])

        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)

# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# 3.集成 StackedRegressor、XGBoost 和 LightGBM
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# 堆叠分类器得分
stacked_averaged_models.fit(new_train_data.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(new_train_data.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(new_test_data.values))

print("[INFO] stacked_pred score is : {}".format(rmsle(y_train, stacked_train_pred)))

# 使用XGBoost
model_xgb.fit(new_train_data, y_train)
xgb_train_pred = model_xgb.predict(new_train_data)
xgb_pred = np.expm1(model_xgb.predict(new_test_data))
print("[INFO] xgboost score is : {}".format(rmsle(y_train, xgb_train_pred)))


# LightGBM算法
model_lgb.fit(new_train_data, y_train)
lgb_train_pred = model_lgb.predict(new_train_data)
lgb_pred = np.expm1(model_lgb.predict(new_test_data))
print("[INFO] lgb score is : {}".format(rmsle(y_train, lgb_train_pred)))


# 使用RMSE在整个train数据上的loss
print("[INFO] RMSLE score on train data is : {}".
      format(rmsle(y_train, stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15)))


# 集合模型在test data上进行预测的loss值，也就是最后提交的值
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

# 提交
sub = pd.DataFrame()
sub['Id'] = test_data_Id
sub['SalePrice'] = ensemble
sub.to_csv('./submission/202207311443_house_price_submission.csv', index=False)























