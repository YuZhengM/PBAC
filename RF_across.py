import numpy as np
import torch
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from sklearn.model_selection import GridSearchCV


def Get_data(Datafile_name,Clinicalfile_name):
    set_data=pd.read_csv(Datafile_name)
    set_data=set_data.iloc[:,1:]
    set_clinical=pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:,1])
    return set_data,set_label
# 读取训练和测试数据
def RF_across(data_train, label_train, data_test, label_test):
    # data_train, label_train = Get_data('./Drug_data/CelllineFile/Paclitaxel_exp_after3.csv',
    #                                './Drug_data/CelllineFile/Paclitaxel_label2.csv')
    # data_train = preprocessing.scale(data_train)
    # data_test, label_test = Get_data('./Drug_data/PatientFile/GSE22513_data_after2.csv',
    #                              './Drug_data/PatientFile/GSE22513_label.csv')
    # data_test = preprocessing.scale(data_test)


    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)


    rf.fit(data_train, label_train)

    # Predict probabilities on test data
    y_pred = rf.predict_proba(data_test)[:, 1]  # get the probabilities of positive class
    return label_test,y_pred
# label_test,y_pred=RF_across()
# auc = roc_auc_score(label_test, y_pred)
# auprc = average_precision_score(label_test, y_pred)
# print("AUC: ", auc)
# print("AUPRC: ", auprc)

# data_train, label_train = Get_data('./Drug_data/CelllineFile/Paclitaxel_exp_after3.csv',
#                                    './Drug_data/CelllineFile/Paclitaxel_label2.csv')
# data_train = preprocessing.scale(data_train)
# data_test, label_test = Get_data('./Drug_data/PatientFile/GSE22513_data_after2.csv',
#                                  './Drug_data/PatientFile/GSE22513_label.csv')
# ata_test = preprocessing.scale(data_test)
#
# # 创建随机森林分类器
# rf = RandomForestClassifier(random_state=42)
#
# # 定义要搜索的参数网格
# param_grid = {
#     'n_estimators': [10, 50, 100],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }
#
# # 创建网格搜索对象
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc')
#
# # 对训练数据执行网格搜索
# grid_search.fit(data_train, label_train)
#
# # 打印最佳参数
# print("Best parameters: ", grid_search.best_params_)
#
# # 使用最佳参数在测试集上进行预测
# y_pred = grid_search.predict_proba(data_test)[:, 1]  # 获取正类的概率
#
#
# # 计算并打印AUC和AUPRC
# auc = roc_auc_score(label_test, y_pred)
# auprc = average_precision_score(label_test, y_pred)
# print("AUC: ", auc)
# print("AUPRC: ", auprc)