import numpy as np
from sklearn import preprocessing, svm
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from sklearn.model_selection import GridSearchCV


def Get_data(Datafile_name,Clinicalfile_name):
    set_data=pd.read_csv(Datafile_name)
    set_data=set_data.iloc[:,1:]
    set_clinical=pd.read_csv(Clinicalfile_name)
    set_label = np.array(set_clinical.iloc[:,1])
    return set_data,set_label

def SVM_across(data_train, label_train, data_test, label_test):
    # data_train, label_train = Get_data('./Drug_data/CelllineFile/Paclitaxel_exp_after3.csv',
    #                                './Drug_data/CelllineFile/Paclitaxel_label2.csv')
    # data_train = preprocessing.scale(data_train)
    # data_test, label_test = Get_data('./Drug_data/PatientFile/GSE22513_data_after2.csv',
    #                              './Drug_data/PatientFile/GSE22513_label.csv')
    # data_test = preprocessing.scale(data_test)

    # Create SVM classifier
    svc = svm.SVC(C=5, kernel='linear', probability=True)

    # Fit the model on your training data
    svc.fit(data_train, label_train)

    # Predict probabilities on test data
    y_pred = svc.predict_proba(data_test)[:, 1]
    return label_test, y_pred

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
#
# # 创建 SVM 分类器
# svc = svm.SVC(probability=True)
#
# # 定义要搜索的参数网格
# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf']
# }
#
# # 创建网格搜索对象
# grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='roc_auc')
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
# # 计算并打印AUC和AUPRC
# auc = roc_auc_score(label_test, y_pred)
# auprc = average_precision_score(label_test, y_pred)
# print("AUC: ", auc)
# print("AUPRC: ", auprc)