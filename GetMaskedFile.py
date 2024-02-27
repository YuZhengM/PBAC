import pandas as pd
import numpy as np

# DavidLiu=pd.read_csv('./Immune_data/PatientFile/DavidLiu.csv')
# PatientGeneList=list(DavidLiu.iloc[:,0])
# print(len(PatientGeneList))

# GSE115821=pd.read_csv('Immune_data/PatientFile/GSE115821.csv')
# PatientGeneList=list(GSE115821.iloc[:,0])

# GSE100797=pd.read_csv('Immune_data/PatientFile/GSE100797.csv')
# PatientGeneList=list(GSE100797.iloc[:,0])

# GSE35640=pd.read_csv('Immune_data/PatientFile/GSE35640.csv')
# PatientGeneList=list(GSE35640.iloc[:,0])

# GSE19293=pd.read_csv('Immune_data/PatientFile/GSE19293_across.csv')
# GENE = GSE19293.columns
# PatientGeneList=list(GENE)
# print(PatientGeneList)

# GSE91061=pd.read_csv('Immune_data/PatientFile/GSE91061.csv')
# PatientGeneList=list(GSE91061.iloc[:,0])

# GSE78220=pd.read_csv('Immune_data/PatientFile/GSE78220.csv')
# PatientGeneList=list(GSE78220.iloc[:,0])

# IMvigor210=pd.read_csv('Immune_data/PatientFile/IMvigor210.csv')
# PatientGeneList=list(IMvigor210.iloc[:,0])

# PRJNA482620=pd.read_csv('Immune_data/PatientFile/PRJNA482620.csv')
# PatientGeneList=list(PRJNA482620.iloc[:,0])
#
# GSE176307=pd.read_csv('Immune_data/PatientFile/GSE176307.csv')
# PatientGeneList=list(GSE176307.iloc[:,0])

# Braun_2020=pd.read_csv('Immune_data/PatientFile/Braun_2020.csv')
# PatientGeneList=list(Braun_2020.iloc[:,0])

# GSE106128=pd.read_csv('Immune_data/PatientFile/GSE106128.csv')
# PatientGeneList=list(GSE106128.iloc[:,0])

# phs000452=pd.read_csv('Immune_data/PatientExpressFile/phs000452_across.csv')
# GENE = phs000452.columns
# PatientGeneList=list(GENE)
# print(PatientGeneList)
# PatientGeneList=list(phs000452.iloc[:,0])

# PRJEB23709=pd.read_csv('Immune_data/PatientFile/PRJEB23709.csv')
# PatientGeneList=list(PRJEB23709.iloc[:,0])

df1 = pd.read_csv('./Drug_data/CelllineFile/Paclitaxel_exp.csv', index_col=0)  # 假设基因名是第一列
df2 = pd.read_csv('./Drug_data/PatientFile/GSE22513_data.csv', index_col=0)  # 假设基因名是第一列
PatientGeneList = list(df1.index.intersection(df2.index))


#从基因通路关系文件中提取关键信息
# pathName_Geneset = pd.read_excel('./Immune_data/PathwayMaskedFile/pathwaySTEMI.xls')
# pathName_Geneset[['pathway_name','geneset']].to_csv("./Immune_data/PathwayMaskedFile/pathName_Geneset.csv")

pathName_Geneset=pd.read_csv('Immune_data/PathwayMaskedFile/pathName_Geneset.csv', index_col=0)
# print(pathName_Geneset.shape[0])
# print(pathName_Geneset.values[1][1].split(';'))


# 计算数据集与通路基因的交集
f=open("Immune_data/PathwayMaskedFile/Paclitaxel_gene.txt", "w")
for g in PatientGeneList:
    for i in range(pathName_Geneset.shape[0]):
        if g in pathName_Geneset.values[i][1].split(';'):
            f.write((g+'\n'))
            break
f.close()

Gene_result=[]
for line in open('Immune_data/PathwayMaskedFile/Paclitaxel_gene.txt'):
    Gene_result.append(line.strip('\n'))
print(len(Gene_result))

# data1=pd.read_csv('./Drug_data/CelllineFile/Bortezomib_exp.csv', index_col=0)
# data2 = pd.read_csv('./Drug_data/PatientFile/GSE9782_data.csv', index_col=0)
data1=df1.loc[Gene_result,:]
data2=df2.loc[Gene_result,:]
data1.to_csv('./Drug_data/CelllineFile/Paclitaxel_exp_after.csv')
data2.to_csv('./Drug_data/PatientFile/GSE22513_data_after.csv')

#计算pathwaymask的基因数
num=0
for g in Gene_result:
        num=num+1
print(num)

#构造pathwaymask
pathway_mask=np.zeros((pathName_Geneset.shape[0],num))
print(pathway_mask.shape)
for g in Gene_result:
    for i in range(pathName_Geneset.values.shape[0]):
        if g in pathName_Geneset.values[i][1].split(';'):
            pathway_mask[i][Gene_result.index(g)]=1
pd.DataFrame(pathway_mask).to_csv('./Drug_data/PathwaymaskFile/pathway_mask_Paclitaxel.csv', index=0, header=0)
