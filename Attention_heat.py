# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 读取两个xlsx文件
# df1 = pd.read_excel('./Attention_score/GSE78220_Attention_score.xlsx')
# df2 = pd.read_excel('./Attention_score/GSE100797_Attetion_score.xlsx')
# df3 = pd.read_excel('./Attention_score/GSE115821_Attention_score.xlsx')
# df4 = pd.read_excel('./Attention_score/phs000452_Attention_score.xlsx')
#
# # 在每个DataFrame中添加一个新列来表示数据来源
# df1['source'] = 'GSE78220'
# df2['source'] = 'GSE100797'
# df3['source'] = 'GSE115821'
# df4['source'] = 'phs000452'
#
# # 合并两个DataFrames
# df_combined = pd.concat([df1, df2, df3, df4])
#
# # 根据'注意力分数'列进行排序
# df_sorted = df_combined.sort_values(by='Attention_score', ascending=False)
#
# # 取前20个通路
# top_20 = df_sorted.head(20)
#
# # 创建热图
# plt.figure(figsize=(10,8))
# sns.heatmap(top_20.pivot("pathway_name", "source", "Attention_score"), annot=True, cmap="YlGnBu")
# plt.title('Top 20 pathway based on attention score')
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取xlsx文件
df = pd.read_csv('./Attention_score/IMvigor210_mean_attentions.csv')

# 根据'注意力分数'列进行排序
df_sorted = df.sort_values(by='mean_attention', ascending=False)

# 取前20个通路
top_10 = df_sorted.head(10)

# 创建热图
fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(top_10[['mean_attention']], annot=False, cmap="GnBu", yticklabels=top_10['pathway_name'], ax=ax)

# 调整边距以适应yticklabels
plt.subplots_adjust(left=0.5)

plt.title('Top 10 pathway based on attention score')
plt.show()