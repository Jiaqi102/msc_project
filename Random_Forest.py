import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns
import shap
df=pd.read_csv("D:/shannon_metabolites.csv")
# Create binary outcome based on Shannon index tertiles
df['tertile'] = pd.qcut(df['diversity_shannon'], q=3, labels=[0, 1, 2])

sns.histplot(df['diversity_shannon'], bins=30, kde=True)
plt.axvline(df['diversity_shannon'].quantile(1/3), color='r', linestyle='--', label='1st tertile')
plt.axvline(df['diversity_shannon'].quantile(2/3), color='g', linestyle='--', label='2nd tertile')
plt.legend()
plt.title('Shannon Diversity(urine) with Tertile Cutoffs')
plt.show()

df['binary_outcome'] = df['tertile'].apply(lambda x: 1 if x == 2 else (0 if x == 0 else None))
df = df.dropna(subset=['binary_outcome'])  # Remove middle tertile
df['binary_outcome'] = df['binary_outcome'].astype(int)

# Define variables
covariates = ['age', 'sex.x', 'bmi']
# Encode sex (assuming binary)
df['sex.x'] = LabelEncoder().fit_transform(df['sex.x'])

# Exclude specific columns to get metabolite names
exclude_cols = ['diversity_shannon', 'group', 'tertile', 'binary_outcome'] + covariates
metabolites = [col for col in df.columns if col not in exclude_cols]

# Prepare data
X_covariates = df[covariates]
X_metabolites = df[metabolites]
X_combined = pd.concat([X_covariates, X_metabolites], axis=1)
y = df['binary_outcome']
groups = df['group']

# 数据集划分
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X_combined, y, groups=groups))

# 训练集和测试集索引
Xcov_train, Xcov_test = X_covariates.iloc[train_idx], X_covariates.iloc[test_idx]
Xmet_train, Xmet_test = X_metabolites.iloc[train_idx], X_metabolites.iloc[test_idx]
Xcomb_train, Xcomb_test = X_combined.iloc[train_idx], X_combined.iloc[test_idx]

y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

#模型定义
def train_model(X_train, y_train, groups_train, X_test, y_test, name='Model'):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    group_kfold = GroupKFold(n_splits=5)
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=group_kfold,
        scoring='f1_macro',
        n_jobs=-1,
        refit=True
    )
    
    grid.fit(X_train, y_train, groups=groups_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    recall = recall_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Best Params:", grid.best_params_)
    print(f"F1: {f1:.3f}, AUC: {auc:.3f}, Recall: {recall:.3f}")
    
    return best_model, y_score
def train_model_lr(X_train, y_train, X_test, y_test, name='Model'):
    """逻辑回归"""
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    recall = recall_score(y_test, y_pred)

    print(f"\n=== {name} (Logistic Regression) ===")
    print(f"F1: {f1:.3f}, AUC: {auc:.3f}, Recall: {recall:.3f}")

    return model, y_score
# 模型训练与评估
model_cov1, yscore_cov1 = train_model(Xcov_train, y_train, groups_train, Xcov_test, y_test, name='Covariates')
model_met1, yscore_met1 = train_model(Xmet_train, y_train, groups_train, Xmet_test, y_test, name='Metabolites')
model_comb1, yscore_comb1 = train_model(Xcomb_train, y_train, groups_train, Xcomb_test, y_test, name='Combined')

model_cov2, yscore_cov2 = train_model_lr(Xcov_train, y_train, Xcov_test, y_test, name='Covariates')
model_met2, yscore_met2 = train_model_lr(Xmet_train, y_train, Xmet_test, y_test, name='Metabolites')
model_comb2, yscore_comb2 = train_model_lr(Xcomb_train, y_train, Xcomb_test, y_test, name='Combined')
# 绘图函数
def plot_roc(y_test, y_score, label, ax):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = roc_auc_score(y_test, y_score)
    ax.plot(fpr, tpr, label=f'{label} (AUC={auc_val:.2f})')

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 横排两个图

# 第一张图：Random Forest
plot_roc(y_test, yscore_cov1, 'Covariates (RF)', axes[0])
plot_roc(y_test, yscore_met1, 'Metabolites (RF)', axes[0])
plot_roc(y_test, yscore_comb1, 'Combined (RF)', axes[0])
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('A. ROC Urine - Random Forest', loc='left', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True)

# 第二张图：Logistic Regression
plot_roc(y_test, yscore_cov2, 'Covariates (LR)', axes[1])
plot_roc(y_test, yscore_met2, 'Metabolites (LR)', axes[1])
plot_roc(y_test, yscore_comb2, 'Combined (LR)', axes[1])
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('B. ROC Urine - Logistic Regression', loc='left', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
#查看covariates的分布
# sns.violinplot(x='binary_outcome', y='age', data=df)
# plt.title('Age by Shannon Diversity Class')
# plt.show()
# sns.violinplot(x='binary_outcome', y='bmi', data=df)
# plt.title('BMI by Shannon Diversity Class')
# plt.show()
# 创建一个新的列用于显示标签
df['diversity_label'] = df['binary_outcome'].map({0: 'Low', 1: 'High'})

# 绘制 age 分布图
sns.violinplot(x='diversity_label', y='age', data=df)
plt.title('Age by Shannon Diversity Class')
plt.xlabel('Shannon Diversity Class')
plt.ylabel('Age')
plt.show()

# 绘制 bmi 分布图
sns.violinplot(x='diversity_label', y='bmi', data=df)
plt.title('BMI by Shannon Diversity Class')
plt.xlabel('Shannon Diversity Class')
plt.ylabel('BMI')
plt.show()

df['diversity_label'] = df['binary_outcome'].map({0: 'Low', 1: 'High'})

# 函数：绘图+统计检验
def plot_violin_with_pval(var):
    df_high = df[df['diversity_label'] == 'High']
    df_low = df[df['diversity_label'] == 'Low']
    stat, p = mannwhitneyu(df_high[var], df_low[var], alternative='two-sided')

    sns.violinplot(x='diversity_label', y=var, data=df)
    plt.title(f'{var} by Shannon Diversity Group\nMann–Whitney U test: p = {p:.3f}')
    plt.xlabel('Shannon Diversity Group')
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

# 调用
plot_violin_with_pval('age')
plot_violin_with_pval('bmi')

from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt

# 性别比例表
contingency = pd.crosstab(df['diversity_label'], df['sex.x'])
chi2, p, dof, expected = chi2_contingency(contingency)

# 百分比堆叠条形图
sex_perc = contingency.div(contingency.sum(axis=1), axis=0)
sex_perc.plot(kind='bar', stacked=True)

plt.title(f'Sex Distribution by Shannon Diversity Group\nChi-squared test: p = {p:.3f}')
plt.xlabel('Shannon Diversity Group')
plt.ylabel('Proportion')
plt.legend(title='Sex (0=Male, 1=Female)')
plt.tight_layout()
plt.show()
def mean_std(series):
    return f"{series.mean():.1f} ± {series.std():.1f}"

# 分组统计
age_summary = df.groupby('diversity_label')['age'].apply(mean_std)
bmi_summary = df.groupby('diversity_label')['bmi'].apply(mean_std)

print("Age summary by diversity group:")
print(age_summary)
print("\nBMI summary by diversity group:")
print(bmi_summary)
# 统计频数
sex_counts = pd.crosstab(df['diversity_label'], df['sex.x'])

# 统计比例
sex_props = sex_counts.div(sex_counts.sum(axis=1), axis=0).round(2)

print("\nSex counts (0 = Male, 1 = Female):")
print(sex_counts)

print("\nSex proportions (row-wise):")
print(sex_props)

from scipy.stats import mannwhitneyu, chi2_contingency

# 先确保数据准备好
df_high = df[df['binary_outcome'] == 1]
df_low = df[df['binary_outcome'] == 0]

# 连续变量：age, bmi → Mann–Whitney U 检验
for var in ['age', 'bmi']:
    stat, p = mannwhitneyu(df_high[var], df_low[var], alternative='two-sided')
    print(f"{var} Mann–Whitney U test: U={stat:.2f}, p={p:.4f}", "← 显著差异" if p < 0.05 else "← not significant")
# 类别变量：sex.x → 卡方检验
contingency = pd.crosstab(df['binary_outcome'], df['sex.x'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nsex.x chi2 test: chi2={chi2:.2f}, p={p:.4f}", "← 显著差异" if p < 0.05 else "← not significant")

#特征选择
importances = model_comb1.feature_importances_
feature_names = Xcomb_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# 可视化前 15 个特征
feat_imp.head(15).plot(kind='barh', title='Top 15 Important Features')
plt.gca().invert_yaxis()
plt.show()

# #手动选择 Top N 个特征（可以逐步测试 Top 5, Top 10, Top 15）
# top_k = 5
# top_features = feat_imp.head(top_k).index

# Xcomb_train_top = Xcomb_train[top_features]
# Xcomb_test_top = Xcomb_test[top_features]

# #用精简后的征集重新训练模型并评估
# model_met_top, yscore_met_top = train_model(
#     Xcomb_train_top, y_train, groups_train, Xcomb_test_top, y_test, name=f'Metabolites_Top{top_k}'
# )

# 循环 top_k 从 5 到 30，每次增加 5
results = []
for top_k in range(5, 41, 5):
    top_features = feat_imp.head(top_k).index

    # 构建训练集和测试集（仅限选出的特征）
    X_train_top = Xmet_train[top_features]
    X_test_top = Xmet_test[top_features]

    # 训练模型
    model_top, yscore_top = train_model(
        X_train_top, y_train, groups_train,
        X_test_top, y_test,
        name=f'Metabolites_Top{top_k}'
    )

    # 模型预测
    y_pred = model_top.predict(X_test_top)

    # 评估指标
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, yscore_top)
    recall = recall_score(y_test, y_pred)

    # 保存结果
    results.append({
        'Top_K_Features': top_k,
        'F1': round(f1, 3),
        'AUC': round(auc, 3),
        'Recall': round(recall, 3),
        'Model': model_top,
        'Features': list(top_features)
    })

# 转成 DataFrame 查看结果
results_df = pd.DataFrame(results)
print("\n===== The changes in model performance with the number of features =====")
print(results_df[['Top_K_Features', 'F1', 'AUC', 'Recall']])

# 假设你已有的 results_df 含有 Top_K_Features, F1, AUC
plt.figure(figsize=(8, 5))

# 绘制 AUC 和 F1 曲线
plt.plot(results_df['Top_K_Features'], results_df['AUC'], marker='o', label='AUC')
plt.plot(results_df['Top_K_Features'], results_df['F1'], marker='s', label='F1 Score')

# 图形美化
plt.title('Model Performance vs. Number of Metabolite Features')
plt.xlabel('Number of Top Features (Top-K)')
plt.ylabel('Score')
plt.ylim(0.5, 0.8)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 选择 Top 15 特征
top_k = 15
top_features = results_df.loc[results_df['Top_K_Features'] == top_k, 'Features'].values[0]

# 准备数据
X_train_top15 = Xmet_train[top_features]
X_test_top15 = Xmet_test[top_features]

# 重新训练模型
model_top15, yscore_top15 = train_model(
    X_train_top15, y_train, groups_train,
    X_test_top15, y_test,
    name=f'Metabolites_Top{top_k}'
)

# 预测
y_pred_top15 = model_top15.predict(X_test_top15)

# 计算 accuracy
acc = accuracy_score(y_test, y_pred_top15)
print(f"Accuracy (Top {top_k} features): {acc:.3f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_top15)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix (Top {top_k} of Urinary Metabolites)\nAccuracy: {acc:.3f}')
plt.show()

# 加载代谢物名称对照表
mapping_df = pd.read_csv("D:/msc_projecct/QC_urine/QC_urine/KGCO-0504-20DSMLTA+ Urine Data Tables_Chemical Annotation.csv")  
mapping_df['Metabolite_ID'] = 'M' + mapping_df['COMP_ID'].astype(str)
# 创建字典：编号 → 化学名称
mapping_dict = dict(zip(mapping_df['Metabolite_ID'], mapping_df['CHEMICAL_NAME']))
# 提取前 15 个重要特征
top_feat_imp = feat_imp.head(15).copy()
# 将编号替换为化学名称（若找不到，则保留原编号）
top_feat_imp.index = top_feat_imp.index.map(lambda x: mapping_dict.get(x, x))
# 可视化
top_feat_imp.plot(kind='barh', title='Top 15 Important Metabolites by Random Forest')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# SHAP 初始化 explainer
explainer = shap.Explainer(model_top15, X_train_top15)
shap_values = explainer(X_train_top15)

# 映射代谢物名称
X_named = X_train_top15.copy()
X_named.columns = [mapping_dict.get(f, f) for f in X_named.columns]

print("Top15 原始列名（代谢物编号）:")
print(X_train_top15.columns.tolist())

print("\n映射字典中的键（示例）:")
print(list(mapping_dict.keys())[:5])

for mid in top_features:
    if mid not in mapping_dict:
        print(f"{mid} 不在映射表中 ❌")
    else:
        print(f"{mid} -> {mapping_dict[mid]} ✅")
        
 # 1. 构造名字映射后的列名列表
mapped_feature_names = [mapping_dict.get(f, f) for f in X_train_top15.columns]

# 2. 构造新的 Explanation 对象（使用 class=1 的 shap 值）
shap_exp = shap.Explanation(
    values=shap_values[..., 1].values,
    base_values=shap_values[..., 1].base_values,
    data=X_train_top15.values,
    feature_names=mapped_feature_names
)

# 3. 绘制 beeswarm 图（按真实分类着色）
shap.summary_plot(
    shap_exp,
    color=y_train.values,
    plot_type="dot",
    plot_size=(10, 6),
    show=True
)


# #加入cov
# results = []

# for top_k in range(5, 41, 5):
#     top_features = feat_imp.head(top_k).index

#     # ====== 1. 仅代谢物 ======
#     X_train_top = Xmet_train[top_features]
#     X_test_top = Xmet_test[top_features]

#     model_met, yscore_met = train_model(
#         X_train_top, y_train, groups_train,
#         X_test_top, y_test,
#         name=f'Metabolites_Top{top_k}'
#     )

#     y_pred_met = model_met.predict(X_test_top)
#     f1_met = f1_score(y_test, y_pred_met)
#     auc_met = roc_auc_score(y_test, yscore_met)
#     recall_met = recall_score(y_test, y_pred_met)

#     results.append({
#         'Model': 'MetabolitesOnly',
#         'Top_K_Features': top_k,
#         'F1': round(f1_met, 3),
#         'AUC': round(auc_met, 3),
#         'Recall': round(recall_met, 3),
#         'Features': list(top_features)
#     })

    # # ====== 2. 代谢物 + 协变量 ======
    # Xcomb_train_top = pd.concat([Xcov_train, X_train_top], axis=1)
    # Xcomb_test_top = pd.concat([Xcov_test, X_test_top], axis=1)

    # model_comb, yscore_comb = train_model(
    #     Xcomb_train_top, y_train, groups_train,
    #     Xcomb_test_top, y_test,
    #     name=f'Metab+Cov_Top{top_k}'
    # )

    # y_pred_comb = model_comb.predict(Xcomb_test_top)
    # f1_comb = f1_score(y_test, y_pred_comb)
    # auc_comb = roc_auc_score(y_test, yscore_comb)
    # recall_comb = recall_score(y_test, y_pred_comb)

    # results.append({
    #     'Model': 'Metab+Covariates',
    #     'Top_K_Features': top_k,
    #     'F1': round(f1_comb, 3),
    #     'AUC': round(auc_comb, 3),
    #     'Recall': round(recall_comb, 3),
    #     'Features': list(Xcomb_train_top.columns)
    # })

# # 查看结果
# results_df = pd.DataFrame(results)
# print("\n===== 模型性能对比（是否加入协变量）=====")
# print(results_df[['Model', 'Top_K_Features', 'F1', 'AUC', 'Recall']])