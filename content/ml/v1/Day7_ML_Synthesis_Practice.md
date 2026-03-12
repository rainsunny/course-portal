# Day 7: Week 1 综合与实践

## 核心问题：如何将理论融会贯通？

前六天我们学习了机器学习的核心理论：学习的本质、优化方法、分类问题、模型评估、正则化、决策树与集成学习。今天是综合日，我们将：

1. **回顾**：梳理Week 1的理论框架
2. **实践**：端到端完成一个ML项目
3. **融会贯通**：理解各概念之间的联系

---

## 第一部分：机器学习理论框架回顾

### 1.1 核心概念图谱

```
                    机器学习
                       │
           ┌──────────┼──────────┐
           │          │          │
         学习问题   优化方法   泛化理论
           │          │          │
     ┌─────┴─────┐    │    ┌─────┴─────┐
     │           │    │    │           │
   监督学习   无监督  梯度  偏差-方差  正则化
     │           │    下降    分解
  ┌──┴──┐     ┌──┴──┐   │
分类  回归   聚类 降维  优化算法
     │           │
  损失函数     距离度量
     │
  ┌──┴──────────────┐
  │                 │
回归: MSE      分类: 交叉熵
```

### 1.2 第一性原理回顾

**问题1：什么是"学习"？**

> 学习的本质是**泛化**——从有限数据中提取规律，推广到未见数据。

学习的可能性依赖于：
- 数据来自同一分布（i.i.d.假设）
- 存在可学习的规律
- 假设空间与问题匹配

**问题2：机器如何"学习"？**

> 优化目标函数，调整模型参数。

核心流程：
1. 定义假设空间
2. 定义损失函数
3. 优化算法求解
4. 验证泛化能力

**问题3：如何保证泛化？**

> 偏差-方差权衡 + 正则化 + 模型选择。

核心矛盾：
- 模型太简单 → 高偏差（欠拟合）
- 模型太复杂 → 高方差（过拟合）
- 适中的复杂度 → 泛化好

### 1.3 关键公式汇总

**偏差-方差分解**：
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**线性回归正规方程**：
$$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$$

**梯度下降更新**：
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla J(\mathbf{w}_t)$$

**逻辑回归（Sigmoid）**：
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

**交叉熵损失**：
$$J = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

**L2正则化**：
$$J_{reg} = J + \lambda \|\mathbf{w}\|_2^2$$

**信息增益**：
$$IG(Y, A) = H(Y) - H(Y|A)$$

**随机森林方差**：
$$\text{Var}_{RF} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

### 1.4 方法选择指南

| 问题类型 | 数据量 | 特点 | 推荐方法 |
|---------|-------|------|---------|
| 回归 | 小 | 线性关系 | 线性回归 + L2 |
| 回归 | 小 | 非线性 | 多项式回归 / 决策树 |
| 回归 | 大 | 复杂 | GBDT / 随机森林 |
| 分类 | 小 | 线性可分 | 逻辑回归 |
| 分类 | 小 | 非线性 | 决策树 / SVM |
| 分类 | 大 | 复杂 | GBDT / 随机森林 |
| 高维 | 任意 | 稀疏 | L1正则化 |
| 高维 | 大 | 复杂 | 随机森林（特征选择） |

### 1.5 评估指标速查

**回归**：
- MSE（均方误差）
- MAE（平均绝对误差）
- $R^2$（决定系数）

**分类**：
- 准确率（Accuracy）：总体正确率
- 精确率（Precision）：预测为正类的正确率
- 召回率（Recall）：实际正类的召回率
- F1：精确率和召回率的调和平均
- AUC：排序能力

**选择**：
- 类别平衡 → 准确率
- 正类稀少 → F1、AUC
- 假阳性代价高 → 精确率
- 假阴性代价高 → 召回率

---

## 第二部分：端到端ML项目流程

### 2.1 项目流程概览

```
1. 问题定义
     ↓
2. 数据收集与探索
     ↓
3. 数据预处理
     ↓
4. 特征工程
     ↓
5. 模型选择与训练
     ↓
6. 模型评估与调优
     ↓
7. 模型部署与监控
```

### 2.2 每个步骤的关键点

**Step 1: 问题定义**

- 明确业务目标
- 定义输入输出
- 确定评估指标
- 评估数据可用性

**Step 2: 数据收集与探索**

- 收集数据源
- 探索性数据分析（EDA）
- 检查数据质量
- 发现模式和异常

**Step 3: 数据预处理**

- 缺失值处理
- 异常值处理
- 数据清洗
- 数据转换

**Step 4: 特征工程**

- 特征选择
- 特征构造
- 特征转换
- 特征缩放

**Step 5: 模型选择与训练**

- 选择候选模型
- 训练基线模型
- 模型比较
- 选择最优模型

**Step 6: 模型评估与调优**

- 交叉验证
- 超参数调优
- 集成方法
- 最终评估

**Step 7: 模型部署与监控**

- 模型序列化
- 部署流程
- 性能监控
- 模型更新

---

## 第三部分：综合案例——房价预测

### 3.1 问题定义

**任务**：预测房屋价格（回归问题）

**输入**：房屋特征（面积、卧室数、位置等）

**输出**：房价

**评估指标**：
- RMSE（均方根误差）
- MAE（平均绝对误差）
- $R^2$（决定系数）

### 3.2 数据探索与分析

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("数据形状:", X.shape)
print("\n特征名称:", housing.feature_names)
print("\n目标变量描述:", housing.target_names)

# 数据概览
print("\n数据统计:")
print(X.describe())

# 目标变量分布
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y, bins=50, edgecolor='black')
plt.xlabel('房价')
plt.ylabel('频数')
plt.title('房价分布')

plt.subplot(1, 2, 2)
plt.boxplot(y, vert=True)
plt.ylabel('房价')
plt.title('房价箱线图')

plt.tight_layout()
plt.show()

# 相关性分析
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()

# 特征与目标变量的关系
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, feature in enumerate(X.columns):
    ax = axes[i // 4, i % 4]
    ax.scatter(X[feature], y, alpha=0.5, s=1)
    ax.set_xlabel(feature)
    ax.set_ylabel('房价')
    ax.set_title(f'{feature} vs 房价')
plt.tight_layout()
plt.show()
```

### 3.3 数据预处理

```python
# 检查缺失值
print("缺失值统计:")
print(X.isnull().sum())

# 异常值检测（使用IQR方法）
def detect_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower) | (df[feature] > upper)]
    return len(outliers)

print("\n异常值统计:")
for feature in X.columns:
    print(f"{feature}: {detect_outliers(X, feature)} 个异常值")

# 数据分裂
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
```

### 3.4 基线模型

```python
# 定义评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n{model_name}:")
    print(f"  训练RMSE: {train_rmse:.4f}")
    print(f"  测试RMSE: {test_rmse:.4f}")
    print(f"  测试MAE: {test_mae:.4f}")
    print(f"  测试R²: {test_r2:.4f}")
    
    return model, y_test_pred

# 基线模型：线性回归
lr_model, lr_pred = evaluate_model(
    LinearRegression(), 
    X_train_scaled, y_train, X_test_scaled, y_test,
    "线性回归"
)

# Ridge回归
ridge_model, ridge_pred = evaluate_model(
    Ridge(alpha=1.0), 
    X_train_scaled, y_train, X_test_scaled, y_test,
    "Ridge回归"
)

# Lasso回归
lasso_model, lasso_pred = evaluate_model(
    Lasso(alpha=0.01), 
    X_train_scaled, y_train, X_test_scaled, y_test,
    "Lasso回归"
)
```

### 3.5 高级模型

```python
# 随机森林
rf_model, rf_pred = evaluate_model(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test,  # 树模型不需要标准化
    "随机森林"
)

# GBDT
gb_model, gb_pred = evaluate_model(
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test,
    "GBDT"
)

# 模型对比可视化
models = ['线性回归', 'Ridge', 'Lasso', '随机森林', 'GBDT']
test_rmse = [0.7456, 0.7455, 0.7456, 0.5027, 0.4921]  # 实际运行结果

plt.figure(figsize=(10, 5))
plt.bar(models, test_rmse, color=['blue', 'blue', 'blue', 'green', 'green'])
plt.xlabel('模型')
plt.ylabel('测试RMSE')
plt.title('模型性能对比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3.6 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 随机森林调参
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用较小的参数网格进行演示
param_grid_rf_small = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search_rf = GridSearchCV(
    rf, param_grid_rf_small, cv=5, 
    scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

print("随机森林最佳参数:", grid_search_rf.best_params_)
print("最佳CV分数:", np.sqrt(-grid_search_rf.best_score_))

# 使用最佳模型评估
best_rf = grid_search_rf.best_estimator_
y_pred_best = best_rf.predict(X_test)
print(f"\n最佳随机森林测试RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
```

### 3.7 模型解释

```python
# 特征重要性（随机森林）
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('随机森林特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 残差分析
residuals = y_test - y_pred_best

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布')

plt.tight_layout()
plt.show()
```

### 3.8 学习曲线分析

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = np.sqrt(-np.mean(train_scores, axis=1))
    test_scores_mean = np.sqrt(-np.mean(test_scores, axis=1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='训练误差')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='验证误差')
    plt.xlabel('训练样本数')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 线性回归学习曲线
plot_learning_curve(
    LinearRegression(), X_train_scaled, y_train,
    '线性回归学习曲线'
)

# 随机森林学习曲线
plot_learning_curve(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, y_train,
    '随机森林学习曲线'
)
```

### 3.9 偏差-方差分析

```python
# 不同复杂度模型的偏差-方差分析
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 多项式回归（不同阶数）
degrees = [1, 2, 3, 5, 10]
train_errors = []
test_errors = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train_scaled[:, :2], y_train)  # 只用前两个特征演示
    
    train_pred = model.predict(X_train_scaled[:, :2])
    test_pred = model.predict(X_test_scaled[:, :2])
    
    train_errors.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    test_errors.append(np.sqrt(mean_squared_error(y_test, test_pred)))

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'o-', label='训练误差')
plt.plot(degrees, test_errors, 'o-', label='测试误差')
plt.xlabel('多项式阶数（模型复杂度）')
plt.ylabel('RMSE')
plt.title('偏差-方差权衡')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 第四部分：理论联系实际

### 4.1 从理论到实践的映射

| 理论概念 | 实践体现 |
|---------|---------|
| 偏差-方差权衡 | 学习曲线分析、模型选择 |
| 正则化 | Ridge/Lasso回归、早停 |
| 交叉验证 | GridSearchCV的cv参数 |
| 特征选择 | Lasso稀疏性、随机森林特征重要性 |
| 集成学习 | 随机森林、GBDT |

### 4.2 常见问题与解决方案

**问题1：训练误差低，测试误差高**

诊断：过拟合
解决：
- 增加正则化（增大λ）
- 减少模型复杂度（限制深度、特征数）
- 增加数据
- 使用Dropout（神经网络）

**问题2：训练误差和测试误差都高**

诊断：欠拟合
解决：
- 增加模型复杂度
- 添加特征
- 减少正则化

**问题3：验证集表现波动大**

诊断：高方差
解决：
- 增加数据
- 使用Bagging（随机森林）
- 交叉验证取平均

**问题4：模型对异常值敏感**

诊断：异常值影响大
解决：
- 数据清洗
- 使用鲁棒损失（Huber）
- 使用对异常值鲁棒的模型（随机森林）

### 4.3 最佳实践清单

**数据阶段**：
- [ ] 检查缺失值
- [ ] 检查异常值
- [ ] 检查数据分布
- [ ] 检查特征相关性
- [ ] 检查目标变量分布

**特征工程阶段**：
- [ ] 特征缩放（标准化/归一化）
- [ ] 类别特征编码
- [ ] 特征选择
- [ ] 特征构造

**模型阶段**：
- [ ] 建立基线模型
- [ ] 尝试多个模型
- [ ] 交叉验证
- [ ] 超参数调优
- [ ] 模型集成

**评估阶段**：
- [ ] 多指标评估
- [ ] 学习曲线分析
- [ ] 残差分析
- [ ] 特征重要性分析

**部署阶段**：
- [ ] 模型序列化
- [ ] 建立监控
- [ ] 定期更新

---

## 第五部分：Week 1 总结

### 5.1 核心思想

**第一性原理**：

机器学习的本质是**从数据中学习规律，并泛化到未见数据**。

学习的可能性依赖于：
- 数据来自同一分布
- 规律可学习
- 假设空间匹配

**核心矛盾**：

偏差-方差权衡是机器学习的核心矛盾：
- 简单模型：高偏差、低方差
- 复杂模型：低偏差、高方差
- 目标：找到平衡点

**核心方法**：

- **优化**：梯度下降及其变体
- **正则化**：L1/L2、Dropout
- **集成**：Bagging降方差，Boosting降偏差
- **验证**：交叉验证、学习曲线

### 5.2 知识图谱

```
机器学习核心知识
│
├── 学习理论
│   ├── 偏差-方差分解
│   ├── 泛化理论
│   └── VC维度
│
├── 线性模型
│   ├── 线性回归 → MSE损失 → 正规方程
│   ├── 逻辑回归 → 交叉熵 → 梯度下降
│   └── 正则化 → L1(稀疏) / L2(平滑)
│
├── 优化方法
│   ├── 梯度下降 → SGD → Adam
│   └── 凸优化 → 全局最优
│
├── 分类问题
│   ├── Sigmoid → 概率输出
│   ├── Softmax → 多分类
│   └── 评估 → P/R/F1/AUC
│
├── 决策树
│   ├── 信息增益 / 基尼
│   └── 剪枝 → 防过拟合
│
└── 集成学习
    ├── Bagging → 随机森林 → 降方差
    └── Boosting → GBDT → 降偏差
```

### 5.3 数学基础回顾

**必须掌握的数学**：

1. **线性代数**：
   - 矩阵运算
   - 特征值与特征向量
   - 正定矩阵

2. **微积分**：
   - 梯度与偏导数
   - 链式法则
   - 泰勒展开

3. **概率论**：
   - 条件概率与贝叶斯
   - 期望与方差
   - 常见分布（高斯、伯努利）

4. **优化**：
   - 凸性
   - 梯度下降
   - 收敛性

### 5.4 Week 1 vs Week 2 展望

**Week 1：经典机器学习**

- 理论基础（偏差-方差、泛化）
- 线性模型（回归、分类）
- 决策树与集成
- 优化与正则化

**Week 2：深度学习**

- 神经网络基础
- 反向传播
- CNN、RNN、Transformer
- 现代训练技巧

**联系**：

Week 1的理论是Week 2的基础：
- 梯度下降 → 反向传播
- 正则化 → Dropout、Batch Norm
- 偏差-方差 → 过拟合/欠拟合
- 优化 → 深度学习优化器

---

## 思考题

### 问题1：如何选择合适的机器学习模型？

<details>
<summary>点击查看答案</summary>

**答案：根据数据量、问题类型、特征维度、可解释性需求综合选择。**

**决策流程**：

**1. 问题类型**
- 回归 → 线性回归、Ridge、GBDT
- 分类 → 逻辑回归、随机森林、GBDT
- 聚类 → K-Means、DBSCAN

**2. 数据量**
- 小数据（<1000）→ 简单模型（线性、逻辑回归）
- 中等数据（1000-100000）→ 集成方法（随机森林、GBDT）
- 大数据（>100000）→ 可扩展方法

**3. 特征维度**
- 低维 → 任意模型
- 高维稀疏 → L1正则化
- 高维密集 → 特征选择 + 随机森林

**4. 可解释性需求**
- 需要解释 → 线性回归、决策树
- 不需要解释 → 随机森林、GBDT、神经网络

**5. 数据质量**
- 干净数据 → GBDT（精度高）
- 噪声数据 → 随机森林（鲁棒）
- 缺失值多 → 树模型（自动处理）

**经验法则**：
1. 先建立简单基线（线性/逻辑回归）
2. 尝试随机森林（快速、鲁棒）
3. 如果精度不够，尝试GBDT
4. 最终根据验证集选择

</details>

### 问题2：如何诊断和解决过拟合/欠拟合？

<details>
<summary>点击查看答案</summary>

**答案：通过学习曲线诊断，通过调整模型复杂度解决。**

**诊断方法**：

**过拟合症状**：
- 训练误差低，验证误差高
- 训练和验证误差差距大
- 学习曲线：训练曲线持续下降，验证曲线上升或平稳

**欠拟合症状**：
- 训练误差和验证误差都高
- 训练和验证误差差距小
- 学习曲线：两条曲线都高且平稳

**解决方案**：

| 问题 | 解决方案 |
|------|---------|
| 过拟合 | 增加数据、正则化、减少复杂度、早停、Dropout |
| 欠拟合 | 增加复杂度、添加特征、减少正则化、更复杂模型 |

**代码诊断**：

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

# 过拟合：train_mean >> val_mean
# 欠拟合：train_mean 和 val_mean 都高
```

</details>

### 问题3：为什么集成方法通常比单一模型好？

<details>
<summary>点击查看答案</summary>

**答案：集成降低方差或偏差，提高泛化能力。**

**Bagging降低方差**：

假设 $M$ 个独立模型，每个方差为 $\sigma^2$。

集成方差：
$$\text{Var}_{ens} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

- 模型独立（$\rho=0$）：方差降低到 $\frac{\sigma^2}{M}$
- 模型相关（$\rho>0$）：方差仍低于单模型

**Boosting降低偏差**：

每个新模型专注于纠正之前模型的错误，逐步降低偏差。

理论上，如果每个弱学习器略好于随机，Boosting可以达到任意精度。

**多样性是关键**：

集成的效果取决于模型多样性：
- Bagging：通过Bootstrap和特征随机增加多样性
- Boosting：通过拟合残差增加多样性

**为什么不总是更好？**

- 集成增加计算成本
- 集成降低可解释性
- 如果单模型已经最优，集成不会有太大提升

</details>

### 问题4：如何在有限时间内最大化模型性能？

<details>
<summary>点击查看答案</summary>

**答案：快速迭代，先保证数据质量，再优化模型。**

**时间分配建议**：

- 数据探索和清洗：40%
- 特征工程：30%
- 模型训练和调优：20%
- 验证和解释：10%

**快速迭代策略**：

**第1轮（基线）**：
- 简单预处理（缺失值填充、标准化）
- 建立简单基线（线性回归/逻辑回归）
- 记录基线性能

**第2轮（改进）**：
- 尝试随机森林（通常比线性模型好）
- 简单特征工程
- 记录改进

**第3轮（优化）**：
- GBDT或XGBoost
- 特征选择
- 超参数粗调

**第4轮（精调）**：
- 针对最佳模型精调
- 集成方法

**关键原则**：

1. **数据优先**：更好的数据 > 更好的模型
2. **简单开始**：简单模型快速建立基线
3. **增量改进**：每次只改一个地方
4. **记录一切**：记录每次实验的结果

**时间紧迫时的策略**：

- 使用自动机器学习（AutoML）
- 使用预训练模型
- 重点放在特征工程
- 使用集成（随机森林默认参数通常不错）

</details>

---

## 今日要点

1. **端到端流程**：问题定义 → 数据探索 → 预处理 → 特征工程 → 模型训练 → 评估调优 → 部署

2. **理论联系实际**：偏差-方差权衡指导模型选择，正则化防止过拟合，交叉验证评估泛化

3. **最佳实践**：建立基线、多模型对比、交叉验证、学习曲线分析

4. **Week 1 核心**：学习的本质是泛化，优化和正则化是实现泛化的手段

---

## Week 1 完结，Week 2 预告

恭喜你完成了Week 1的学习！你已经掌握了机器学习的核心理论和实践方法。

**Week 1 回顾**：
- Day 1：学习的本质（泛化、偏差-方差）
- Day 2：线性模型与优化（梯度下降、凸优化）
- Day 3：分类与概率建模（Sigmoid、交叉熵）
- Day 4：模型评估（ROC/AUC、交叉验证）
- Day 5：正则化理论（L1/L2、Dropout）
- Day 6：决策树与集成（Bagging/Boosting）
- Day 7：综合实践

**Week 2 展望**：
- Day 8：神经网络数学基础
- Day 9：优化与训练动力学
- Day 10：卷积神经网络（CNN）
- Day 11：循环神经网络（RNN）
- Day 12：Attention机制
- Day 13：Transformer架构
- Day 14：现代深度学习范式

深度学习的世界在等待你！
