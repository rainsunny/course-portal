# Day 5: 正则化理论

## 核心问题：如何让模型学会"克制"？

前四天我们学习了训练和评估模型。但有一个核心问题：**如何防止模型"学太多"？**

这听起来矛盾——我们不是希望模型学习吗？但问题在于：**学习"规律"是好事，学习"噪声"是坏事**。

正则化就是让模型学会"克制"的技术——只学习真正的规律，忽略数据中的噪声。

---

## 第一部分：过拟合的本质回顾

### 1.1 过拟合的再认识

**过拟合定义**：模型在训练数据上表现很好，但在新数据上表现差。

**为什么会过拟合？**

假设真实关系：
$$y = f(x) + \epsilon$$

其中 $f(x)$ 是真实规律，$\epsilon$ 是噪声。

**模型学到的是**：
$$\hat{y} = \hat{f}(x)$$

**目标**：$\hat{f}(x) \approx f(x)$（学习规律）

**过拟合时**：$\hat{f}(x)$ 不仅拟合了 $f(x)$，还拟合了 $\epsilon$（学习噪声）

**例子**：

真实关系：$y = 2x + \epsilon$

训练数据（含噪声）：
- $(1, 2.1)$
- $(2, 4.3)$
- $(3, 5.8)$

**正常模型**：$\hat{y} = 2x$（接近真实关系）

**过拟合模型**：$\hat{y} = 2x + 0.1(x-1)(x-2)(x-3)$（完美拟合训练点，但扭曲了真实关系）

### 1.2 模型复杂度与过拟合

**模型复杂度**：模型能够表达的函数的复杂程度。

- **低复杂度**：线性模型
- **高复杂度**：高阶多项式、深度神经网络

**复杂度与过拟合的关系**：

| 模型复杂度 | 训练误差 | 测试误差 | 过拟合风险 |
|-----------|---------|---------|-----------|
| 低 | 高 | 高 | 低（欠拟合）|
| 适中 | 低 | 低 | 适中 |
| 高 | 很低 | 高 | 高（过拟合）|

**直觉**：

- 低复杂度模型"能力有限"，无法学习复杂规律 → 欠拟合
- 高复杂度模型"能力太强"，连噪声都学 → 过拟合
- 适中复杂度模型"能力刚好"，学到规律但不学噪声 → 泛化好

### 1.3 控制复杂度的方法

**方法1：限制假设空间**

- 选择简单模型（线性模型 vs 高阶多项式）
- 限制参数数量（神经网络宽度、深度）

**方法2：限制参数范围**

- 正则化：限制参数不能太大
- 这是今天的主角

**方法3：数据增强**

- 增加数据量，让噪声"平均掉"

**方法4：早停（Early Stopping）**

- 在验证误差开始上升时停止训练

---

## 第二部分：正则化的数学本质

### 2.1 正则化基本思想

**原始优化问题**：

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i)$$

**正则化优化问题**：

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \left[ \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i) + \lambda \Omega(\mathbf{w}) \right]$$

其中：
- $\mathcal{L}$：损失函数（如MSE、交叉熵）
- $\Omega(\mathbf{w})$：正则化项（惩罚复杂参数）
- $\lambda$：正则化系数（控制惩罚强度）

**直觉**：

- 损失项：希望拟合数据
- 正则化项：希望参数简单
- $\lambda$：两者的权衡

### 2.2 L2正则化（Ridge）

**定义**：

$$\Omega(\mathbf{w}) = \|\mathbf{w}\|_2^2 = \sum_{j=1}^{d} w_j^2$$

**完整优化问题**：

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \left[ \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i) + \lambda \sum_{j=1}^{d} w_j^2 \right]$$

**几何解释**：

L2正则化限制参数在以原点为球心的球内：

$$\sum_{j=1}^{d} w_j^2 \leq t$$

**梯度**：

$$\frac{\partial}{\partial w_j}\left(\text{Loss} + \lambda \sum_j w_j^2\right) = \frac{\partial \text{Loss}}{\partial w_j} + 2\lambda w_j$$

**梯度下降更新**：

$$w_j^{(t+1)} = w_j^{(t)} - \eta \frac{\partial \text{Loss}}{\partial w_j} - 2\eta\lambda w_j^{(t)}$$
$$= (1 - 2\eta\lambda)w_j^{(t)} - \eta \frac{\partial \text{Loss}}{\partial w_j}$$

**权重衰减（Weight Decay）**：

注意更新公式中的 $(1 - 2\eta\lambda)$ 项。每次更新，参数都会"收缩"一点。这就是"权重衰减"名称的由来。

**L2正则化的效果**：

1. **参数收缩**：所有参数都向0靠近，但不为0
2. **平滑性**：惩罚大参数，模型更平滑
3. **稳定性**：减少参数的波动

### 2.3 L1正则化（Lasso）

**定义**：

$$\Omega(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum_{j=1}^{d} |w_j|$$

**完整优化问题**：

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \left[ \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i) + \lambda \sum_{j=1}^{d} |w_j| \right]$$

**几何解释**：

L1正则化限制参数在一个菱形（高维是"钻石"形状）内：

$$\sum_{j=1}^{d} |w_j| \leq t$$

**L1 vs L2 几何形状**：

在二维情况下：

- **L2约束区域**：圆形 $\{(w_1, w_2) : w_1^2 + w_2^2 \leq t\}$
- **L1约束区域**：菱形 $\{(w_1, w_2) : |w_1| + |w_2| \leq t\}$

**关键洞察**：L1约束区域有"尖角"（坐标轴上的点），L2约束区域是圆滑的。

**L1正则化的特殊效果：稀疏性**

当约束区域（菱形）与损失等高线相切时，切点更可能落在坐标轴上。这意味着某些参数恰好为0。

**为什么产生稀疏性？**

**几何解释**：

L1约束区域有"尖角"（在坐标轴上）。损失等高线与约束区域相切时，更容易碰到这些尖角。尖角意味着某些参数为0。

**数学解释**（次梯度）：

$|w|$ 在 $w=0$ 处不可微，其次梯度是区间 $[-1, 1]$。

优化条件：
$$\frac{\partial \text{Loss}}{\partial w_j} + \lambda \cdot \text{sign}(w_j) = 0$$

当 $\left|\frac{\partial \text{Loss}}{\partial w_j}\right| < \lambda$ 时，$w_j = 0$ 可以满足次梯度条件。

**L1正则化的效果**：

1. **稀疏性**：某些参数恰好为0
2. **特征选择**：非零参数对应的特征被"选中"
3. **可解释性**：模型更简单，易于解释

### 2.4 L1 vs L2：对比总结

| 特性 | L1正则化 | L2正则化 |
|------|---------|---------|
| 公式 | $\lambda \sum \|w_j\|$ | $\lambda \sum w_j^2$ |
| 几何形状 | 菱形（有尖角） | 圆形（光滑） |
| 参数值 | 许多恰好为0 | 所有参数收缩但不为0 |
| 特征选择 | 有（自动选择） | 无 |
| 解的唯一性 | 可能不唯一 | 唯一 |
| 解析解 | 通常没有 | 有（线性回归） |
| 计算复杂度 | 较高（需迭代） | 较低 |

**什么时候用L1？**

- 需要特征选择
- 相信只有少数特征重要
- 想要稀疏模型

**什么时候用L2？**

- 不需要特征选择
- 所有特征可能都有用
- 计算效率重要

**弹性网络（Elastic Net）**：

结合L1和L2：

$$\Omega(\mathbf{w}) = \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

既有稀疏性，又有稳定性。

### 2.5 线性回归的正则化解

**线性回归 + L2正则化（Ridge回归）**：

原始问题：
$$J(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2$$

解析解：
$$\mathbf{w}^* = (X^TX + \lambda I)^{-1} X^T \mathbf{y}$$

**为什么 $(X^TX + \lambda I)$ 总是可逆？**

$X^TX$ 是半正定矩阵，可能不可逆。加上 $\lambda I$ 后：

$$X^TX + \lambda I$$

对于任意非零向量 $\mathbf{v}$：
$$\mathbf{v}^T(X^TX + \lambda I)\mathbf{v} = \mathbf{v}^T X^TX \mathbf{v} + \lambda \|\mathbf{v}\|^2 \geq \lambda \|\mathbf{v}\|^2 > 0$$

因此 $X^TX + \lambda I$ 是正定矩阵，可逆。

**正则化的另一个好处**：解决了矩阵不可逆的问题！

---

## 第三部分：贝叶斯视角——正则化即先验

### 3.1 最大后验估计（MAP）

**回顾最大似然估计（MLE）**：

$$\mathbf{w}_{MLE} = \arg\max_{\mathbf{w}} P(\mathcal{D}|\mathbf{w}) = \arg\min_{\mathbf{w}} -\log P(\mathcal{D}|\mathbf{w})$$

**最大后验估计（MAP）**：

$$\mathbf{w}_{MAP} = \arg\max_{\mathbf{w}} P(\mathbf{w}|\mathcal{D}) = \arg\max_{\mathbf{w}} \frac{P(\mathcal{D}|\mathbf{w})P(\mathbf{w})}{P(\mathcal{D})}$$

由于 $P(\mathcal{D})$ 与 $\mathbf{w}$ 无关：

$$\mathbf{w}_{MAP} = \arg\max_{\mathbf{w}} P(\mathcal{D}|\mathbf{w})P(\mathbf{w})$$

取对数：

$$\mathbf{w}_{MAP} = \arg\min_{\mathbf{w}} \left[ -\log P(\mathcal{D}|\mathbf{w}) - \log P(\mathbf{w}) \right]$$

**关键洞察**：

- $-\log P(\mathcal{D}|\mathbf{w})$：损失函数（似然）
- $-\log P(\mathbf{w})$：正则化项（先验）

**正则化 = 对参数施加先验分布！**

### 3.2 L2正则化 = 高斯先验

**假设**：参数 $\mathbf{w}$ 服从高斯先验：

$$P(\mathbf{w}) = \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{w_j^2}{2\sigma^2}\right)$$

取对数：

$$-\log P(\mathbf{w}) = \frac{d}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{j=1}^{d} w_j^2$$

忽略常数项：

$$-\log P(\mathbf{w}) \propto \frac{1}{2\sigma^2} \|\mathbf{w}\|_2^2$$

**结论**：

高斯先验 → L2正则化

正则化系数 $\lambda = \frac{1}{2\sigma^2}$

**先验的含义**：

高斯先验表示：我们认为参数应该"在0附近"，大部分参数值较小。

$\sigma$ 越小（$\lambda$ 越大）：先验越强，参数越接近0。

### 3.3 L1正则化 = 拉普拉斯先验

**假设**：参数 $\mathbf{w}$ 服从拉普拉斯先验：

$$P(\mathbf{w}) = \prod_{j=1}^{d} \frac{1}{2b} \exp\left(-\frac{|w_j|}{b}\right)$$

**拉普拉斯分布**：

$$p(w) = \frac{1}{2b} e^{-|w|/b}$$

与高斯分布相比，拉普拉斯分布在0处有更高的峰值，尾部更"重"。

取对数：

$$-\log P(\mathbf{w}) = d \log(2b) + \frac{1}{b} \sum_{j=1}^{d} |w_j|$$

忽略常数项：

$$-\log P(\mathbf{w}) \propto \frac{1}{b} \|\mathbf{w}\|_1$$

**结论**：

拉普拉斯先验 → L1正则化

正则化系数 $\lambda = \frac{1}{b}$

**先验的含义**：

拉普拉斯先验在0处有很高的概率密度，表示：我们认为"大部分参数应该恰好为0"。

### 3.4 先验选择的直觉

**高斯先验（L2）**：

- 形状：钟形，平滑
- 含义：参数应该在0附近，但不必为0
- 效果：参数收缩，但不稀疏

**拉普拉斯先验（L1）**：

- 形状：尖峰在0，重尾
- 含义：参数应该为0，除非有强证据
- 效果：参数稀疏

**图示**：

```
概率密度
    |
    |     拉普拉斯（尖峰）
    |    /|\
    |   / | \
    |  /  |  \_____
    | /   |       \____  高斯（平滑）
    |/____|____________\____
    0     参数值
```

### 3.5 贝叶斯视角的深刻含义

**正则化的本质**：引入先验知识，限制参数空间。

**奥卡姆剃刀**：在数据拟合相同的情况下，偏好更简单的模型。

**贝叶斯解释**：

- 简单模型 = 参数空间受限 = 高先验概率
- 复杂模型 = 参数空间宽泛 = 低先验概率

正则化就是数学化的奥卡姆剃刀。

---

## 第四部分：Dropout——深度学习的正则化利器

### 4.1 Dropout的基本思想

**问题**：深度神经网络参数量大，容易过拟合。

**Dropout**：在训练时随机"丢弃"一部分神经元。

**算法**：

对于每一层，训练时：
1. 以概率 $p$ 随机将神经元输出置为0
2. 其余神经元输出乘以 $\frac{1}{1-p}$（保持期望不变）

测试时：
- 所有神经元都保留
- 输出乘以 $p$（或训练时使用 inverted dropout）

### 4.2 Dropout的数学表达

**前向传播（训练时）**：

$$\mathbf{h}' = \mathbf{h} \odot \mathbf{m}, \quad m_j \sim \text{Bernoulli}(1-p)$$

其中 $\mathbf{m}$ 是掩码向量，$m_j \in \{0, 1\}$。

**期望保持**：

$$\mathbb{E}[\mathbf{h}'] = (1-p) \mathbf{h}$$

为了保持期望不变，训练时可以：

$$\mathbf{h}' = \frac{\mathbf{h} \odot \mathbf{m}}{1-p}$$

这样测试时直接使用 $\mathbf{h}$ 即可。

### 4.3 Dropout为什么有效？

#### 解释一：模型集成

**直觉**：Dropout相当于训练了 $2^n$ 个子网络（$n$ 是神经元数）。

每次训练，随机丢弃不同的神经元，相当于训练一个不同的子网络。

测试时，使用所有神经元，相当于对所有子网络取平均。

**形式化**：

设子网络集合为 $\mathcal{S}$，每个子网络 $s \in \mathcal{S}$ 的参数为 $\mathbf{w}_s$。

集成预测：
$$\hat{y} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} f_s(\mathbf{x}; \mathbf{w}_s)$$

Dropout近似了这个集成。

#### 解释二：防止协同适应

**协同适应（Co-adaptation）**：神经元之间形成复杂的依赖关系，某个神经元只在其他特定神经元存在时才有用。

**Dropout的效果**：迫使每个神经元都能独立工作，不能过度依赖其他神经元。

**类比**：公司里，如果员工A总是依赖员工B完成工作，当B不在时A就无法工作。Dropout相当于随机让员工休假，迫使每个人都学会独立完成任务。

#### 解释三：贝叶斯近似

**更深刻的理解**：Dropout可以看作贝叶斯神经网络的近似。

贝叶斯神经网络中，参数是分布而非固定值。Dropout通过随机丢弃，近似了参数的不确定性。

预测时，多次前向传播（MC Dropout）可以得到预测的不确定性估计。

### 4.4 Dropout与其他正则化的关系

**Dropout vs L2**：

- L2：参数层面的正则化，惩罚大参数
- Dropout：结构层面的正则化，破坏神经元依赖

**两者可以结合使用**。

**Dropout vs 数据增强**：

- 数据增强：增加数据多样性
- Dropout：增加模型随机性

**两者可以结合使用**。

### 4.5 Dropout的实践建议

**丢弃率选择**：

- 输入层：$p \approx 0.1 - 0.2$（不要丢弃太多输入信息）
- 隐藏层：$p \approx 0.5$（常用值）

**何时使用**：

- 大网络、数据少：Dropout效果好
- 小网络、数据多：可能不需要Dropout

**不要在测试时使用Dropout**：

测试时使用完整网络，输出保持稳定。

### 4.6 Dropout的理论分析

**正则化效果**：

研究表明，Dropout近似于L2正则化，但对不同参数有不同的正则化强度。

具体地，Dropout等价于：

$$\Omega(\mathbf{w}) \propto \|\mathbf{w}\|_2^2$$

但系数与该神经元的输入方差相关。

**自适应正则化**：输入方差大的神经元，正则化更强。

---

## 第五部分：正则化的选择策略

### 5.1 正则化强度 $\lambda$ 的选择

**验证集选择**：

1. 尝试多个 $\lambda$ 值（如 $10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10$）
2. 在验证集上评估每个 $\lambda$ 的表现
3. 选择验证误差最小的 $\lambda$

**学习曲线**：

绘制训练误差和验证误差随 $\lambda$ 变化的曲线：

- $\lambda$ 太小：训练误差低，验证误差高（过拟合）
- $\lambda$ 适中：训练误差和验证误差都低
- $\lambda$ 太大：训练误差和验证误差都高（欠拟合）

### 5.2 何时使用哪种正则化？

| 场景 | 推荐正则化 | 原因 |
|------|-----------|------|
| 特征很多，只有少数重要 | L1 | 自动特征选择 |
| 所有特征可能有用 | L2 | 稳定，有解析解 |
| 不确定 | Elastic Net | 结合两者优点 |
| 深度神经网络 | Dropout + L2 | 防止过拟合 |
| 数据很少 | 强正则化 | 防止过拟合 |
| 数据很多 | 弱正则化 | 数据本身防止过拟合 |

### 5.3 正则化与其他技术的配合

**正则化 + 早停**：

- 早停：在验证误差开始上升时停止训练
- 正则化：限制参数范围
- 两者可以结合使用

**正则化 + 数据增强**：

- 数据增强：增加数据多样性
- 正则化：限制模型复杂度
- 两者互补

**正则化 + Batch Normalization**：

- Batch Normalization 有正则化效果
- 可以减少对Dropout的依赖

---

## 第六部分：代码示例

### 6.1 L1/L2正则化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 生成数据
np.random.seed(42)
n_samples = 30
X = np.random.uniform(0, 1, n_samples).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X).ravel()
y = y_true + np.random.normal(0, 0.1, n_samples)

# 测试数据
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_true = np.sin(2 * np.pi * X_test).ravel()

# 模型设置
degree = 15  # 高阶多项式，容易过拟合

# 无正则化
model_no_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])

# L2正则化
model_l2 = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('ridge', Ridge(alpha=0.1))
])

# L1正则化
model_l1 = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('lasso', Lasso(alpha=0.001, max_iter=10000))
])

# 训练
model_no_reg.fit(X, y)
model_l2.fit(X, y)
model_l1.fit(X, y)

# 预测
y_pred_no_reg = model_no_reg.predict(X_test)
y_pred_l2 = model_l2.predict(X_test)
y_pred_l1 = model_l1.predict(X_test)

# 绘图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X, y, c='red', s=20, label='训练数据')
plt.plot(X_test, y_test_true, 'g--', label='真实函数')
plt.plot(X_test, y_pred_no_reg, 'b-', label='无正则化')
plt.ylim(-2, 2)
plt.title('无正则化（过拟合）')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X, y, c='red', s=20, label='训练数据')
plt.plot(X_test, y_test_true, 'g--', label='真实函数')
plt.plot(X_test, y_pred_l2, 'b-', label='L2正则化')
plt.ylim(-2, 2)
plt.title('L2正则化（Ridge）')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(X, y, c='red', s=20, label='训练数据')
plt.plot(X_test, y_test_true, 'g--', label='真实函数')
plt.plot(X_test, y_pred_l1, 'b-', label='L1正则化')
plt.ylim(-2, 2)
plt.title('L1正则化（Lasso）')
plt.legend()

plt.tight_layout()
plt.show()

# 查看L1正则化的稀疏性
lasso_coefs = model_l1.named_steps['lasso'].coef_
print(f"L1正则化后非零系数数量: {np.sum(lasso_coefs != 0)} / {len(lasso_coefs)}")
```

### 6.2 正则化路径

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_path
from sklearn.datasets import make_regression

# 生成数据
X, y = make_regression(n_samples=100, n_features=10, n_informative=3, 
                       noise=10, random_state=42)

# L1正则化路径
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, alphas=np.logspace(-3, 1, 50))

# L2正则化路径
alphas_ridge, coefs_ridge = ridge_path(X, y, alphas=np.logspace(-3, 3, 50))

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# L1正则化路径
ax1.plot(np.log10(alphas_lasso), coefs_lasso.T)
ax1.set_xlabel('log10(lambda)')
ax1.set_ylabel('系数值')
ax1.set_title('L1正则化路径（Lasso）')
ax1.invert_xaxis()

# L2正则化路径
ax2.plot(np.log10(alphas_ridge), coefs_ridge.T)
ax2.set_xlabel('log10(lambda)')
ax2.set_ylabel('系数值')
ax2.set_title('L2正则化路径（Ridge）')
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
```

### 6.3 Dropout效果可视化

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 生成数据
n_samples = 200
X = np.random.uniform(-1, 1, (n_samples, 1)).astype(np.float32)
y = (X**3 + 0.1 * np.random.randn(n_samples, 1)).astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 定义网络
class NetWithoutDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练函数
def train(model, X, y, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses

# 训练模型
model_no_dropout = NetWithoutDropout()
model_with_dropout = NetWithDropout()

losses_no_dropout = train(model_no_dropout, X_tensor, y_tensor)
losses_with_dropout = train(model_with_dropout, X_tensor, y_tensor)

# 预测
X_test = torch.linspace(-1.5, 1.5, 100).reshape(-1, 1)
model_no_dropout.eval()
model_with_dropout.eval()

with torch.no_grad():
    y_pred_no_dropout = model_no_dropout(X_test).numpy()
    y_pred_with_dropout = model_with_dropout(X_test).numpy()

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, s=10, label='训练数据')
plt.plot(X_test.numpy(), y_pred_no_dropout, 'r-', label='无Dropout')
plt.plot(X_test.numpy(), y_pred_with_dropout, 'b-', label='有Dropout')
plt.xlabel('x')
plt.ylabel('y')
plt.title('拟合结果')
plt.legend()
plt.xlim(-1.5, 1.5)

plt.subplot(1, 2, 2)
plt.plot(losses_no_dropout, label='无Dropout')
plt.plot(losses_with_dropout, label='有Dropout')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失')
plt.legend()

plt.tight_layout()
plt.show()
```
---

## 第七部分：核心直觉总结

### 7.1 正则化的本质

**三个等价视角**：

| 视角 | 表达式 | 含义 |
|------|--------|------|
| 优化视角 | $\min \text{Loss} + \lambda\Omega(\mathbf{w})$ | 惩罚复杂参数 |
| 约束视角 | $\min \text{Loss}$ s.t. $\Omega(\mathbf{w}) \leq t$ | 限制参数范围 |
| 贝叶斯视角 | $\max P(\mathbf{w}\|\mathcal{D})$ | 参数先验 |

### 7.2 L1 vs L2的本质区别

**几何视角**：

- L1约束区域有"尖角"，容易碰到坐标轴 → 稀疏解
- L2约束区域是圆滑的，不会碰到坐标轴 → 非稀疏解

**先验视角**：

- L1：拉普拉斯先验，相信参数应该为0
- L2：高斯先验，相信参数应该小但不为0

**梯度视角**：

- L1梯度：$\lambda \cdot \text{sign}(w)$，常数梯度，容易推到0
- L2梯度：$2\lambda w$，梯度随 $w$ 变小，缓慢逼近0

### 7.3 Dropout的独特之处

**不是参数正则化，而是结构正则化**：

- L1/L2：直接惩罚参数
- Dropout：破坏神经元依赖，间接正则化

**集成视角**：

Dropout ≈ 训练 $2^n$ 个子网络并集成

**防协同适应**：

迫使每个神经元都能独立工作

### 7.4 正则化的选择原则

| 正则化 | 适用场景 | 效果 |
|--------|---------|------|
| L1 | 特征选择、稀疏模型 | 稀疏解 |
| L2 | 一般场景、稳定解 | 参数收缩 |
| Elastic Net | L1+L2的优点 | 稀疏+稳定 |
| Dropout | 深度网络 | 防过拟合、防协同适应 |

---

## 思考题

### 问题1：L1正则化为什么能产生稀疏解？

<details>
<summary>点击查看提示</summary>

从几何形状和梯度两个角度思考。L1约束区域有什么特殊形状？L1的梯度有什么特点？
</details>

<details>
<summary>点击查看答案</summary>

**答案：L1正则化的稀疏性来自几何形状和梯度特性的双重作用。**

**视角一：几何形状（约束区域）**

**优化问题的等价形式**：

$$\min_{\mathbf{w}} \text{Loss}(\mathbf{w}) \quad \text{s.t.} \quad \Omega(\mathbf{w}) \leq t$$

- L1约束：$|w_1| + |w_2| \leq t$ → 菱形（有尖角）
- L2约束：$w_1^2 + w_2^2 \leq t$ → 圆形（光滑）

**关键洞察**：

损失函数的等高线（椭圆）与约束区域相切。L1约束区域的尖角（在坐标轴上）更容易与等高线相切。

**图示**：

```
      w2
       |     
    \  |  /    
     \ | /      L1约束区域（菱形）
      \|/       
-------●------- w1
      /|\       
     / | \      
    /  |  \     

      w2
       |     
      /|\     
     / | \     L2约束区域（圆形）
    /  |  \    
   /   ●   \   
  /    |    \  
-------●------- w1
```

当等高线与L1菱形相切时，切点更可能落在坐标轴上（尖角处），意味着 $w_1 = 0$ 或 $w_2 = 0$。

**视角二：梯度特性**

**L1梯度**：

$$\frac{\partial}{\partial w}(\text{Loss} + \lambda|w|) = \frac{\partial \text{Loss}}{\partial w} + \lambda \cdot \text{sign}(w)$$

其中 $\text{sign}(w)$ 是符号函数：
- $w > 0$：$\text{sign}(w) = 1$
- $w < 0$：$\text{sign}(w) = -1$
- $w = 0$：次梯度 $[-1, 1]$

**关键**：L1的梯度是常数 $\lambda$（绝对值），与 $w$ 的大小无关。

**优化过程**：

当 $w > 0$ 时：
$$w^{(t+1)} = w^{(t)} - \eta \left(\frac{\partial \text{Loss}}{\partial w} + \lambda\right)$$

梯度下降的量包含恒定的 $\lambda$ 项，持续将 $w$ 推向0。

**当 $w$ 接近0时**：

次梯度条件：$w = 0$ 是最优解当且仅当
$$\left|\frac{\partial \text{Loss}}{\partial w}\right| \leq \lambda$$

这意味着，如果损失函数的梯度不够大，$w$ 会被推到恰好为0。

**对比L2梯度**：

$$\frac{\partial}{\partial w}(\text{Loss} + \lambda w^2) = \frac{\partial \text{Loss}}{\partial w} + 2\lambda w$$

L2梯度包含 $2\lambda w$，当 $w$ 接近0时，梯度也接近0，推力变小。因此 $w$ 会接近0，但不会恰好为0。

**视角三：先验分布**

**L1先验（拉普拉斯分布）**：

$$p(w) = \frac{1}{2b}e^{-|w|/b}$$

在 $w=0$ 处有尖峰，表示认为"大部分参数应该为0"。

**L2先验（高斯分布）**：

$$p(w) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-w^2/(2\sigma^2)}$$

在 $w=0$ 处平滑，表示认为"参数应该小但不一定为0"。

**总结**：

L1产生稀疏解的原因：
1. **几何**：约束区域有尖角，切点容易落在坐标轴上
2. **梯度**：梯度是常数，持续推力将参数推到0
3. **先验**：拉普拉斯先验在0处有高概率密度

</details>

### 问题2：正则化强度 $\lambda$ 越大越好吗？

<details>
<summary>点击查看答案</summary>

**答案：不是。正则化强度需要在欠拟合和过拟合之间权衡。**

**正则化强度的作用**：

| $\lambda$ | 训练误差 | 测试误差 | 状态 |
|-----------|---------|---------|------|
| 0 | 很低 | 高 | 过拟合 |
| 适中 | 低 | 低 | 最优 |
| 很大 | 高 | 高 | 欠拟合 |

**解释**：

**$\lambda = 0$（无正则化）**：

- 模型自由度最大
- 可以完美拟合训练数据（包括噪声）
- 过拟合风险高

**$\lambda$ 适中**：

- 参数被适度约束
- 学到规律但不学噪声
- 泛化性能最好

**$\lambda$ 很大**：

- 参数被强约束，接近0
- 模型几乎是常数预测
- 欠拟合

**极端情况**：

当 $\lambda \to \infty$ 时，$\mathbf{w} \to 0$。

- 线性回归：$\hat{y} = 0$（预测所有样本为0）
- 逻辑回归：$\hat{y} = 0.5$（预测所有样本概率为0.5）

模型完全失去表达能力。

**选择 $\lambda$ 的方法**：

**1. 验证集选择**：

尝试多个 $\lambda$ 值，选择验证误差最小的。

**2. 学习曲线**：

绘制训练误差和验证误差随 $\lambda$ 变化的曲线：

```
误差
  |\
  | \      训练误差
  |  \
  |   \___________
  |    \
  |     \        验证误差
  |      \   /\  
  |       \_/  \_______
  |___________________ λ
  0    适中    大
```

- $\lambda$ 小：训练误差低，验证误差高（过拟合）
- $\lambda$ 适中：训练误差和验证误差都低（最优）
- $\lambda$ 大：训练误差和验证误差都高（欠拟合）

**3. 交叉验证**：

对每个 $\lambda$ 进行交叉验证，选择平均验证误差最小的。

**4. 信息准则**（线性模型）：

- AIC（Akaike Information Criterion）
- BIC（Bayesian Information Criterion）

自动平衡拟合度和复杂度。

**正则化路径**：

可以绘制参数随 $\lambda$ 变化的路径，观察哪些参数先变为0（L1正则化）。

**总结**：

正则化强度 $\lambda$ 是超参数，需要通过验证集或交叉验证选择。过大导致欠拟合，过小导致过拟合，适中最优。

</details>

### 问题3：Dropout在训练和测试时的行为为什么不同？

<details>
<summary>点击查看答案</summary>

**答案：训练时引入随机性防止过拟合，测试时需要稳定的预测。**

**训练时的Dropout**：

每次前向传播，随机"丢弃"一部分神经元：

$$\mathbf{h}' = \mathbf{h} \odot \mathbf{m}, \quad m_j \sim \text{Bernoulli}(p)$$

**效果**：
- 每次训练不同的子网络
- 破坏神经元之间的协同适应
- 相当于训练了指数多个子网络

**测试时的问题**：

如果测试时也随机丢弃：
1. 每次预测结果不同，不稳定
2. 无法利用全部神经元的能力
3. 预测性能下降

**解决方案：缩放**

**方法1：训练时不缩放，测试时缩放**

训练时：$\mathbf{h}' = \mathbf{h} \odot \mathbf{m}$

测试时：$\mathbf{h}' = p \cdot \mathbf{h}$（缩放）

原理：训练时期望 $\mathbb{E}[\mathbf{h}'] = p \cdot \mathbf{h}$，测试时用期望值代替。

**方法2：Inverted Dropout（训练时缩放）**

训练时：$\mathbf{h}' = \frac{\mathbf{h} \odot \mathbf{m}}{p}$

测试时：$\mathbf{h}' = \mathbf{h}$（不缩放）

原理：训练时已经缩放，期望 $\mathbb{E}[\mathbf{h}'] = \mathbf{h}$，测试时直接使用。

**为什么测试时缩放？**

**期望匹配**：

设训练时神经元输出为 $h$，经过Dropout后：

$$\mathbb{E}[h'] = p \cdot h + (1-p) \cdot 0 = p \cdot h$$

测试时如果不缩放，输出为 $h$，与训练时期望不匹配。

测试时乘以 $p$：

$$h'_{test} = p \cdot h$$

与训练时期望匹配。

**为什么训练时缩放更好（Inverted Dropout）？**

训练时缩放的好处：
1. 测试时不需要额外操作
2. 测试时计算更简单
3. 不影响推理速度

**Dropout的集成视角**：

训练时，Dropout相当于训练了 $2^n$ 个子网络（$n$ 是神经元数）。

测试时，我们希望用所有子网络的"平均"预测。

理论上，对所有子网络取平均：

$$\hat{y} = \frac{1}{2^n} \sum_{s \in \mathcal{S}} f_s(\mathbf{x})$$

实际上，这近似等价于使用完整网络并缩放。

**MC Dropout（测试时也用Dropout）**：

有时，测试时也使用Dropout，进行多次预测：

```python
predictions = []
for _ in range(100):
    pred = model(x, dropout=True)  # 启用Dropout
    predictions.append(pred)
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)  # 不确定性估计
```

**用途**：估计预测的不确定性（Bayesian近似）。

**总结**：

训练时：随机丢弃 → 防止过拟合
测试时：保留所有神经元 + 缩放 → 稳定预测 + 匹配期望

</details>

### 问题4：为什么Batch Normalization有正则化效果？

<details>
<summary>点击查看答案</summary>

**答案：Batch Normalization通过批次统计量的噪声，起到类似Dropout的正则化作用。**

**Batch Normalization回顾**：

对于每个mini-batch，计算均值和方差：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

归一化：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

缩放和平移：

$$y_i = \gamma \hat{x}_i + \beta$$

**正则化效果来源**：

**1. 批次统计量的噪声**：

$\mu_B$ 和 $\sigma_B^2$ 是基于mini-batch估计的，与真实均值和方差存在偏差。

这种偏差是随机的，取决于mini-batch的组成。

因此，归一化后的 $\hat{x}_i$ 包含随机噪声：

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$\mu_B$ 和 $\sigma_B^2$ 的随机性 → $\hat{x}_i$ 的随机性 → 正则化

**与Dropout的比较**：

| 技术 | 噪声来源 | 噪声类型 |
|------|---------|---------|
| Dropout | 随机丢弃神经元 | 离散（0或缩放）|
| BN | 批次统计量估计误差 | 连续（高斯近似）|

**2. 训练/测试差异**：

训练时：使用批次统计量 $\mu_B, \sigma_B^2$

测试时：使用全局统计量（移动平均）$\mu_{global}, \sigma_{global}^2$

这种差异也起到正则化作用。

**3. 梯度噪声**：

由于BN依赖批次统计量，反向传播时梯度也包含噪声。

**正则化效果的程度**：

Batch Normalization的正则化效果通常比Dropout弱，但在某些情况下可以替代Dropout。

**实践中的选择**：

- 使用BN时，可以减少或不用Dropout
- 对于非常深的网络，BN + Dropout 可能过强
- 数据量小时，BN的正则化效果更明显

**理论分析**：

有研究表明，BN的正则化效果来自：
1. 噪声注入（类似Dropout）
2. 优化景观平滑（更容易优化）
3. 隐式权重归一化

**总结**：

Batch Normalization的正则化效果来自批次统计量的随机性，类似于给模型注入噪声。这种效果比Dropout弱，但在实践中常常足够。

</details>

---

## 今日要点

1. **正则化的本质**：惩罚复杂参数，限制模型容量，防止过拟合

2. **L2正则化**：参数收缩，权重衰减，高斯先验

3. **L1正则化**：稀疏解，特征选择，拉普拉斯先验

4. **贝叶斯视角**：正则化 = 参数先验，L2 ↔ 高斯，L1 ↔ 拉普拉斯

5. **Dropout**：结构正则化，训练时随机丢弃，测试时保留并缩放

6. **正则化选择**：根据场景选择L1/L2/Dropout，通过验证集选择强度

---

## 明日预告

Day 6 我们将深入决策树与集成学习：

- 信息增益与特征选择
- 决策树的构建原理
- Bagging vs Boosting
- 随机森林与梯度提升

决策树是非参数化方法的代表，集成学习则是"三个臭皮匠顶个诸葛亮"的数学实现。
