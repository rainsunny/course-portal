# Day 2: 线性模型与优化理论

## 核心问题：机器如何"学习"参数？

昨天我们理解了"什么是学习"——从数据中提取规律并泛化。今天我们深入探讨一个更具体的问题：**机器如何"学习"？**

这个问题可以分解为：
1. 如何量化"错误"？
2. 如何寻找"最优"？
3. 如何保证找到"真正最优"？

线性模型是最简单的模型，但它包含了机器学习的所有核心要素。理解它，是理解复杂模型的基石。

---

## 第一部分：线性模型——从最简单开始

### 1.1 为什么从线性模型开始？

线性模型是最简单的有参数模型，但它揭示了机器学习的核心问题：

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + ... + w_d x_d = \mathbf{w}^T \mathbf{x} + b$$

**为什么它重要？**

1. **可解释性强**：每个权重 $w_i$ 直接表示特征 $x_i$ 的重要性
2. **计算高效**：有闭式解，可以精确求解
3. **理论基础完备**：凸优化理论保证了全局最优
4. **是复杂模型的基础**：神经网络本质上是非线性激活的线性组合叠加

### 1.2 线性回归的设定

**问题设定**：

给定训练数据 $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$，其中：
- $\mathbf{x}_i \in \mathbb{R}^d$ 是输入特征向量
- $y_i \in \mathbb{R}$ 是目标值（回归问题）

目标是找到参数 $\mathbf{w}$ 使得预测 $\hat{y}_i = \mathbf{w}^T \mathbf{x}_i$ 尽可能接近 $y_i$。

**矩阵形式**：

设计矩阵 $X \in \mathbb{R}^{n \times d}$（每行是一个样本），目标向量 $\mathbf{y} \in \mathbb{R}^n$：

$$X = \begin{pmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_n^T \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$$

预测：$\hat{\mathbf{y}} = X\mathbf{w}$

---

## 第二部分：损失函数——量化"错误"的数学语言

### 2.1 什么是损失函数？

**损失函数（Loss Function）** $\mathcal{L}(y, \hat{y})$ 将"预测错误"映射为一个标量值，量化"有多错"。

**核心问题**：为什么不能用简单的"对/错"（0-1损失）？

因为"对/错"不连续，无法优化。我们需要"连续"的损失函数来指导优化方向。

### 2.2 均方误差（MSE）——最自然的损失函数

$$\mathcal{L}_{MSE}(y, \hat{y}) = (y - \hat{y})^2$$

**为什么MSE是自然选择？**

#### 原因1：几何直觉——最小化距离

MSE就是预测值与真实值之间的欧氏距离平方。最小化MSE就是让预测"尽量接近"真实值。

#### 原因2：最大似然估计（统计学视角）

假设目标值与输入之间的关系为：

$$y = \mathbf{w}^T \mathbf{x} + \epsilon$$

其中 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 是高斯噪声。

那么给定 $\mathbf{x}$，$y$ 的条件分布为：

$$p(y|\mathbf{x}, \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mathbf{w}^T\mathbf{x})^2}{2\sigma^2}\right)$$

**最大似然估计**：找到使数据概率最大的参数。

$$\mathbf{w}_{MLE} = \arg\max_{\mathbf{w}} \prod_{i=1}^{n} p(y_i|\mathbf{x}_i, \mathbf{w})$$

取对数（单调性）：

$$\mathbf{w}_{MLE} = \arg\max_{\mathbf{w}} \sum_{i=1}^{n} \log p(y_i|\mathbf{x}_i, \mathbf{w})$$

$$= \arg\max_{\mathbf{w}} \sum_{i=1}^{n} \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2} \right]$$

忽略常数项：

$$= \arg\max_{\mathbf{w}} \sum_{i=1}^{n} -(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

$$= \arg\min_{\mathbf{w}} \sum_{i=1}^{n} (y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

**结论**：MSE损失等价于在高斯噪声假设下的最大似然估计！

**深刻含义**：使用MSE隐含了一个假设——预测误差服从高斯分布。

### 2.3 MSE的优良性质

**定义（凸函数）**：函数 $f$ 是凸的，当且仅当对任意 $\mathbf{x}, \mathbf{y}$ 和 $\lambda \in [0, 1]$：

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**几何直觉**：函数图像上任意两点的连线在函数图像之上。

**MSE是凸函数**：

对于线性回归，损失函数为：

$$J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2$$

展开：

$$J(\mathbf{w}) = \frac{1}{n}(\mathbf{y} - X\mathbf{w})^T(\mathbf{y} - X\mathbf{w}) = \frac{1}{n}\|\mathbf{y}\|^2 - \frac{2}{n}\mathbf{y}^TX\mathbf{w} + \frac{1}{n}\mathbf{w}^TX^TX\mathbf{w}$$

Hessian矩阵（二阶导数矩阵）：

$$\nabla^2 J(\mathbf{w}) = \frac{2}{n}X^TX$$

**关键**：$X^TX$ 是半正定矩阵（对任意向量 $\mathbf{v}$，$\mathbf{v}^T(X^TX)\mathbf{v} = \|X\mathbf{v}\|^2 \geq 0$）。

**因此**：$J(\mathbf{w})$ 是凸函数！

**凸函数的伟大意义**：
- **局部最优 = 全局最优**
- 任何优化算法找到的极值点都是全局最优
- 不用担心"陷入局部最优"

### 2.4 其他损失函数

MSE不是唯一选择。不同损失函数反映不同的假设和偏好：

| 损失函数 | 形式 | 适用场景 | 特点 |
|---------|------|---------|------|
| MSE | $(y-\hat{y})^2$ | 回归 | 对异常值敏感 |
| MAE | $\|y-\hat{y}\|$ | 回归 | 对异常值鲁棒 |
| Huber | 分段函数 | 回归 | 兼顾两者 |
| 交叉熵 | $-y\log\hat{y}$ | 分类 | 概率输出 |
| Hinge | $\max(0, 1-y\hat{y})$ | 分类 | SVM |

**MAE vs MSE**：

为什么MAE对异常值更鲁棒？

- MSE的梯度：$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$
- MAE的梯度：$\frac{\partial L}{\partial \hat{y}} = \text{sign}(\hat{y} - y)$

MSE的梯度与误差大小成正比，大误差会产生大梯度，导致模型过度调整以适应异常值。MAE的梯度恒为±1，不受误差大小影响。

### 2.5 损失函数设计的原则

**原则1：反映问题本质**

- 回归问题：预测值与真实值的"距离"
- 分类问题：预测概率与真实标签的"分歧"

**原则2：可优化**

- 连续（或几乎处处可微）
- 梯度有意义（不消失、不爆炸）

**原则3：凸性（理想但非必须）**

- 凸损失 → 全局最优保证
- 非凸损失 → 可能陷入局部最优（深度学习的常态）

---

## 第三部分：优化——寻找最优的数学

### 3.1 什么是优化？

**优化问题**：

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} J(\mathbf{w})$$

找到使损失函数最小的参数。

### 3.2 闭式解——精确求解（仅限特殊情况）

对于线性回归 + MSE损失，我们可以直接求导并令其为零：

$$\nabla J(\mathbf{w}) = 0$$

$$\nabla \left[\frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2\right] = 0$$

$$-\frac{2}{n}X^T(\mathbf{y} - X\mathbf{w}) = 0$$

$$X^TX\mathbf{w} = X^T\mathbf{y}$$

如果 $X^TX$ 可逆（$X$ 列满秩）：

$$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$$

这就是著名的**正规方程（Normal Equation）**。

**为什么大部分情况没有闭式解？**

1. **损失函数非线性**：如交叉熵损失，对数里面有参数
2. **模型非线性**：神经网络，激活函数使得无法直接求逆
3. **数据量大**：$X^TX$ 的逆矩阵计算复杂度 $O(d^3)$，维度高时不可行

**因此**：我们需要迭代优化方法——梯度下降。

### 3.3 梯度下降——最基础的优化方法

#### 核心思想：沿着"下坡"方向走

想象你站在山上，蒙着眼睛，想要下山。最自然的策略是：感受脚下哪个方向最陡，然后往那个方向走一步。

**数学表达**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla J(\mathbf{w}_t)$$

其中：
- $\mathbf{w}_t$：第 $t$ 步的参数
- $\eta$：学习率（步长）
- $\nabla J(\mathbf{w}_t)$：损失函数在当前点的梯度

#### 为什么是负梯度方向？

**定理（最速下降方向）**：在当前位置，负梯度方向是函数值下降最快的方向。

**证明**：

考虑当前位置 $\mathbf{w}$，沿方向 $\mathbf{d}$ 移动一小步 $\epsilon$，函数值变化为：

$$J(\mathbf{w} + \epsilon\mathbf{d}) \approx J(\mathbf{w}) + \epsilon \nabla J(\mathbf{w})^T \mathbf{d}$$

我们希望函数值下降最多，即最小化：

$$\min_{\|\mathbf{d}\|=1} \nabla J(\mathbf{w})^T \mathbf{d}$$

由柯西-施瓦茨不等式：

$$\nabla J(\mathbf{w})^T \mathbf{d} \geq -\|\nabla J(\mathbf{w})\| \cdot \|\mathbf{d}\| = -\|\nabla J(\mathbf{w})\|$$

等号成立当且仅当：

$$\mathbf{d} = -\frac{\nabla J(\mathbf{w})}{\|\nabla J(\mathbf{w})\|}$$

这正是负梯度方向（单位化）。

**因此**：负梯度方向是函数值下降最快的方向。

#### 梯度下降的几何理解

**一维情况**：

```
J(w)
 |     
 |  \      /
 |   \    /
 |    \  /
 |     \/____
 |     w*
 |
-------- w --->
```

- 在 $w < w^*$ 时，$\frac{dJ}{dw} < 0$，负梯度 > 0，向右移动，接近最优
- 在 $w > w^*$ 时，$\frac{dJ}{dw} > 0$，负梯度 < 0，向左移动，接近最优

**二维情况**：

等高线图上，梯度方向垂直于等高线，指向函数值增加最快的方向。负梯度指向"下山"方向。

### 3.4 学习率——控制"步子大小"

**学习率 $\eta$ 的作用**：

- $\eta$ 太小：收敛太慢，可能陷入局部最优
- $\eta$ 太大：可能震荡甚至发散
- $\eta$ 适中：稳定收敛

**可视化**：

```
学习率太小（η = 0.01）    学习率适中（η = 0.1）    学习率太大（η = 1.0）
                                                                  
    J(w)                      J(w)                      J(w)
     |                          |                          |
     |\      /                  |\      /                  |    /\
     | \    /                   | \~~~/                    |   /  \
     |  \  /                    |  \ /                     |  /    \
     |   \/                     |   V                      | /      \
     |                          |                          |/        \
     |-----------> t            |-----------> t            |-----------> t
     (慢慢爬行)                  (平稳下降)                 (震荡发散)
```

#### 如何选择学习率？

**常用策略**：

1. **试错法**：从较小的值（如0.01）开始，逐步增加直到不稳定
2. **学习率衰减**：
   $$\eta_t = \frac{\eta_0}{1 + \alpha t}$$
3. **自适应学习率**：Adam、RMSprop等算法自动调整

### 3.5 梯度下降的收敛性

**什么时候梯度下降能收敛？**

**定理（凸函数收敛）**：如果满足：
1. $J$ 是凸函数
2. $J$ 的梯度是Lipschitz连续的：$\|\nabla J(\mathbf{w}) - \nabla J(\mathbf{w}')\| \leq L\|\mathbf{w} - \mathbf{w}'\|$
3. 学习率 $\eta < \frac{1}{L}$

则梯度下降以 $O(1/t)$ 的速度收敛到全局最优。

**非凸函数的情况**：

对于非凸函数（如深度神经网络），梯度下降只能保证收敛到**临界点**（梯度为零的点）：
- 可能是局部最优
- 可能是全局最优
- 可能是鞍点

### 3.6 梯度下降的变体

#### 批量梯度下降（Batch GD）

使用全部数据计算梯度：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{n}\sum_{i=1}^{n}\nabla_\mathbf{w} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i)$$

**优点**：梯度精确，收敛稳定
**缺点**：数据量大时计算慢，内存占用大

#### 随机梯度下降（SGD）

每次使用一个样本：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i)$$

**优点**：计算快，能够"跳出"局部最优（因为梯度有噪声）
**缺点**：梯度噪声大，收敛路径不稳定

#### 小批量梯度下降（Mini-batch GD）

使用一小批样本（如32、64、128个）：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}}\nabla_\mathbf{w} \mathcal{L}(y_i, \mathbf{w}^T\mathbf{x}_i)$$

**优点**：平衡了精确性和计算效率
**缺点**：需要调整批量大小

**实践中最常用**：Mini-batch GD。

---

## 第四部分：凸优化——为什么线性回归能精确求解

### 4.1 凸集与凸函数

#### 凸集（Convex Set）

**定义**：集合 $\mathcal{C}$ 是凸集，当且仅当对任意 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ 和 $\lambda \in [0, 1]$：

$$\lambda \mathbf{x} + (1-\lambda)\mathbf{y} \in \mathcal{C}$$

**几何直觉**：集合内任意两点的连线仍在集合内。

**例子**：
- 凸集：直线、半平面、球、多面体
- 非凸集：月牙形、星形

#### 凸函数（Convex Function）

**定义**：函数 $f$ 在凸集 $\mathcal{C}$ 上是凸函数，当且仅当对任意 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ 和 $\lambda \in [0, 1]$：

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**几何直觉**：函数图像上任意两点的连线在函数图像之上。

### 4.2 凸函数的等价条件

**一阶条件**：$f$ 可微，则 $f$ 是凸函数当且仅当：

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$$

**含义**：函数值永远在其切线之上。

**二阶条件**：$f$ 二阶可微，则 $f$ 是凸函数当且仅当Hessian矩阵半正定：

$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

**含义**：函数的曲率非负（向上凹）。

### 4.3 凸优化的伟大性质

**性质1：局部最优 = 全局最优**

**证明**：

设 $\mathbf{x}^*$ 是局部最优（即存在邻域 $N$，对任意 $\mathbf{y} \in N$，$f(\mathbf{x}^*) \leq f(\mathbf{y})$）。

对于任意 $\mathbf{z}$（不一定是邻域内的点），取 $\lambda \in (0, 1)$ 足够小，使得 $\lambda \mathbf{z} + (1-\lambda)\mathbf{x}^* \in N$。

由凸性：

$$f(\mathbf{x}^*) \leq f(\lambda \mathbf{z} + (1-\lambda)\mathbf{x}^*) \leq \lambda f(\mathbf{z}) + (1-\lambda)f(\mathbf{x}^*)$$

整理得：

$$f(\mathbf{x}^*) \leq f(\mathbf{z})$$

**因此**：$\mathbf{x}^*$ 是全局最优。

**性质2：临界点 = 全局最优**

对于凸函数，$\nabla f(\mathbf{x}^*) = 0$ 当且仅当 $\mathbf{x}^*$ 是全局最优。

**证明**：

由一阶条件，对任意 $\mathbf{y}$：

$$f(\mathbf{y}) \geq f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^T(\mathbf{y} - \mathbf{x}^*) = f(\mathbf{x}^*)$$

**性质3：解的唯一性**

如果 $f$ 是**严格凸函数**（不等式严格成立），则全局最优唯一。

### 4.4 为什么凸优化是机器学习的基石？

| 凸优化 | 非凸优化 |
|--------|----------|
| 局部最优 = 全局最优 | 可能有多个局部最优 |
| 梯度下降保证收敛到全局最优 | 梯度下降可能陷入局部最优 |
| 理论保证完整 | 理论分析困难 |
| 可以高效求解 | 求解困难 |

**线性回归是凸优化问题**：

$$J(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2$$

- 这是 $\mathbf{w}$ 的二次函数
- Hessian矩阵 $X^TX \succeq 0$（半正定）
- 因此 $J(\mathbf{w})$ 是凸函数

**深度学习是非凸优化问题**：

神经网络的损失函数通常是非凸的：
- 多个局部最优
- 大量鞍点
- 优化困难

**为什么深度学习能工作？**

1. 高维空间中，局部最优往往已经"足够好"
2. 随机梯度下降的噪声有助于跳出局部最优
3. 过参数化使得存在很多"好"的解

---

## 第五部分：一个完整的例子——线性回归

让我们从头到尾实现一个线性回归，理解每个步骤。

### 5.1 问题设定

假设真实关系：

$$y = 2x_1 + 3x_2 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.5^2)$$

真实参数：$w_1^* = 2, w_2^* = 3$

### 5.2 生成数据

```python
import numpy as np

# 设置随机种子，保证可复现
np.random.seed(42)

# 生成数据
n = 100  # 样本数
d = 2    # 特征维度

X = np.random.randn(n, d)  # 标准正态分布
true_w = np.array([2, 3])
epsilon = np.random.randn(n) * 0.5
y = X @ true_w + epsilon

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

### 5.3 方法一：闭式解

```python
# 正规方程: w* = (X^T X)^(-1) X^T y
w_closed = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"闭式解: w = {w_closed}")
print(f"真实值: w = {true_w}")
print(f"误差: {np.linalg.norm(w_closed - true_w):.4f}")
```

### 5.4 方法二：梯度下降

```python
def loss_function(w, X, y):
    """计算MSE损失"""
    return np.mean((X @ w - y) ** 2)

def gradient(w, X, y):
    """计算梯度"""
    n = len(y)
    return 2 * X.T @ (X @ w - y) / n

def gradient_descent(X, y, lr=0.1, epochs=100):
    """梯度下降"""
    n, d = X.shape
    w = np.zeros(d)  # 初始化
    
    losses = []
    for epoch in range(epochs):
        grad = gradient(w, X, y)
        w = w - lr * grad
        loss = loss_function(w, X, y)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}, w = {w}")
    
    return w, losses

w_gd, losses = gradient_descent(X, y, lr=0.1, epochs=100)
print(f"\n梯度下降结果: w = {w_gd}")
```

### 5.5 比较结果

```python
print("=== 结果比较 ===")
print(f"真实参数:     w = {true_w}")
print(f"闭式解:       w = {w_closed}")
print(f"梯度下降:     w = {w_gd}")
print(f"\n闭式解误差:   {np.linalg.norm(w_closed - true_w):.6f}")
print(f"梯度下降误差: {np.linalg.norm(w_gd - true_w):.6f}")
```

### 5.6 学习率的影响

```python
import matplotlib.pyplot as plt

learning_rates = [0.01, 0.1, 0.5]
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, losses = gradient_descent(X, y, lr=lr, epochs=50)
    plt.plot(losses, label=f'lr={lr}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同学习率的收敛曲线')
plt.legend()
plt.show()
```

---

## 第六部分：优化算法的演进

### 6.1 动量法（Momentum）

**问题**：梯度下降在"峡谷"形状的损失景观中震荡。

**思想**：引入"惯性"，累积历史梯度。

**更新规则**：

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla J(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

**直观理解**：像一个球在山上滚，有惯性，不会轻易改变方向。

### 6.2 自适应学习率：RMSprop

**问题**：不同参数需要不同的学习率。

**思想**：根据历史梯度大小调整学习率。

**更新规则**：

$$\mathbf{s}_{t+1} = \beta \mathbf{s}_t + (1-\beta)(\nabla J(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1}} + \epsilon} \nabla J(\mathbf{w}_t)$$

**含义**：梯度大的参数，学习率自动减小；梯度小的参数，学习率自动增大。

### 6.3 Adam：结合两者

**Adam = Momentum + RMSprop**

**更新规则**：

$$\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1)\nabla J(\mathbf{w}_t) \quad \text{（一阶矩估计）}$$
$$\mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1-\beta_2)(\nabla J(\mathbf{w}_t))^2 \quad \text{（二阶矩估计）}$$
$$\hat{\mathbf{m}}_{t+1} = \frac{\mathbf{m}_{t+1}}{1 - \beta_1^{t+1}} \quad \text{（偏差修正）}$$
$$\hat{\mathbf{v}}_{t+1} = \frac{\mathbf{v}_{t+1}}{1 - \beta_2^{t+1}} \quad \text{（偏差修正）}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_{t+1}} + \epsilon} \hat{\mathbf{m}}_{t+1}$$

**为什么Adam是最常用的优化器？**

1. 自适应学习率：每个参数有不同的学习率
2. 动量：平滑梯度，加速收敛
3. 偏差修正：解决初始化偏差问题
4. 对超参数不敏感：默认参数（$\beta_1=0.9, \beta_2=0.999$）通常就很好

---

## 第七部分：核心直觉总结

### 7.1 损失函数的本质

**损失函数将"预测好坏"量化为可优化的数学目标。**

- MSE：假设误差服从高斯分布 → 最大似然估计
- 不同损失函数反映不同的假设和偏好

### 7.2 梯度下降的本质

**梯度下降是"最速下降"策略的实现。**

- 负梯度方向是函数值下降最快的方向
- 学习率控制"步子大小"
- 凸函数保证收敛到全局最优

### 7.3 凸优化的重要性

**凸性是机器学习理论的基础。**

- 局部最优 = 全局最优
- 梯度下降保证找到最优解
- 线性回归是凸优化问题

### 7.4 非凸优化的挑战

**深度学习的损失函数通常非凸。**

- 多个局部最优和鞍点
- 没有全局最优保证
- 但实践中往往能找到"足够好"的解

---

## 思考题

### 问题1：为什么深度学习的损失函数通常非凸？

<details>
<summary>点击查看提示</summary>

考虑神经网络的输出形式：层层非线性激活的组合。损失函数是参数的复合函数，其中包含非线性变换。
</details>

<details>
<summary>点击查看答案</summary>

**答案：非线性激活函数导致非凸性。**

**详细分析**：

1. **线性组合保持凸性**：
   如果 $f$ 和 $g$ 都是凸函数，则 $\alpha f + \beta g$（$\alpha, \beta \geq 0$）也是凸函数。
   线性回归的输出是 $\mathbf{w}^T\mathbf{x}$，是参数的线性函数，MSE损失是凸函数。

2. **非线性激活破坏凸性**：
   神经网络的输出是：
   $$\hat{y} = \sigma(W_k \sigma(W_{k-1} ... \sigma(W_1 \mathbf{x})))$$
   
   即使 $\sigma$ 是凸函数（如ReLU），凸函数的复合不一定是凸函数。
   
   例如：$f(x) = x^2$ 是凸函数，但 $g(x) = f(-x) = (-x)^2 = x^2$ 也是凸函数。但如果是 $h(x) = f(\sigma(x))$，其中 $\sigma$ 是非线性函数，$h$ 通常不是凸函数。

3. **具体例子**：
   考虑最简单的单隐层网络：
   $$\hat{y} = w_2 \cdot \text{ReLU}(w_1 x)$$
   
   损失函数 $L = (y - \hat{y})^2$ 对参数 $(w_1, w_2)$ 是非凸的。可以验证，存在多个局部最优。

4. **深度网络的损失景观**：
   - 参数对称性：隐藏层神经元置换得到等价解
   - 权重符号翻转：某些激活函数下，$w$ 和 $-w$ 可以产生相同输出
   - 大量鞍点：高维空间中，鞍点比局部最优多得多

**为什么这很重要？**

- 非凸性意味着梯度下降不能保证找到全局最优
- 但实践中，SGD往往能找到"足够好"的解
- 理论上，深度学习的优化仍然是开放问题

</details>

### 问题2：梯度下降一定收敛到全局最优吗？

<details>
<summary>点击查看提示</summary>

区分凸函数和非凸函数的情况。凸函数有什么保证？非凸函数呢？
</details>

<details>
<summary>点击查看答案</summary>

**答案：不一定。取决于损失函数的性质和学习率的设置。**

**情况1：凸函数 + 合适的学习率 → 收敛到全局最优**

**定理**：如果 $J$ 是凸函数，梯度Lipschitz连续（$\|\nabla J(\mathbf{w}) - \nabla J(\mathbf{w}')\| \leq L\|\mathbf{w} - \mathbf{w}'\|$），学习率 $\eta \leq \frac{1}{L}$，则梯度下降满足：

$$J(\mathbf{w}_t) - J(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|^2}{2\eta t}$$

即以 $O(1/t)$ 的速度收敛到全局最优。

**情况2：凸函数 + 学习率太大 → 可能发散**

如果学习率太大，梯度下降可能震荡甚至发散。

**例子**：$J(w) = \frac{1}{2}w^2$

梯度：$\nabla J(w) = w$

更新：$w_{t+1} = w_t - \eta w_t = (1-\eta)w_t$

- 如果 $\eta < 2$：$w_t \to 0$（收敛）
- 如果 $\eta = 2$：$w_t = (-1)^t w_0$（震荡）
- 如果 $\eta > 2$：$|w_t| \to \infty$（发散）

**情况3：非凸函数 → 可能陷入局部最优或鞍点**

梯度下降只能保证收敛到临界点（$\nabla J(\mathbf{w}) = 0$），可能是：
- 全局最优
- 局部最优
- 鞍点

**鞍点问题**：

在高维空间中，鞍点（某些方向是极小值，某些方向是极大值）比局部最优多得多。

但是，随机梯度下降（SGD）的噪声有助于跳出鞍点。

**情况4：强凸函数 → 线性收敛**

如果 $J$ 是强凸函数（存在 $\mu > 0$，使得 $J(\mathbf{w}) - \frac{\mu}{2}\|\mathbf{w}\|^2$ 是凸函数），则收敛速度提升到线性：

$$J(\mathbf{w}_t) - J(\mathbf{w}^*) \leq O\left(\left(1 - \frac{\mu}{L}\right)^t\right)$$

**总结**：

| 条件 | 收敛性 |
|------|--------|
| 凸 + 学习率合适 | 全局最优，$O(1/t)$ |
| 凸 + 学习率太大 | 可能发散 |
| 强凸 + 学习率合适 | 全局最优，线性收敛 |
| 非凸 | 局部最优或鞍点 |

</details>

### 问题3：为什么随机梯度下降（SGD）能"跳出"局部最优？

<details>
<summary>点击查看答案</summary>

**答案：SGD的梯度噪声提供了"逃逸"能力。**

**机制分析**：

1. **梯度噪声**：
   批量梯度下降使用全部数据计算梯度：
   $$\nabla J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\nabla \mathcal{L}_i(\mathbf{w})$$
   
   SGD使用单个样本：
   $$\nabla \mathcal{L}_i(\mathbf{w}) = \nabla J(\mathbf{w}) + \underbrace{(\nabla \mathcal{L}_i(\mathbf{w}) - \nabla J(\mathbf{w}))}_{\text{噪声}}$$
   
   这个噪声使得SGD即使在局部最优（梯度为零），也可能因为噪声而"跳出"。

2. **逃离局部最优的直觉**：
   - 批量GD：精确遵循梯度方向，一旦到达局部最优（梯度为零），就停止
   - SGD：梯度有噪声，即使在局部最优，"噪声梯度"可能不为零，继续移动

3. **理论视角：随机过程**：
   SGD可以看作是在损失景观上进行随机游走。从统计物理的角度，这种随机性赋予了一定的"能量"来翻越势垒。

4. **模拟退火视角**：
   学习率可以看作"温度"。高学习率 → 高噪声 → 容易跳出局部最优；低学习率 → 低噪声 → 精细搜索。

**实践启示**：

- 训练初期：较大的学习率有助于探索，避免过早陷入局部最优
- 训练后期：较小的学习率有助于收敛，精细优化

这就是为什么"学习率衰减"（learning rate decay）是常用策略。

</details>

### 问题4：为什么线性回归有闭式解，但逻辑回归没有？

<details>
<summary>点击查看答案</summary>

**答案：MSE损失是参数的二次函数，可以解析求解；交叉熵损失涉及非线性（对数、Sigmoid），无法解析求解。**

**线性回归**：

损失函数：
$$J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

这是 $\mathbf{w}$ 的二次函数，可以写成：
$$J(\mathbf{w}) = \frac{1}{n}\|\mathbf{y} - X\mathbf{w}\|^2 = \frac{1}{n}\mathbf{w}^TX^TX\mathbf{w} - \frac{2}{n}\mathbf{y}^TX\mathbf{w} + \frac{1}{n}\mathbf{y}^T\mathbf{y}$$

对 $\mathbf{w}$ 求导并令其为零：
$$\nabla J(\mathbf{w}) = \frac{2}{n}X^TX\mathbf{w} - \frac{2}{n}X^T\mathbf{y} = 0$$

解得：
$$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$$

**逻辑回归**：

预测：
$$\hat{y}_i = \sigma(\mathbf{w}^T\mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}_i}}$$

损失函数（交叉熵）：
$$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))\right]$$

梯度：
$$\nabla J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(\sigma(\mathbf{w}^T\mathbf{x}_i) - y_i)\mathbf{x}_i$$

令梯度为零：
$$\sum_{i=1}^{n}(\sigma(\mathbf{w}^T\mathbf{x}_i) - y_i)\mathbf{x}_i = 0$$

**关键问题**：$\sigma(\mathbf{w}^T\mathbf{x}_i)$ 是 $\mathbf{w}$ 的非线性函数，这个方程没有解析解。

**为什么是非线性？**

$\sigma(z) = \frac{1}{1+e^{-z}}$ 包含指数函数，导致方程无法解析求解。

**但好消息**：

虽然逻辑回归没有闭式解，但：
1. 损失函数是凸的，存在唯一全局最优
2. 梯度下降保证收敛到全局最优

**对比**：

| 模型 | 损失函数 | 参数关系 | 闭式解 |
|------|---------|---------|--------|
| 线性回归 | MSE | 线性 | 有 |
| 逻辑回归 | 交叉熵 | 非线性（Sigmoid） | 无 |
| 神经网络 | 各种 | 高度非线性 | 无 |

</details>

---

## 今日要点

1. **损失函数量化"错误"**：MSE等价于高斯噪声下的最大似然估计

2. **梯度下降是最速下降**：负梯度方向是函数值下降最快的方向

3. **凸性是理论保证**：凸函数保证局部最优 = 全局最优

4. **学习率至关重要**：太大震荡/发散，太小收敛慢

5. **线性回归有闭式解**：$\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}$（但大多数模型没有）

6. **非凸优化是深度学习的常态**：虽然没有全局最优保证，但实践中常能找到好的解

---

## 数学附录

### A. 向量和矩阵微积分

**向量对标量求导**：

$$\frac{\partial \mathbf{a}^T\mathbf{x}}{\partial x_i} = a_i \implies \nabla_\mathbf{x}(\mathbf{a}^T\mathbf{x}) = \mathbf{a}$$

**二次型求导**：

$$\frac{\partial \mathbf{x}^TA\mathbf{x}}{\partial \mathbf{x}} = (A + A^T)\mathbf{x}$$

如果 $A$ 是对称矩阵：

$$\frac{\partial \mathbf{x}^TA\mathbf{x}}{\partial \mathbf{x}} = 2A\mathbf{x}$$

**矩阵求导公式**：

$$\frac{\partial \|A\mathbf{x} - \mathbf{b}\|^2}{\partial \mathbf{x}} = 2A^T(A\mathbf{x} - \mathbf{b})$$

### B. 正定矩阵

**定义**：对称矩阵 $A$ 是正定的，如果对任意非零向量 $\mathbf{x}$：

$$\mathbf{x}^TA\mathbf{x} > 0$$

**半正定**：$\mathbf{x}^TA\mathbf{x} \geq 0$

**性质**：
- 正定矩阵可逆
- 半正定矩阵可能有零特征值

### C. Lipschitz连续

**定义**：函数 $f$ 是Lipschitz连续的，如果存在常数 $L > 0$：

$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

**意义**：函数值变化不会太快，保证了优化的稳定性。

---

## 明日预告

Day 3 我们将从回归转向分类，探讨：

- Sigmoid函数的数学本质：为什么它能将线性组合映射到概率？
- 交叉熵损失：为什么分类不用MSE？
- 逻辑回归：第一个分类模型

分类问题引入了新的复杂性：预测从连续值变成离散标签，损失函数从MSE变成交叉熵。理解这些转变，是理解深度学习输出层设计的关键。
