# Day 8: 神经网络数学基础

## 核心问题：为什么需要"深度"？网络如何"学习"？

Week 1 我们学习了经典机器学习方法。今天开始，我们进入深度学习的世界。

神经网络看似复杂，但其本质是：
1. **线性变换 + 非线性激活** 的重复堆叠
2. **反向传播** 高效计算梯度
3. **梯度下降** 优化参数

理解这三点，就理解了深度学习的核心。

---

## 第一部分：从线性到非线性

### 1.1 线性模型的局限

**回顾线性模型**：

$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

**优势**：
- 简单、可解释
- 有闭式解（线性回归）
- 凸优化，保证全局最优

**致命局限**：只能学习**线性关系**。

**例子：XOR问题**

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**几何视角**：

在二维平面上，XOR问题的正类和负类无法用一条直线分开。

**证明**：假设存在线性分类器 $y = w_1 x_1 + w_2 x_2 + b$。

- 对于 (0, 0)：$b < 0$（预测为负类）
- 对于 (0, 1)：$w_2 + b > 0$
- 对于 (1, 0)：$w_1 + b > 0$
- 对于 (1, 1)：$w_1 + w_2 + b < 0$

矛盾：如果 $w_1 + b > 0$ 且 $w_2 + b > 0$，则 $w_1 + w_2 + 2b > 0$。

由于 $b < 0$，$w_1 + w_2 + b > w_1 + w_2 + 2b > 0$，与 $w_1 + w_2 + b < 0$ 矛盾。

**结论**：线性模型无法解决XOR问题。

### 1.2 非线性的必要性

**核心问题**：如何让模型学习非线性关系？

**答案**：引入非线性变换。

**方法1：手动特征工程**

将 $\mathbf{x}$ 变换为 $\phi(\mathbf{x})$，使得在新的特征空间中线性可分。

**例子**：XOR问题

原始特征：$\mathbf{x} = (x_1, x_2)$

新特征：$\phi(\mathbf{x}) = (x_1, x_2, x_1 \cdot x_2)$

| $x_1$ | $x_2$ | $x_1 \cdot x_2$ | XOR |
|-------|-------|-----------------|-----|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

在新特征空间中，可以用超平面分离。

**问题**：需要手动设计特征，对于复杂问题不可行。

**方法2：神经网络自动学习特征**

神经网络自动学习非线性变换，无需手动设计。

### 1.3 感知机：最简单的神经网络

**感知机（Perceptron）**：

$$\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

其中 $\sigma$ 是激活函数。

**单层感知机仍然是线性分类器**：

虽然输出经过非线性激活，但决策边界仍然是线性的：

$$\mathbf{w}^T\mathbf{x} + b = 0$$

**多层感知机（MLP）才能学习非线性**：

$$\hat{y} = \sigma(\mathbf{w}_2^T \sigma(\mathbf{w}_1^T \mathbf{x} + \mathbf{b}_1) + b_2)$$

两层之间的非线性激活使得整个函数可以学习非线性关系。

### 1.4 激活函数：非线性的来源

**为什么需要激活函数？**

**关键洞察**：线性变换的复合仍然是线性变换。

$$\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} = \mathbf{W}\mathbf{x}$$

没有非线性激活，多层网络等价于单层网络。

**激活函数引入非线性**，使网络能够学习复杂的非线性关系。

**常见激活函数**：

#### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**优点**：
- 输出范围 (0, 1)，可解释为概率
- 平滑、可微

**缺点**：
- 饱和区梯度接近0（梯度消失）
- 输出非零中心（影响梯度更新）
- 计算exp较慢

#### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**优点**：
- 输出范围 (-1, 1)，零中心
- 平滑、可微

**缺点**：
- 饱和区梯度消失

#### ReLU（Rectified Linear Unit）

$$\text{ReLU}(x) = \max(0, x)$$

**优点**：
- 计算简单（比较和取最大值）
- 非饱和区梯度恒为1（缓解梯度消失）
- 稀疏激活（负值输出为0）

**缺点**：
- 负值区域梯度为0（神经元"死亡"）
- 输出非零中心

**死亡ReLU问题**：

如果权重更新使得某个神经元的输入始终为负，该神经元将永远输出0，梯度永远为0，无法更新。

**解决方案**：
- Leaky ReLU：$\max(\alpha x, x)$，$\alpha$ 通常取 0.01
- Parametric ReLU：$\alpha$ 可学习
- ELU：负区间使用指数函数

#### Softmax

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**用途**：多分类输出层，将输出转化为概率分布。

### 1.5 多层感知机（MLP）

**定义**：

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\hat{y} = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

**术语**：
- 输入层：$\mathbf{x}$
- 隐藏层：$\mathbf{h}^{(1)}, \mathbf{h}^{(2)}, ...$
- 输出层：$\hat{y}$
- 深度：层数 $L$

**为什么"深度"有效？**

**直觉1：层次化表示**

深度网络学习层次化特征：
- 浅层：边缘、纹理
- 中层：局部模式
- 深层：语义概念

**直觉2：参数效率**

深度网络可以用更少的参数表示复杂函数。

**例子**：表示 $2^n$ 个线性区域

- 单层网络：需要 $2^n$ 个神经元
- 深度网络：只需要 $O(n)$ 层，每层 $O(1)$ 个神经元

### 1.6 XOR问题的神经网络解

**问题**：用两层神经网络解决XOR。

**网络结构**：
- 输入层：2个神经元
- 隐藏层：2个神经元
- 输出层：1个神经元

**手动设计权重**：

隐藏层：
$$\mathbf{W}^{(1)} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}, \quad \mathbf{b}^{(1)} = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$$

输出层：
$$\mathbf{W}^{(2)} = \begin{pmatrix} 1 \\ -2 \end{pmatrix}, \quad b^{(2)} = 0$$

激活函数：ReLU（或阶跃函数）

**计算**：

| $x_1$ | $x_2$ | $h_1$ | $h_2$ | $\hat{y}$ |
|-------|-------|-------|-------|-----------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 2 | 1 | 0 |

**几何理解**：

隐藏层学习两个线性边界：
- $h_1$：$x_1 + x_2 > 0$
- $h_2$：$x_1 + x_2 - 1 > 0$

输出层组合这两个边界，形成非线性决策边界。

---

## 第二部分：万能近似定理

### 2.1 定理陈述

**万能近似定理（Universal Approximation Theorem）**：

对于任意连续函数 $f: [0, 1]^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在一个具有单隐藏层的神经网络 $g$，使得：

$$\sup_{x \in [0,1]^n} |f(x) - g(x)| < \epsilon$$

**直觉**：单隐藏层神经网络可以近似任意连续函数到任意精度。

### 2.2 定理的意义

**正面意义**：

神经网络是**通用函数逼近器**，理论上可以学习任何连续函数。

这解释了为什么神经网络如此强大——只要有足够的神经元，就能表示任何函数。

**局限性**：

**局限1：存在性 vs 构造性**

定理只保证存在这样的网络，但不告诉我们：
- 需要多少神经元
- 如何找到这样的网络
- 权重是多少

**局限2：有限数据**

定理假设无限数据，现实中数据有限，需要泛化能力。

**局限3：效率**

单隐藏层可能需要指数级神经元才能近似某些函数，而深度网络可能只需要多项式级。

### 2.3 为什么还需要深度？

**效率论证**：

**例子**：表示函数 $f(x) = x^{2^k}$

**单隐藏层**：需要 $O(2^k)$ 个神经元

**深度网络**：$k$ 层，每层1个神经元，计算 $x^2$ 的 $k$ 次复合

$$f(x) = (((x^2)^2)^2)...^2 = x^{2^k}$$

**结论**：深度网络可以用指数级更少的参数表示某些函数。

**层次化表示**：

深度网络学习层次化特征：
- 图像：像素 → 边缘 → 纹理 → 部件 → 物体
- 文本：字符 → 词 → 短语 → 句子 → 段落

浅层网络需要直接从原始输入学习高层特征，效率低且难优化。

### 2.4 万能近似的几何直觉

**隐藏层学习区域划分**

考虑ReLU激活函数：

$$\text{ReLU}(x) = \max(0, x)$$

每个ReLU神经元定义一个超平面，将空间分成两部分。

**单隐藏层**：

$n$ 个ReLU神经元可以将空间分成多个线性区域。

**深度网络**：

每层ReLU都对空间进行进一步划分，区域数量指数增长。

**定理**：

$L$ 层ReLU网络，每层宽度为 $n$，可以产生 $O(n^L)$ 个线性区域。

### 2.5 从逼近到泛化

**万能近似定理只解决表示能力**，不解决泛化问题。

**泛化依赖于**：
- 网络结构（归纳偏置）
- 训练数据
- 正则化
- 优化算法

**深度学习的成功**不是因为万能近似定理，而是：
- 层次化结构匹配数据的层次化特性
- 正则化技术防止过拟合
- 优化算法能够找到好的解

---

## 第三部分：反向传播——深度学习的引擎

### 3.1 反向传播的核心思想

**问题**：如何高效计算神经网络的梯度？

**朴素方法**：数值微分

$$\frac{\partial f}{\partial w} \approx \frac{f(w + \epsilon) - f(w)}{\epsilon}$$

**问题**：
- 慢：每个参数需要一次前向传播
- 不精确：截断误差

**反向传播**：
- 只需要一次前向传播 + 一次反向传播
- 精确计算梯度

### 3.2 计算图

**计算图**：将计算过程表示为有向无环图（DAG）。

**例子**：$f(x, y, z) = (x + y) \cdot z$

**计算图**：
```
    x   y   z
    |   |   |
    +---+   |
        |   |
        +   |
        |   |
        |   |
        *---+
        |
        f
```

**前向传播**：
1. $a = x + y$
2. $f = a \cdot z$

**反向传播**：计算 $\frac{\partial f}{\partial x}$, $\frac{\partial f}{\partial y}$, $\frac{\partial f}{\partial z}$

### 3.3 链式法则

**标量情况**：

如果 $y = f(u)$，$u = g(x)$，则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**向量情况**：

如果 $\mathbf{y} = f(\mathbf{u})$，$\mathbf{u} = g(\mathbf{x})$，则：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$$

其中 $\frac{\partial \mathbf{y}}{\partial \mathbf{u}}$ 是雅可比矩阵。

### 3.4 反向传播算法推导

**网络结构**：

$$\mathbf{a}^{(0)} = \mathbf{x}$$
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$

**损失函数**：

$$L = \mathcal{L}(\mathbf{a}^{(L)}, \mathbf{y})$$

**目标**：计算 $\frac{\partial L}{\partial \mathbf{W}^{(l)}}$ 和 $\frac{\partial L}{\partial \mathbf{b}^{(l)}}$

**步骤1：输出层误差**

定义误差项：

$$\boldsymbol{\delta}^{(L)} = \frac{\partial L}{\partial \mathbf{z}^{(L)}}$$

**步骤2：误差反向传播**

$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

其中 $\odot$ 是逐元素乘法。

**步骤3：计算梯度**

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$
$$\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

### 3.5 详细推导

**使用链式法则**：

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

**计算 $\frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$**：

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

$$\frac{\partial z_i^{(l)}}{\partial W_{ij}^{(l)}} = a_j^{(l-1)}$$

因此：

$$\frac{\partial L}{\partial W_{ij}^{(l)}} = \delta_i^{(l)} \cdot a_j^{(l-1)}$$

即：

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

**计算 $\boldsymbol{\delta}^{(l)}$**：

$$\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}} = \frac{\partial L}{\partial \mathbf{a}^{(l)}} \cdot \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}}$$

其中：

$$\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} = \text{diag}(\sigma'(\mathbf{z}^{(l)}))$$

进一步：

$$\frac{\partial L}{\partial \mathbf{a}^{(l)}} = \frac{\partial L}{\partial \mathbf{z}^{(l+1)}} \cdot \frac{\partial \mathbf{z}^{(l+1)}}{\partial \mathbf{a}^{(l)}} = (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}$$

因此：

$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

### 3.6 反向传播算法

**算法：反向传播**

**前向传播**：
1. $\mathbf{a}^{(0)} = \mathbf{x}$
2. For $l = 1, 2, ..., L$:
   - $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$
   - $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$

**反向传播**：
1. 计算输出层误差：$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} L \odot \sigma'(\mathbf{z}^{(L)})$
2. For $l = L-1, L-2, ..., 1$:
   - $\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$
3. 计算梯度：
   - $\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$
   - $\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$

### 3.7 计算复杂度分析

**前向传播**：

每层的计算：
- 矩阵乘法：$O(n_{l-1} \cdot n_l)$
- 激活函数：$O(n_l)$

总复杂度：$O\left(\sum_{l=1}^{L} n_{l-1} \cdot n_l\right)$

**反向传播**：

每层的计算：
- 误差传播：$O(n_l \cdot n_{l+1})$
- 梯度计算：$O(n_l \cdot n_{l-1})$

总复杂度：与前向传播同阶

**数值微分的复杂度**：

假设有 $P$ 个参数，数值微分需要 $P$ 次前向传播，复杂度为 $O(P \cdot \text{前向传播})$。

**反向传播的优势**：

反向传播只需要一次前向传播和一次反向传播，复杂度为 $O(\text{前向传播})$。

对于有百万参数的网络，反向传播比数值微分快百万倍！

---

## 第四部分：梯度问题

### 4.1 梯度消失

**问题**：深层网络中，梯度在反向传播过程中逐层减小，浅层参数几乎不更新。

**原因分析**：

考虑第 $l$ 层的梯度：

$$\boldsymbol{\delta}^{(l)} = \left((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

展开到输出层：

$$\boldsymbol{\delta}^{(l)} = \left(\prod_{k=l+1}^{L} (\mathbf{W}^{(k)})^T \text{diag}(\sigma'(\mathbf{z}^{(k)}))\right) \boldsymbol{\delta}^{(L)}$$

**Sigmoid的导数**：

$$\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$$

最大值只有 0.25！

**梯度消失的数学**：

假设每层导数 $\sigma'(\mathbf{z}^{(k)}) \approx 0.25$，有 $L$ 层：

$$\|\boldsymbol{\delta}^{(1)}\| \approx 0.25^L \cdot \|\boldsymbol{\delta}^{(L)}\|$$

当 $L = 10$ 时，梯度缩小到 $0.25^{10} \approx 10^{-6}$！

**影响**：
- 浅层参数几乎不更新
- 网络难以学习层次化特征
- 训练极慢甚至停滞

### 4.2 梯度爆炸

**问题**：深层网络中，梯度在反向传播过程中逐层增大，导致数值溢出。

**原因分析**：

如果权重矩阵的谱范数大于1：

$$\|\mathbf{W}\|_2 > 1$$

则反向传播时：

$$\|\boldsymbol{\delta}^{(l)}\| \geq \|\mathbf{W}\|_2^{L-l} \cdot \|\boldsymbol{\delta}^{(L)}\|$$

梯度指数增长。

**影响**：
- 参数更新过大
- 数值溢出（NaN）
- 训练不稳定

### 4.3 解决方案

#### 方案1：激活函数选择

**ReLU的优势**：

$$\text{ReLU}(x) = \max(0, x)$$

导数：
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

正区间导数恒为1，不会梯度消失或爆炸！

**注意**：ReLU在负区间导数为0，可能导致神经元死亡。

**改进版本**：
- Leaky ReLU：$\max(\alpha x, x)$，$\alpha \approx 0.01$
- ELU：负区间使用指数函数
- Swish：$x \cdot \sigma(x)$

#### 方案2：权重初始化

**Xavier初始化**：

适用于Sigmoid、Tanh等饱和激活函数。

$$W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)$$

或：

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

目标：使前向传播和反向传播的方差保持一致。

**He初始化**：

适用于ReLU激活函数。

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

考虑ReLU会使一半神经元输出为0，方差减半。

#### 方案3：批归一化（Batch Normalization）

**思想**：在每层对激活值进行归一化，保持均值和方差稳定。

**算法**：

对于mini-batch $\{z_1, z_2, ..., z_m\}$：

1. 计算均值和方差：
   $$\mu = \frac{1}{m}\sum_{i=1}^{m} z_i$$
   $$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(z_i - \mu)^2$$

2. 归一化：
   $$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

3. 缩放和平移：
   $$y_i = \gamma \hat{z}_i + \beta$$

$\gamma$ 和 $\beta$ 是可学习参数。

**作用**：
- 稳定每层的输入分布
- 缓解梯度消失/爆炸
- 允许使用更大的学习率
- 有正则化效果

#### 方案4：残差连接（Residual Connection）

**残差块**：

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

**梯度传播**：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{I}$$

即使 $\frac{\partial F(\mathbf{x})}{\partial \mathbf{x}}$ 很小，梯度仍然可以通过恒等映射 $\mathbf{I}$ 传播。

**作用**：
- 解决梯度消失问题
- 允许训练非常深的网络（100+层）
- 恒等映射易于学习（$F(\mathbf{x}) = 0$）

#### 方案5：梯度裁剪

**方法**：限制梯度的范数。

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{c}{\|\mathbf{g}\|} \quad \text{if } \|\mathbf{g}\| > c$$

**作用**：防止梯度爆炸，适用于RNN。

---

## 第五部分：完整代码示例

### 5.1 从零实现神经网络

```python
import numpy as np

class NeuralNetwork:
    """从零实现多层感知机"""
    
    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: 每层神经元数量 [input, hidden1, hidden2, ..., output]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He初始化（适用于ReLU）
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation(self, z):
        """激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
    
    def _activation_derivative(self, z):
        """激活函数导数"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activation(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # 最后一层不用激活函数（回归）或用softmax（分类）
            if i < self.num_layers - 2:
                a = self._activation(z)
            else:
                a = z  # 输出层
            
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]
        
        # 存储梯度
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)
        
        # 输出层误差（MSE损失）
        delta = self.activations[-1] - y
        
        # 反向传播
        for i in range(self.num_layers - 2, -1, -1):
            dW[i] = self.activations[i].T @ delta / m
            db[i] = np.mean(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._activation_derivative(self.z_values[i-1])
        
        return dW, db
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """训练"""
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # 反向传播
            dW, db = self.backward(X, y)
            
            # 更新参数
            for i in range(self.num_layers - 1):
                self.weights[i] -= learning_rate * dW[i]
                self.biases[i] -= learning_rate * db[i]
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# 测试
if __name__ == "__main__":
    # XOR问题
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 创建网络
    nn = NeuralNetwork([2, 4, 1], activation='relu')
    
    # 训练
    losses = nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=True)
    
    # 预测
    predictions = nn.forward(X)
    print("\n预测结果:")
    for i in range(len(X)):
        print(f"输入: {X[i]}, 预测: {predictions[i][0]:.4f}, 真实: {y[i][0]}")
```

### 5.2 使用PyTorch实现

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 准备数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 创建模型
model = SimpleNN(input_size=2, hidden_size=4, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(5000):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# 预测
with torch.no_grad():
    predictions = model(X)
    print("\n预测结果:")
    for i in range(len(X)):
        print(f"输入: {X[i].tolist()}, 预测: {predictions[i].item():.4f}, 真实: {y[i].item()}")
```

---

## 思考题

### 问题1：为什么深度网络比浅层网络更高效？

<details>
<summary>点击查看答案</summary>

**答案：深度网络可以用指数级更少的参数表示某些函数。**

**表示效率**：

考虑表示函数 $f(x) = x^{2^k}$：

**浅层网络（单隐藏层）**：
- 需要近似 $x^{2^k}$ 这个多项式
- 多项式次数为 $2^k$，需要 $O(2^k)$ 个神经元

**深度网络**：
- 第1层计算 $x^2$
- 第2层计算 $(x^2)^2 = x^4$
- ...
- 第k层计算 $x^{2^k}$
- 只需要 $k$ 层，每层1个神经元，总共 $O(k)$ 个神经元

**参数效率**：

浅层网络：$O(2^k)$ 参数
深度网络：$O(k)$ 参数

指数级差异！

**线性区域数量**：

**定理**：$L$ 层ReLU网络，每层宽度为 $n$，可以产生 $O(n^L)$ 个线性区域。

**浅层网络**：$L=1$，宽度为 $n$，产生 $O(n)$ 个区域
**深度网络**：$L$ 层，每层宽度为 $n$，产生 $O(n^L)$ 个区域

**例子**：

要产生 $10^6$ 个线性区域：
- 浅层网络：需要 $10^6$ 个神经元
- 深度网络：6层，每层10个神经元，总共60个神经元

**层次化特征**：

深度网络学习层次化表示：
- 浅层：边缘、纹理
- 中层：局部模式
- 深层：语义概念

浅层网络需要直接从原始输入学习高层特征，效率低且难优化。

**总结**：

深度网络更高效的原因：
1. **参数效率**：指数级更少的参数
2. **表示效率**：指数级更多的线性区域
3. **层次化学习**：逐层抽象特征

</details>

### 问题2：反向传播为什么比数值微分高效？

<details>
<summary>点击查看答案</summary>

**答案：反向传播只需一次前向+一次反向传播，复杂度与参数数量无关。**

**数值微分**：

对于每个参数 $w_i$，计算：

$$\frac{\partial L}{\partial w_i} \approx \frac{L(w_i + \epsilon) - L(w_i)}{\epsilon}$$

**复杂度**：
- 每个参数需要一次前向传播
- 总复杂度：$O(P \cdot F)$，其中 $P$ 是参数数量，$F$ 是前向传播复杂度

**例子**：
- 参数数量 $P = 10^6$
- 前向传播时间 $F = 1$ms
- 数值微分总时间：$10^6 \times 1$ms = $10^3$秒 ≈ 17分钟

**反向传播**：

**复杂度**：
- 一次前向传播：$O(F)$
- 一次反向传播：$O(F)$（与前向传播同阶）
- 总复杂度：$O(F)$，与参数数量无关！

**为什么这么高效？**

反向传播利用了**计算图的结构**和**链式法则**：

1. **前向传播**：记录所有中间值
2. **反向传播**：从输出层逐层计算梯度，复用中间值

关键在于：**梯度可以复用**

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$$

$\boldsymbol{\delta}^{(l)}$ 从输出层逐层传播，只需计算一次。

**数学本质**：

数值微分：对每个参数独立计算
反向传播：自动微分，利用计算图的依赖关系

**自动微分 vs 数值微分**：

| 特性 | 数值微分 | 自动微分（反向传播） |
|------|---------|---------------------|
| 复杂度 | $O(P \cdot F)$ | $O(F)$ |
| 精度 | 有截断误差 | 精确 |
| 实现 | 简单 | 复杂 |
| 内存 | 低 | 需存储中间值 |

**例子对比**：

参数数量 $P = 10^6$，前向传播 $F = 1$ms：

| 方法 | 时间 |
|------|------|
| 数值微分 | 17分钟 |
| 反向传播 | 2ms |

**百万倍差距！**

</details>

### 问题3：为什么ReLU能缓解梯度消失问题？

<details>
<summary>点击查看答案</summary>

**答案：ReLU在正区间导数恒为1，不会像Sigmoid那样梯度逐层衰减。**

**Sigmoid的梯度消失**：

Sigmoid导数：
$$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

**最大值**：$\sigma'(0) = 0.25$

**饱和区**：
- $x \to \infty$：$\sigma'(x) \to 0$
- $x \to -\infty$：$\sigma'(x) \to 0$

**梯度传播**：

假设10层网络，每层导数为0.25：

$$\|\boldsymbol{\delta}^{(1)}\| \approx 0.25^{10} \cdot \|\boldsymbol{\delta}^{(10)}\| \approx 10^{-6} \cdot \|\boldsymbol{\delta}^{(10)}\|$$

梯度衰减到百万分之一！

**ReLU的优势**：

ReLU：
$$\text{ReLU}(x) = \max(0, x)$$

导数：
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

**正区间导数恒为1！**

**梯度传播**：

如果大部分神经元的输入为正（大约50%），则：

$$\|\boldsymbol{\delta}^{(1)}\| \approx \|\boldsymbol{\delta}^{(L)}\|$$

梯度不会衰减！

**ReLU的问题：死亡神经元**

当神经元输入始终为负时：
- 输出恒为0
- 梯度恒为0
- 参数永远不更新
- 神经元"死亡"

**解决方案**：

**Leaky ReLU**：
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

导数：
$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

$\alpha$ 通常取 0.01。

负区间梯度为 $\alpha \neq 0$，神经元不会死亡。

**对比**：

| 激活函数 | 正区间导数 | 负区间导数 | 梯度消失 | 死亡神经元 |
|---------|-----------|-----------|---------|-----------|
| Sigmoid | $\leq 0.25$ | $\leq 0.25$ | 严重 | 无 |
| Tanh | $\leq 1$ | $\leq 1$ | 中等 | 无 |
| ReLU | 1 | 0 | 无 | 有 |
| Leaky ReLU | 1 | $\alpha$ | 无 | 无 |

**总结**：

ReLU缓解梯度消失的原因：
1. 正区间导数恒为1，不衰减
2. 计算简单，效率高
3. 稀疏激活，有正则化效果

代价是死亡神经元问题，可通过Leaky ReLU等变体解决。

</details>

### 问题4：批归一化为什么能加速训练？

<details>
<summary>点击查看答案</summary>

**答案：批归一化稳定每层输入分布，允许使用更大学习率，并有正则化效果。**

**问题：内部协变量偏移**

深度网络中，每层输入的分布在训练过程中不断变化。

**原因**：
- 前一层的参数更新
- 导致当前层输入分布变化
- 当前层需要重新适应

**后果**：
- 学习率必须很小
- 训练缓慢
- 初始化敏感

**批归一化的解决方案**：

对每层输入进行归一化，使其分布稳定：

$$\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

然后进行缩放和平移：

$$y = \gamma \hat{z} + \beta$$

**作用机制**：

**1. 稳定输入分布**

每层输入的均值和方差被归一化到固定值，减少了内部协变量偏移。

**2. 允许更大学习率**

输入分布稳定后，可以使用更大的学习率而不担心梯度爆炸。

**例子**：

没有BN：学习率可能需要 0.001
有BN：学习率可以用 0.01 甚至更大

**3. 减少对初始化的依赖**

输入分布被归一化，初始化的影响减小。

**4. 正则化效果**

BN使用mini-batch的统计量，引入了噪声：
- 均值和方差是估计值
- 不同batch的估计不同
- 这种噪声有正则化效果

**数学分析**：

**前向传播**：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} z_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(z_i - \mu_B)^2$$
$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{z}_i + \beta$$

**反向传播**：

BN是可微的，梯度可以正常传播。

**训练 vs 推理**：

**训练时**：使用当前batch的均值和方差

**推理时**：使用训练时累积的running mean和running variance

```python
# 训练时
running_mean = 0.9 * running_mean + 0.1 * batch_mean
running_var = 0.9 * running_var + 0.1 * batch_var

# 推理时
z_normalized = (z - running_mean) / sqrt(running_var + epsilon)
```

**BN的位置**：

通常放在激活函数之前：

$$z = \text{BN}(\mathbf{W}\mathbf{x} + \mathbf{b})$$
$$a = \text{ReLU}(z)$$

**总结**：

批归一化加速训练的原因：
1. 稳定每层输入分布，减少内部协变量偏移
2. 允许使用更大学习率
3. 减少对初始化的依赖
4. 有正则化效果

</details>

---

## 今日要点

1. **非线性的必要性**：线性变换的复合仍是线性，激活函数引入非线性

2. **万能近似定理**：单隐藏层神经网络可近似任意连续函数，但深度网络更高效

3. **反向传播**：利用链式法则高效计算梯度，复杂度与前向传播同阶

4. **梯度问题**：梯度消失/爆炸是深层网络的主要挑战

5. **解决方案**：ReLU激活、He初始化、批归一化、残差连接

6. **计算效率**：反向传播比数值微分快百万倍

---

## 明日预告

Day 9 我们将深入优化与训练动力学：

- 深度网络的损失地形
- SGD、Momentum、Adam等优化器
- 学习率调度策略
- 深度网络为何能泛化

理解优化是训练深度网络的关键，明天我们将揭示训练动力学背后的原理。
