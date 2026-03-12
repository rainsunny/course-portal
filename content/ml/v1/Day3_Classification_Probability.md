# Day 3: 分类与概率建模

## 核心问题：如何从连续预测转向离散决策？

前两天我们学习了回归问题——预测连续值。今天我们转向分类问题——预测离散标签。

这看似只是一个输出的变化，但实际上引发了深刻的理论转变：

1. **输出空间变化**：从 $\mathbb{R}$ 到 $\{0, 1\}$（二分类）或 $\{1, 2, ..., K\}$（多分类）
2. **损失函数变化**：MSE不再适用，需要新的损失函数
3. **概率解释**：分类自然引入概率——不仅是"属于哪一类"，还有"多大把握"

---

## 第一部分：从回归到分类的转变

### 1.1 回顾：回归问题

**问题设定**：
- 输入：$\mathbf{x} \in \mathbb{R}^d$
- 输出：$y \in \mathbb{R}$（连续值）
- 目标：找到 $f: \mathbb{R}^d \to \mathbb{R}$ 使得 $f(\mathbf{x}) \approx y$

**线性回归模型**：
$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

输出是任意实数，与 $y$ 同一空间。

### 1.2 分类问题的特殊性

**二分类问题设定**：
- 输入：$\mathbf{x} \in \mathbb{R}^d$
- 输出：$y \in \{0, 1\}$（离散标签）
- 目标：找到 $f: \mathbb{R}^d \to \{0, 1\}$

**直接应用线性回归的问题**：

如果我们直接用线性回归：

$$\hat{y} = \mathbf{w}^T\mathbf{x} + b$$

问题在于：

1. **输出范围不匹配**：$\hat{y}$ 可以是任意实数，但标签是 $\{0, 1\}$
2. **阈值选择困难**：如何将连续输出转为离散标签？$\hat{y} > 0.5$ 就是正类？
3. **损失函数不合理**：$y=1, \hat{y}=10$ 和 $y=1, \hat{y}=0.6$，哪个"更好"？MSE会认为前者更差，但预测都是正确的。

**我们需要一种方法**：
1. 将线性组合映射到 $[0, 1]$ 区间
2. 输出可以解释为概率
3. 损失函数能正确衡量分类错误

---

## 第二部分：Sigmoid函数——从线性到概率

### 2.1 Sigmoid函数的定义

**Sigmoid函数**（也称Logistic函数）：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**基本性质**：

1. **值域**：$\sigma(z) \in (0, 1)$，完美匹配概率范围
2. **单调性**：$\sigma'(z) > 0$，输入越大输出越大
3. **对称性**：$\sigma(-z) = 1 - \sigma(z)$
4. **中点**：$\sigma(0) = 0.5$
5. **渐近线**：$\lim_{z \to \infty} \sigma(z) = 1$，$\lim_{z \to -\infty} \sigma(z) = 0$

**导数**：

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

这个优美的导数形式使得计算非常方便。

### 2.2 为什么是Sigmoid？（第一性原理推导）

这是今天的核心问题：**为什么选择Sigmoid，而不是其他将实数映射到 $(0,1)$ 的函数？**

#### 视角一：对数几率（Log Odds）

**几率的定义**：

在概率论中，事件发生的**几率（Odds）**定义为：

$$\text{Odds} = \frac{P}{1-P}$$

其中 $P$ 是事件发生的概率。

**例子**：
- $P = 0.5$：Odds = 1（五五开）
- $P = 0.8$：Odds = 4（4:1的赔率）
- $P = 0.9$：Odds = 9（9:1的赔率）

**对数几率（Log Odds，或Logit）**：

$$\text{Logit}(P) = \log\frac{P}{1-P}$$

**关键洞察**：如果我们假设对数几率是输入的线性函数：

$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b$$

那么我们可以反解出概率：

$$\frac{P(y=1|\mathbf{x})}{1 - P(y=1|\mathbf{x})} = e^{\mathbf{w}^T\mathbf{x} + b}$$

$$P(y=1|\mathbf{x}) = \frac{e^{\mathbf{w}^T\mathbf{x} + b}}{1 + e^{\mathbf{w}^T\mathbf{x} + b}} = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

这正是 **Sigmoid函数**！

**结论**：Sigmoid函数来自于**对数几率的线性假设**。

**为什么这个假设合理？**

1. **线性模型的自然推广**：我们希望模型简单、可解释，线性关系是最简单的选择
2. **几率比概率更适合线性建模**：概率范围受限 $[0,1]$，几率范围无界 $[0, \infty)$，对数几率范围 $(-\infty, +\infty)$，可以自由地进行线性组合

#### 视角二：指数族分布

**指数族**是概率论中一类重要的分布族，包括高斯、伯努利、泊松等。

**定义**：分布属于指数族，如果其概率密度/质量函数可以写成：

$$p(y; \eta) = h(y) \exp(\eta T(y) - A(\eta))$$

其中：
- $\eta$：自然参数（natural parameter）
- $T(y)$：充分统计量（sufficient statistic）
- $A(\eta)$：配分函数（log partition function）
- $h(y)$：基础测度

**伯努利分布**：

$$p(y; \mu) = \mu^y (1-\mu)^{1-y}, \quad y \in \{0, 1\}$$

改写为指数族形式：

$$p(y; \mu) = \exp\left[y \log\mu + (1-y)\log(1-\mu)\right]$$
$$= \exp\left[y \log\frac{\mu}{1-\mu} + \log(1-\mu)\right]$$

对照指数族形式：
- 自然参数：$\eta = \log\frac{\mu}{1-\mu}$
- 充分统计量：$T(y) = y$
- 配分函数：$A(\eta) = \log(1 + e^\eta)$

**关键**：反解 $\mu$ 与 $\eta$ 的关系：

$$\eta = \log\frac{\mu}{1-\mu} \implies \mu = \frac{e^\eta}{1 + e^\eta} = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)$$

**广义线性模型（GLM）**：

在GLM框架下，我们假设：
1. $y$ 服从指数族分布
2. 自然参数 $\eta = \mathbf{w}^T\mathbf{x}$（线性预测子）
3. 预测 $\mathbb{E}[y|\mathbf{x}]$ 通过链接函数与 $\eta$ 关联

对于伯努利分布：
$$\mathbb{E}[y|\mathbf{x}] = \mu = \sigma(\eta) = \sigma(\mathbf{w}^T\mathbf{x})$$

**结论**：Sigmoid函数是**伯努利分布在GLM框架下的自然选择**。

#### 视角三：最大熵原理

**最大熵原理**：在满足约束的条件下，选择熵最大的分布。

**问题**：给定特征的期望值 $\mathbb{E}[x]$，选择 $p(y|x)$ 的分布。

**结果**：最大熵分布恰好是指数族分布。对于二分类问题，最大熵模型给出的条件概率正是：

$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

**结论**：Sigmoid函数来自**最大熵原理**，是最"无偏"的选择。

### 2.3 Sigmoid的几何意义

**决策边界**：

当 $P(y=1|\mathbf{x}) = 0.5$ 时，模型无法决定是正类还是负类。此时：

$$\sigma(\mathbf{w}^T\mathbf{x} + b) = 0.5$$

$$\mathbf{w}^T\mathbf{x} + b = 0$$

这是一个**超平面**！

**几何解释**：
- $\mathbf{w}^T\mathbf{x} + b > 0$：预测为正类（$\sigma > 0.5$）
- $\mathbf{w}^T\mathbf{x} + b < 0$：预测为负类（$\sigma < 0.5$）
- $\mathbf{w}^T\mathbf{x} + b = 0$：决策边界

**Sigmoid的作用**：将到决策边界的"距离"映射为概率。

- 离决策边界越远，概率越接近0或1（高置信度）
- 在决策边界上，概率为0.5（不确定）

---

## 第三部分：逻辑回归——第一个分类模型

### 3.1 模型定义

**逻辑回归（Logistic Regression）**：

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = \sigma(-(\mathbf{w}^T\mathbf{x} + b))$$

**预测**：

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**注意**：虽然名字叫"回归"，但逻辑回归是**分类**模型。名字来源于它是对数几率的回归。

### 3.2 为什么不用MSE损失？

**尝试用MSE**：

$$J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \sigma(\mathbf{w}^T\mathbf{x}_i))^2$$

**问题1：非凸性**

考虑单个样本的损失：

$$\ell(\mathbf{w}) = (y - \sigma(\mathbf{w}^T\mathbf{x}))^2$$

梯度：

$$\frac{\partial \ell}{\partial \mathbf{w}} = -2(y - \sigma(\mathbf{w}^T\mathbf{x})) \cdot \sigma(\mathbf{w}^T\mathbf{x})(1-\sigma(\mathbf{w}^T\mathbf{x})) \cdot \mathbf{x}$$

当预测正确时（$y=1, \sigma \approx 1$ 或 $y=0, \sigma \approx 0$），梯度接近0，这是好的。

但问题在于，MSE损失对于Sigmoid函数是非凸的。

**为什么非凸？**

因为 $\sigma(z)$ 是S型函数，复合二次函数后产生非凸形状。可以证明，$J(\mathbf{w})$ 可能存在多个局部最优。

**问题2：梯度消失**

当 $\sigma(\mathbf{w}^T\mathbf{x})$ 接近0或1时，$\sigma(1-\sigma)$ 接近0，梯度消失，学习变慢甚至停止。

**问题3：概率解释不自然**

MSE假设误差服从高斯分布，对于二分类问题，这个假设不成立。

### 3.3 交叉熵损失——自然的损失函数

#### 推导一：最大似然估计

**假设**：样本独立同分布，标签服从伯努利分布。

$$P(y|\mathbf{x}, \mathbf{w}) = \sigma(\mathbf{w}^T\mathbf{x})^y \cdot (1 - \sigma(\mathbf{w}^T\mathbf{x}))^{1-y}$$

**似然函数**：

$$L(\mathbf{w}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \mathbf{w}) = \prod_{i=1}^{n} \sigma(\mathbf{w}^T\mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T\mathbf{x}_i))^{1-y_i}$$

**对数似然**：

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i)) \right]$$

**最大化对数似然 = 最小化负对数似然**：

$$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i)) \right]$$

这就是**交叉熵损失（Cross-Entropy Loss）**，也叫**对数损失（Log Loss）**。

**等价形式**：

令 $\hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$，则：

$$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1-y_i)\log(1 - \hat{p}_i) \right]$$

#### 推导二：信息论视角

**熵（Entropy）**：

一个分布 $p$ 的熵衡量其"不确定性"：

$$H(p) = -\sum_x p(x) \log p(x)$$

**交叉熵（Cross-Entropy）**：

用分布 $q$ 来编码分布 $p$ 所需的平均编码长度：

$$H(p, q) = -\sum_x p(x) \log q(x)$$

**KL散度（相对熵）**：

衡量两个分布的"距离"：

$$D_{KL}(p \| q) = H(p, q) - H(p) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**分类问题的视角**：

真实分布 $p$：$p(1) = y, p(0) = 1-y$（one-hot或label smoothing）
预测分布 $q$：$q(1) = \hat{p}, q(0) = 1 - \hat{p}$

交叉熵：

$$H(p, q) = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$$

**最小化交叉熵 = 最小化预测分布与真实分布的"距离"**

**为什么不用KL散度？**

$$D_{KL}(p \| q) = H(p, q) - H(p)$$

$H(p)$ 是常数（真实分布的熵），最小化KL散度等价于最小化交叉熵。

### 3.4 交叉熵损失的优良性质

#### 性质1：凸性

**定理**：对于逻辑回归，交叉熵损失是凸函数。

**证明**：

单个样本的损失：

$$\ell(\mathbf{w}) = -y \log \sigma(\mathbf{w}^T\mathbf{x}) - (1-y)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}))$$

设 $z = \mathbf{w}^T\mathbf{x}$，则：

$$\ell(z) = -y \log \sigma(z) - (1-y)\log(1 - \sigma(z))$$

对 $z$ 求导：

$$\frac{d\ell}{dz} = -y \cdot \frac{\sigma(z)(1-\sigma(z))}{\sigma(z)} + (1-y) \cdot \frac{\sigma(z)(1-\sigma(z))}{1-\sigma(z)}$$
$$= -y(1-\sigma(z)) + (1-y)\sigma(z)$$
$$= \sigma(z) - y$$

二阶导数：

$$\frac{d^2\ell}{dz^2} = \sigma(z)(1-\sigma(z)) > 0$$

因此 $\ell(z)$ 是凸函数。

对于向量 $\mathbf{w}$：

$$\nabla_{\mathbf{w}} \ell = (\sigma(\mathbf{w}^T\mathbf{x}) - y)\mathbf{x}$$

Hessian矩阵：

$$\nabla^2_{\mathbf{w}} \ell = \sigma(\mathbf{w}^T\mathbf{x})(1-\sigma(\mathbf{w}^T\mathbf{x}))\mathbf{x}\mathbf{x}^T$$

对于任意向量 $\mathbf{v}$：

$$\mathbf{v}^T(\nabla^2 \ell)\mathbf{v} = \sigma(\mathbf{w}^T\mathbf{x})(1-\sigma(\mathbf{w}^T\mathbf{x}))(\mathbf{v}^T\mathbf{x})^2 \geq 0$$

因此Hessian半正定，$\ell(\mathbf{w})$ 是凸函数。

凸函数的和仍然是凸函数，因此总损失 $J(\mathbf{w})$ 是凸函数。

**这意味着**：逻辑回归有全局最优解，梯度下降保证收敛！

#### 性质2：梯度简洁

$$\nabla_{\mathbf{w}} J = \frac{1}{n}\sum_{i=1}^{n}(\sigma(\mathbf{w}^T\mathbf{x}_i) - y_i)\mathbf{x}_i$$

**惊讶地发现**：这个梯度形式与线性回归的梯度非常相似！

线性回归梯度：$\nabla J = \frac{2}{n}X^T(X\mathbf{w} - \mathbf{y})$

逻辑回归梯度：$\nabla J = \frac{1}{n}X^T(\sigma(X\mathbf{w}) - \mathbf{y})$

**唯一的区别**：预测值经过了Sigmoid变换。

**梯度下降更新**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{1}{n}\sum_{i=1}^{n}(\sigma(\mathbf{w}_t^T\mathbf{x}_i) - y_i)\mathbf{x}_i$$

#### 性质3：梯度不会消失

当 $\hat{p} \to 0$ 或 $\hat{p} \to 1$ 时：

- 如果 $y=1$ 且 $\hat{p} \to 0$：$\sigma - y \to -1$（大梯度，快速修正）
- 如果 $y=0$ 且 $\hat{p} \to 1$：$\sigma - y \to 1$（大梯度，快速修正）

**对比MSE**：

MSE的梯度包含 $\sigma(1-\sigma)$ 项，当 $\hat{p} \to 0$ 或 $\hat{p} \to 1$ 时梯度消失。

**交叉熵避免了这个问题**！

### 3.5 逻辑回归的完整算法

**算法：逻辑回归**

**输入**：训练数据 $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$，学习率 $\eta$，迭代次数 $T$

**初始化**：$\mathbf{w}_0 = \mathbf{0}$

**迭代**：对于 $t = 0, 1, ..., T-1$：
1. 计算预测：$\hat{p}_i = \sigma(\mathbf{w}_t^T\mathbf{x}_i)$
2. 计算梯度：$\mathbf{g} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)\mathbf{x}_i$
3. 更新参数：$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{g}$

**输出**：$\mathbf{w}_T$

---

## 第四部分：Softmax——多分类推广

### 4.1 多分类问题

**设定**：
- 输入：$\mathbf{x} \in \mathbb{R}^d$
- 输出：$y \in \{1, 2, ..., K\}$（$K$ 个类别）
- 目标：预测 $P(y=k|\mathbf{x})$，$k = 1, 2, ..., K$

**约束**：$\sum_{k=1}^{K} P(y=k|\mathbf{x}) = 1$

### 4.2 Softmax函数

**定义**：

$$\text{Softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**作用**：将任意实数向量 $\mathbf{z} \in \mathbb{R}^K$ 映射到概率分布 $(p_1, p_2, ..., p_K)$，满足：
- $p_k > 0$ 对所有 $k$
- $\sum_k p_k = 1$

**为什么叫Softmax？**

**Hardmax（硬最大值）**：
$$\text{Hardmax}(\mathbf{z}) = \mathbf{e}_k, \text{ where } k = \arg\max_j z_j$$

输出是one-hot向量，只有最大值位置为1。

**Softmax（软最大值）**：
- 最大值位置获得最大概率
- 但不是"赢家通吃"，其他位置也有非零概率
- 是Hardmax的"软化"版本

**温度参数**：

$$\text{Softmax}(\mathbf{z}/T)_k = \frac{e^{z_k/T}}{\sum_{j=1}^{K} e^{z_j/T}}$$

- $T \to 0$：接近Hardmax（概率集中在最大值）
- $T \to \infty$：接近均匀分布（概率均等）

### 4.3 为什么是Softmax？（推导）

#### 推导一：对数几率的推广

二分类时：
$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x}$$

多分类时，考虑每个类别相对于基准类（设为第 $K$ 类）的对数几率：

$$\log\frac{P(y=k|\mathbf{x})}{P(y=K|\mathbf{x})} = \mathbf{w}_k^T\mathbf{x}, \quad k = 1, 2, ..., K-1$$

反解：

$$\frac{P(y=k|\mathbf{x})}{P(y=K|\mathbf{x})} = e^{\mathbf{w}_k^T\mathbf{x}}$$

$$P(y=k|\mathbf{x}) = P(y=K|\mathbf{x}) e^{\mathbf{w}_k^T\mathbf{x}}$$

由概率和为1：

$$\sum_{k=1}^{K} P(y=k|\mathbf{x}) = P(y=K|\mathbf{x}) \left(1 + \sum_{k=1}^{K-1} e^{\mathbf{w}_k^T\mathbf{x}}\right) = 1$$

$$P(y=K|\mathbf{x}) = \frac{1}{1 + \sum_{k=1}^{K-1} e^{\mathbf{w}_k^T\mathbf{x}}} = \frac{e^{\mathbf{w}_K^T\mathbf{x}}}{\sum_{k=1}^{K} e^{\mathbf{w}_k^T\mathbf{x}}}$$

（设 $\mathbf{w}_K = \mathbf{0}$ 作为基准）

一般形式：

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T\mathbf{x}}}$$

这正是 **Softmax**！

#### 推导二：指数族分布

多分类问题的标签服从**多项分布（Multinomial Distribution）**。

多项分布是指数族分布。类似于二分类情况，Softmax是多项分布在GLM框架下的自然链接函数。

### 4.4 多分类的交叉熵损失

**损失函数**：

$$J(\mathbf{W}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}$$

其中：
- $y_{ik}$：one-hot编码，$y_{ik} = 1$ 如果第 $i$ 个样本属于第 $k$ 类
- $\hat{p}_{ik} = \text{Softmax}(\mathbf{W}\mathbf{x}_i)_k$

**简化形式**：

如果第 $i$ 个样本属于第 $y_i$ 类：

$$J(\mathbf{W}) = -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{i, y_i}$$

**直觉**：最小化负对数似然 = 最大化正确类别的预测概率。

### 4.5 Softmax的梯度

**目标**：计算 $\frac{\partial J}{\partial \mathbf{w}_k}$

**结果**：

$$\frac{\partial J}{\partial \mathbf{w}_k} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_{ik} - y_{ik})\mathbf{x}_i$$

**惊人地简洁**！形式与二分类完全一致。

---

## 第五部分：概率输出的意义

### 5.1 为什么需要概率输出？

**决策只是概率的应用**：

模型输出概率 $P(y=1|\mathbf{x})$，决策规则是：

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

但阈值不一定是0.5！取决于：

1. **代价不对称**：
   - 医疗诊断：漏诊（假阴性）的代价远高于误诊（假阳性），阈值应该降低
   - 垃圾邮件过滤：误判重要邮件为垃圾邮件的代价高，阈值应该提高

2. **类别不平衡**：
   - 如果正类只占1%，0.5的阈值可能不合适
   - 可以根据类别比例调整阈值

**概率的其他用途**：

1. **置信度评估**：$P = 0.95$ 比 $P = 0.51$ 更可信
2. **决策组合**：多个模型的概率可以加权平均
3. **风险评估**：根据概率决定是否采取行动
4. **主动学习**：选择模型最不确定的样本标注

### 5.2 校准（Calibration）

**问题**：模型输出的概率准确吗？

**定义**：如果模型预测 $P(y=1|\mathbf{x}) = 0.7$，那么实际上应该有70%的样本是正类。

**校准曲线**：

将预测概率分成若干区间（如 $[0, 0.1], [0.1, 0.2], ...$），统计每个区间内实际正类的比例。

- **完美校准**：实际比例等于预测概率
- **过度自信**：预测概率极端（接近0或1），但实际比例没那么极端
- **不自信**：预测概率集中在中间，但实际比例很极端

**校准方法**：
- Platt Scaling：用逻辑回归校准
- Isotonic Regression：保序回归校准
- Temperature Scaling：学习一个温度参数

---

## 第六部分：核心直觉总结

### 6.1 Sigmoid的本质

**Sigmoid函数不是随意选择的**，而是从三个基本原理自然导出：

1. **对数几率线性假设**：假设对数几率是线性的
2. **指数族分布**：伯努利分布在GLM框架下的自然链接
3. **最大熵原理**：最无偏的概率模型

### 6.2 交叉熵的本质

**交叉熵损失是自然的损失函数**：

1. **最大似然估计**：假设伯努利分布，最大化数据概率
2. **信息论视角**：最小化编码长度，最小化预测与真实分布的"距离"
3. **凸性保证**：优化友好，有全局最优

### 6.3 分类 vs 回归的本质区别

| 方面 | 回归 | 分类 |
|------|------|------|
| 输出 | 连续值 | 离散标签 |
| 损失函数 | MSE | 交叉熵 |
| 输出解释 | 预测值 | 概率 |
| 优化保证 | 凸（线性模型） | 凸（逻辑回归） |
| 概率假设 | 高斯噪声 | 伯努利/多项分布 |

### 6.4 Softmax的统一视角

**Softmax是Sigmoid的推广**：

- 二分类：Sigmoid
- 多分类：Softmax

**Softmax的优雅性质**：
1. 输出是概率分布（和为1）
2. 梯度形式简洁（$\hat{p}_k - y_k$）
3. 温度参数控制"软硬"程度

---

## 思考题

### 问题1：交叉熵损失和最大似然估计有什么关系？

<details>
<summary>点击查看提示</summary>

回顾交叉熵损失的推导过程，思考它与似然函数的关系。
</details>

<details>
<summary>点击查看答案</summary>

**答案：交叉熵损失 = 负对数似然的平均值**

**推导**：

假设样本独立同分布，标签服从伯努利分布：

$$P(y|\mathbf{x}, \mathbf{w}) = \sigma(\mathbf{w}^T\mathbf{x})^y \cdot (1 - \sigma(\mathbf{w}^T\mathbf{x}))^{1-y}$$

**似然函数**（数据在参数下的概率）：

$$L(\mathbf{w}) = \prod_{i=1}^{n} P(y_i|\mathbf{x}_i, \mathbf{w})$$

**最大似然估计（MLE）**：找到使数据概率最大的参数。

$$\mathbf{w}_{MLE} = \arg\max_{\mathbf{w}} L(\mathbf{w})$$

取对数（单调变换，不改变最优解）：

$$\ell(\mathbf{w}) = \log L(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i)) \right]$$

最大化对数似然等价于最小化负对数似然：

$$\mathbf{w}_{MLE} = \arg\min_{\mathbf{w}} \left\{ -\sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i)) \right] \right\}$$

除以 $n$ 得到平均值，这就是**交叉熵损失**！

**深刻含义**：

1. **频率学派视角**：MLE是寻找最可能产生观测数据的参数
2. **信息论视角**：最小化编码长度
3. **优化视角**：最小化损失函数

这三个视角殊途同归，都指向同一个目标函数。

**推广**：

- 高斯分布 + MLE → MSE损失
- 伯努利分布 + MLE → 二分类交叉熵
- 多项分布 + MLE → 多分类交叉熵

</details>

### 问题2：为什么分类问题需要概率输出？

<details>
<summary>点击查看提示</summary>

思考概率输出比硬分类（直接输出0或1）多提供了什么信息。
</details>

<details>
<summary>点击查看答案</summary>

**答案：概率输出提供了置信度、代价敏感决策、以及更多下游应用的可能性。**

**1. 置信度评估**

硬分类只能告诉你"是哪一类"，概率输出还能告诉你"有多确定"。

- $P(y=1|\mathbf{x}) = 0.51$：几乎不确定
- $P(y=1|\mathbf{x}) = 0.99$：非常确定

**2. 代价敏感决策**

不同错误的代价不同。医疗诊断中，漏诊（假阴性）的代价远高于误诊（假阳性），因此应该降低决策阈值。

**3. 不确定性量化**

概率输出是不确定性量化的一种形式。模型知道自己"不知道"，可以识别分布外样本。

**4. 下游任务**

概率输出支持更多下游任务：期望值计算、风险估计、多模型集成、知识蒸馏等。

**总结**：概率输出是比硬分类更丰富的信息形式，支持更灵活和智能的决策。

</details>

### 问题3：Sigmoid函数的导数为什么是 $\sigma(z)(1-\sigma(z))$？

<details>
<summary>点击查看答案</summary>

**答案：这个优美性质来自Sigmoid函数的定义，对计算效率至关重要。**

**推导**：

设 $\sigma(z) = \frac{1}{1 + e^{-z}}$

使用商法则：

$$\sigma'(z) = \frac{e^z(1 + e^z) - e^z \cdot e^z}{(1 + e^z)^2} = \frac{e^z}{(1 + e^z)^2} = \sigma(z)(1-\sigma(z))$$

**用途**：

1. **计算效率**：前向传播已计算 $\sigma(z)$，导数直接可得
2. **揭示梯度消失问题**：当 $\sigma \to 0$ 或 $\sigma \to 1$ 时，$\sigma'(z) \to 0$
3. **与交叉熵配合**：交叉熵损失的梯度 $\frac{\partial \ell}{\partial z} = \sigma - y$，消除了 $\sigma'(z)$ 项，避免了梯度消失

</details>

### 问题4：Softmax为什么需要"减去最大值"的技巧？

<details>
<summary>点击查看答案</summary>

**答案：数值稳定性。防止指数运算溢出。**

**问题**：当 $z_k$ 很大时，$e^{z_k}$ 可能溢出。

**解决方案**：

$$\text{Softmax}(\mathbf{z})_k = \frac{e^{z_k - \max_j z_j}}{\sum_{j=1}^{K} e^{z_j - \max_j z_j}}$$

数学上等价（分子分母同时乘以 $e^{-\max z}$），但数值上稳定。

**性质**：
- 至少有一个 $z_j - \max_j z_j = 0$，对应 $e^0 = 1$
- 所有 $z_j - \max_j z_j \leq 0$，对应 $e^{z_j - \max_j z_j} \leq 1$
- 不会溢出！

这是深度学习中最常用的数值稳定技巧之一。

</details>

---

## 今日要点

1. **Sigmoid的本质**：对数几率的线性假设、伯努利分布的GLM、最大熵原理

2. **交叉熵的本质**：最大似然估计、信息论视角、凸性保证

3. **分类与回归的区别**：输出空间、损失函数、概率假设

4. **Softmax的推广**：多分类的自然选择，梯度形式简洁

5. **概率输出的意义**：置信度、代价敏感决策、不确定性量化

---

## 明日预告

Day 4 我们将深入模型评估：

- 训练/验证/测试的统计意义
- 精确率、召回率、F1的数学基础
- ROC/AUC的全局视角
- 如何判断模型"真的学到了"

模型评估不是简单的"准确率"，而是需要深入理解泛化能力、权衡取舍、统计显著性。
