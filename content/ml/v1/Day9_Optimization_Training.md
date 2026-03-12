# Day 9: 优化与训练动力学

## 核心问题：非凸优化如何找到好解？

深度学习的优化问题是非凸的——损失函数有无数局部最优和鞍点。然而，简单的梯度下降及其变体在实践中却能找到好解。为什么？

今天我们深入探讨：
1. 深度网络的损失地形
2. 优化算法的演进
3. 学习率调度的策略
4. 深度网络泛化的奥秘

---

## 第一部分：深度网络的损失地形

### 1.1 非凸优化的挑战

**回顾凸优化**：

凸函数的局部最优 = 全局最优。梯度下降保证收敛到全局最优。

**深度学习的现实**：

损失函数是非凸的，存在：
- 多个局部最优
- 鞍点
- 平坦区域

**为什么非凸？**

神经网络是多层非线性函数的复合：

$$f(\mathbf{x}) = \sigma_L(\mathbf{W}_L \sigma_{L-1}(...\sigma_1(\mathbf{W}_1 \mathbf{x})))$$

损失函数是权重的非线性函数，通常是非凸的。

**对称性导致多个等价解**：

- 神经元置换：交换两个神经元及其权重，输出不变
- 权重符号翻转：某些激活函数下，翻转权重符号输出不变

这些对称性导致多个等价的局部最优。

### 1.2 局部最优 vs 鞍点

**关键洞察**：高维空间中，鞍点比局部最优多得多。

**定义**：

- **局部最优**：所有方向都是极值（极大或极小）
- **鞍点**：有些方向是极大，有些方向是极小

**数学分析**：

在 $d$ 维空间中，一个临界点的Hessian矩阵有 $k$ 个负特征值（下降方向）。

- 局部极小：$k = 0$（所有特征值 $\geq 0$）
- 局部极大：$k = d$（所有特征值 $\leq 0$）
- 鞍点：$0 < k < d$

**概率分析**：

假设Hessian矩阵的特征值独立地从某分布中随机抽取：

- 局部极小的概率：$P(k=0) \approx 2^{-d}$
- 局部极大的概率：$P(k=d) \approx 2^{-d}$
- 鞍点的概率：$P(0 < k < d) \approx 1 - 2^{-d+1}$

当 $d$ 很大（如 $10^6$ 参数），鞍点几乎必然存在，局部最优极其稀少。

**结论**：深度网络的优化主要是逃离鞍点，而非逃离局部最优。

### 1.3 鞍点的影响

**鞍点的梯度行为**：

在鞍点附近，梯度可能很小，导致训练停滞。

**为什么SGD能逃离鞍点？**

SGD的梯度有噪声（来自mini-batch）：

$$\mathbf{g} = \nabla L(\mathbf{w}) + \boldsymbol{\epsilon}$$

噪声 $\boldsymbol{\epsilon}$ 可能推动参数离开鞍点。

**理论保证**：

在一定条件下，SGD以概率1逃离鞍点。

### 1.4 平坦与尖锐极小值

**关键洞察**：不是所有局部最优都等价。

**平坦极小值**：损失函数在最小值附近变化缓慢

**尖锐极小值**：损失函数在最小值附近变化剧烈

**泛化差异**：

平坦极小值通常泛化更好。

**解释**：

训练数据和测试数据分布略有差异。平坦极小值对小的参数扰动不敏感，因此对分布偏移更鲁棒。

**数学刻画**：

平坦极小值：Hessian矩阵的特征值小
尖锐极小值：Hessian矩阵的特征值大

**SGD的偏好**：

SGD倾向于找到平坦极小值。原因：
- 随机性使得优化器游走在平坦区域
- 平坦区域更容易"碰到"

### 1.5 损失地形的可视化

**二维截面可视化**：

选择两个方向，绘制损失函数的等高线图。

**方法**：
1. 找到一个最小值点 $\mathbf{w}^*$
2. 选择两个随机方向 $\mathbf{d}_1, \mathbf{d}_2$
3. 绘制 $L(\mathbf{w}^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$

**观察**：
- 损失地形像"山谷"
- 沿某些方向平坦，沿其他方向陡峭
- 存在许多局部结构

---

## 第二部分：优化算法演进

### 2.1 随机梯度下降（SGD）

**Batch GD vs SGD**：

**Batch GD**：使用全部数据计算梯度

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

**SGD**：使用单个样本计算梯度

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L_i(\mathbf{w}_t)$$

**Mini-batch SGD**：使用小批量数据

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \nabla L_i(\mathbf{w}_t)$$

**SGD的噪声**：

梯度估计有噪声：

$$\mathbf{g} = \nabla L(\mathbf{w}) + \boldsymbol{\epsilon}$$

其中 $\boldsymbol{\epsilon}$ 是噪声，方差与batch size成反比。

**SGD的优势**：

1. **计算效率**：每次更新只需少量数据
2. **逃离鞍点**：噪声帮助逃离鞍点
3. **泛化能力**：噪声有正则化效果

**SGD的劣势**：

1. **收敛震荡**：梯度方向不稳定
2. **学习率敏感**：需要仔细调整
3. **不同方向需要不同学习率**：峡谷问题

### 2.2 动量法（Momentum）

**问题**：SGD在"峡谷"中震荡。

**峡谷问题**：

损失函数在某个方向陡峭（山谷壁），在另一个方向平坦（山谷底）。

SGD在陡峭方向来回震荡，沿着平坦方向缓慢前进。

**动量法**：

引入"速度"概念，累积历史梯度：

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

其中 $\gamma$ 是动量系数（通常取 0.9）。

**物理直觉**：

想象一个小球在山谷中滚动：
- 梯度是"推力"
- 动量是"惯性"
- 小球会积累动量，沿着山谷底前进

**动量的作用**：

1. **加速收敛**：沿着一致方向加速
2. **减少震荡**：相反方向的梯度相互抵消
3. **逃离局部最优**：动量可以帮助冲出浅的局部最优

**Nesterov动量**：

先"向前看"，再计算梯度：

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla L(\mathbf{w}_t - \gamma \mathbf{v}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

**优势**：提前"看到"未来的位置，做出更明智的梯度计算。

### 2.3 自适应学习率：AdaGrad

**问题**：不同参数需要不同的学习率。

**稀疏特征问题**：

某些特征很少出现（如NLP中的稀有词），其对应参数的梯度很少非零。如果学习率相同，这些参数更新太少。

**AdaGrad**：

对每个参数维护一个累积梯度平方和：

$$G_t = G_{t-1} + (\nabla L(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\mathbf{w}_t)$$

**效果**：
- 梯度大的参数：学习率降低
- 梯度小的参数：学习率提高

**问题**：

$G_t$ 累积增加，学习率单调递减，最终趋于0，训练停止。

### 2.4 RMSprop

**改进AdaGrad**：

使用指数移动平均代替累积：

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

**效果**：

学习率不再单调递减，而是根据最近的梯度自适应调整。

### 2.5 Adam

**Adam = Momentum + RMSprop**

**算法**：

1. 计算梯度：
   $$g_t = \nabla L(\mathbf{w}_t)$$

2. 更新一阶矩（动量）：
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

3. 更新二阶矩：
   $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

4. 偏差修正：
   $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
   $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

5. 更新参数：
   $$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**默认参数**：
- $\beta_1 = 0.9$（动量系数）
- $\beta_2 = 0.999$（二阶矩系数）
- $\eta = 0.001$（学习率）
- $\epsilon = 10^{-8}$（数值稳定性）

**为什么需要偏差修正？**

初始化时 $m_0 = 0$，导致早期估计偏向0。偏差修正抵消这种偏差。

**Adam的优势**：

1. **自适应学习率**：每个参数有不同的学习率
2. **动量**：加速收敛，减少震荡
3. **鲁棒性**：对超参数不敏感
4. **内存效率**：只需存储一阶和二阶矩

**Adam的问题**：

近期研究表明，Adam可能在某些情况下泛化不如SGD。

**AdamW**：

将权重衰减（L2正则化）从梯度更新中分离，改善泛化能力。

### 2.6 优化器选择指南

| 优化器 | 优势 | 劣势 | 适用场景 |
|--------|------|------|---------|
| SGD | 泛化好、可控 | 收敛慢、调参难 | 图像分类、需要最佳泛化 |
| SGD+Momentum | 加速收敛 | 需要调学习率 | 通用场景 |
| AdaGrad | 稀疏特征 | 学习率递减 | 稀疏数据、NLP |
| RMSprop | 自适应学习率 | 可能不稳定 | RNN、非平稳目标 |
| Adam | 快速收敛、鲁棒 | 泛化可能不如SGD | 通用场景、快速原型 |
| AdamW | 改善泛化 | 稍复杂 | 推荐使用 |

**实践建议**：
1. 快速原型：Adam
2. 最终训练：SGD + Momentum 或 AdamW
3. 需要最佳泛化：SGD + Momentum + 学习率调度

---

## 第三部分：学习率调度

### 3.1 学习率的重要性

**学习率的影响**：

- 太大：训练不稳定，可能发散
- 太小：收敛太慢，可能陷入局部最优
- 合适：快速稳定收敛

**学习率与损失地形**：

不同的训练阶段需要不同的学习率：
- 初期：大学习率，快速下降
- 中期：中等学习率，精细调整
- 后期：小学习率，微调收敛

### 3.2 学习率衰减策略

#### 阶梯衰减（Step Decay）

每隔固定epoch数，学习率乘以衰减因子：

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}$$

**例子**：
- 每30个epoch，学习率乘以0.1
- 初始学习率0.1 → 0.01 → 0.001

#### 指数衰减（Exponential Decay）

$$\eta_t = \eta_0 \cdot \gamma^t$$

连续衰减，更平滑。

#### 余弦退火（Cosine Annealing）

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

学习率按余弦曲线衰减。

**优点**：
- 初期快速下降
- 中期平滑过渡
- 后期缓慢下降

### 3.3 热身（Warmup）

**问题**：训练初期，参数随机，梯度不稳定，大学习率可能导致训练崩溃。

**解决**：先用小学习率"热身"，再逐渐增大到目标学习率。

**线性热身**：

$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}}$$

在 $T_{warmup}$ 步内，学习率从0线性增加到目标学习率。

**适用场景**：
- Transformer训练
- 大batch训练
- 迁移学习

### 3.4 学习率发现器（Learning Rate Finder）

**方法**（Cyclical Learning Rates）：

1. 从很小的学习率开始（如 $10^{-6}$）
2. 每个batch增大学习率
3. 记录损失
4. 找到损失开始快速下降的学习率

**直觉**：

损失曲线通常呈现：
- 初始平坦（学习率太小）
- 快速下降（学习率合适）
- 开始上升（学习率太大）

选择快速下降区间内的学习率。

### 3.5 Cyclical Learning Rates

**思想**：学习率不必单调递减，可以在范围内循环。

**三角策略**：

学习率在最小值和最大值之间线性变化：

$$\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{cycle\_progress}{cycle\_length}$$

**优点**：
- 跳出局部最优
- 更好地探索损失地形
- 不需要手动衰减

---

## 第四部分：深度网络为何能泛化

### 4.1 经典理论的困惑

**传统泛化理论**：

泛化误差上界与模型复杂度相关：

$$R(f) \leq \hat{R}(f) + O\left(\sqrt{\frac{d}{n}}\right)$$

其中 $d$ 是模型复杂度（如VC维度），$n$ 是样本量。

**困惑**：

深度网络有数百万参数，VC维度极高，按理应该严重过拟合。

但实践中，深度网络泛化很好。

**矛盾**：经典理论无法解释深度学习的泛化能力。

### 4.2 隐式正则化

**假设**：优化算法本身具有正则化效果。

**SGD的隐式正则化**：

SGD的梯度噪声相当于添加了正则化：
- 噪声使优化器探索更多区域
- 倾向于找到平坦极小值
- 平坦极小值泛化更好

**早停**：

训练在验证误差开始上升时停止，是一种正则化。

### 4.3 平坦极小值假说

**假说**：SGD倾向于找到平坦极小值，平坦极小值泛化更好。

**直观理解**：

训练集和测试集分布略有差异。平坦极小值对小的参数扰动不敏感，因此对分布偏移更鲁棒。

**数学刻画**：

平坦极小值：Hessian矩阵的特征值小
尖锐极小值：Hessian矩阵的特征值大

**SGD为什么偏好平坦极小值？**

- 平坦区域更容易"碰到"
- 噪声使优化器在平坦区域游走
- 尖锐极小值容易被噪声"推出"

### 4.4 双下降现象

**传统观点**：

随着模型容量增加：
- 先下降（欠拟合缓解）
- 后上升（过拟合）

**双下降现象**：

实际上，测试误差呈"双下降"：
1. 先下降（欠拟合缓解）
2. 后上升（过拟合）
3. 再下降（过参数化）

**解释**：

在过参数化区域，模型容量足够大，优化可以找到多个解。SGD倾向于找到泛化好的解。

### 4.5 神经正切核（NTK）视角

**NTK理论**：

在无限宽网络和特定初始化条件下，神经网络的训练等价于核方法。

**核方法**：

$$f(\mathbf{x}) = \sum_{i=1}^{n} \alpha_i K(\mathbf{x}, \mathbf{x}_i)$$

其中 $K$ 是核函数（神经正切核）。

**含义**：

无限宽网络位于"线性区域"，训练是凸优化问题。

**局限性**：

- 有限宽网络不完全符合NTK假设
- NTK不能解释所有深度学习的现象

### 4.6 归纳偏置

**归纳偏置**：模型对解的偏好。

**深度网络的归纳偏置**：

1. **层次化结构**：匹配数据的层次化特性
2. **局部连接**：卷积网络的局部感受野
3. **权重共享**：卷积核在空间上共享
4. **激活函数**：引入非线性

**这些归纳偏置限制了假设空间，有助于泛化。**

---

## 第五部分：代码示例

### 5.1 优化器比较

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成测试数据
np.random.seed(42)
n = 100
X = np.random.randn(n, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(n) * 0.5

# 定义损失函数和梯度
def loss(w, b, X, y):
    return np.mean((X.squeeze() * w + b - y) ** 2)

def gradient(w, b, X, y):
    y_pred = X.squeeze() * w + b
    dw = 2 * np.mean((y_pred - y) * X.squeeze())
    db = 2 * np.mean(y_pred - y)
    return dw, db

# SGD
def sgd(X, y, lr=0.1, epochs=100):
    w, b = 0, 0
    history = []
    for _ in range(epochs):
        for i in range(len(X)):
            dw, db = gradient(w, b, X[i:i+1], y[i:i+1])
            w -= lr * dw
            b -= lr * db
        history.append(loss(w, b, X, y))
    return w, b, history

# Momentum
def momentum(X, y, lr=0.1, gamma=0.9, epochs=100):
    w, b = 0, 0
    vw, vb = 0, 0
    history = []
    for _ in range(epochs):
        dw, db = gradient(w, b, X, y)
        vw = gamma * vw + lr * dw
        vb = gamma * vb + lr * db
        w -= vw
        b -= vb
        history.append(loss(w, b, X, y))
    return w, b, history

# Adam
def adam(X, y, lr=0.1, beta1=0.9, beta2=0.999, epochs=100):
    w, b = 0, 0
    mw, mb = 0, 0
    vw, vb = 0, 0
    history = []
    for t in range(1, epochs + 1):
        dw, db = gradient(w, b, X, y)
        mw = beta1 * mw + (1 - beta1) * dw
        mb = beta1 * mb + (1 - beta1) * db
        vw = beta2 * vw + (1 - beta2) * dw ** 2
        vb = beta2 * vb + (1 - beta2) * db ** 2
        mw_hat = mw / (1 - beta1 ** t)
        mb_hat = mb / (1 - beta1 ** t)
        vw_hat = vw / (1 - beta2 ** t)
        vb_hat = vb / (1 - beta2 ** t)
        w -= lr * mw_hat / (np.sqrt(vw_hat) + 1e-8)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
        history.append(loss(w, b, X, y))
    return w, b, history

# 运行并比较
w_sgd, b_sgd, h_sgd = sgd(X, y, lr=0.01, epochs=50)
w_mom, b_mom, h_mom = momentum(X, y, lr=0.01, epochs=50)
w_adam, b_adam, h_adam = adam(X, y, lr=0.01, epochs=50)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(h_sgd, label='SGD')
plt.plot(h_mom, label='Momentum')
plt.plot(h_adam, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('优化器比较')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 思考题

### 问题1：为什么高维空间中鞍点比局部最优多？

<details>
<summary>点击查看答案</summary>

**答案：在高维空间中，临界点是局部最优的概率随维度指数下降。**

**数学推导**：

在 $d$ 维参数空间中，临界点的Hessian矩阵有 $d$ 个特征值。

临界点类型由特征值的符号决定：
- 局部极小：所有特征值 $> 0$（$k=0$ 个负特征值）
- 局部极大：所有特征值 $< 0$（$k=d$ 个负特征值）
- 鞍点：部分特征值 $> 0$，部分 $< 0$（$0 < k < d$）

**假设特征值独立地从某分布中随机抽取，正负概率各为 0.5**。

**概率计算**：

局部极小概率：
$$P(k=0) = P(\text{所有特征值} > 0) = 0.5^d$$

局部极大概率：
$$P(k=d) = P(\text{所有特征值} < 0) = 0.5^d$$

鞍点概率：
$$P(0 < k < d) = 1 - P(k=0) - P(k=d) = 1 - 2 \cdot 0.5^d = 1 - 2^{1-d}$$

**数值例子**：

| 维度 $d$ | 局部极小概率 | 鞍点概率 |
|---------|-------------|---------|
| 1 | 50% | 0% |
| 2 | 25% | 50% |
| 10 | 0.1% | 99.8% |
| 100 | $10^{-30}$ | $\approx 100\%$ |
| $10^6$ | $10^{-301030}$ | $\approx 100\%$ |

**结论**：

在深度网络中，参数数量通常在百万级别，局部最优几乎不可能存在！几乎所有临界点都是鞍点。

**对优化的意义**：

逃离鞍点比逃离局部最优更容易：
- 鞍点有下降方向
- SGD的噪声可以帮助找到下降方向
- 理论上，SGD可以在多项式时间内逃离鞍点

</details>

### 问题2：Adam为什么需要偏差修正？

<details>
<summary>点击查看答案</summary>

**答案：初始化为零导致早期估计有偏，偏差修正抵消这种偏差。**

**问题分析**：

Adam维护一阶矩和二阶矩的指数移动平均：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

初始化 $m_0 = 0, v_0 = 0$。

**为什么有偏差？**

假设梯度稳定在 $g$，期望：

$$\mathbb{E}[m_t] = g \cdot (1 - \beta_1^t)$$

推导：
$$m_1 = (1-\beta_1)g$$
$$m_2 = \beta_1 m_1 + (1-\beta_1)g = \beta_1(1-\beta_1)g + (1-\beta_1)g = (1-\beta_1)(\beta_1 + 1)g$$
$$...$$
$$m_t = (1-\beta_1^t)g$$

当 $t$ 很小时，$m_t \ll g$，估计偏低。

**偏差修正**：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

当 $t \to \infty$，$1-\beta_1^t \to 1$，修正项消失。

**数值例子**：

设 $\beta_1 = 0.9$，真实梯度 $g = 1$：

| $t$ | $m_t$ | $1-\beta_1^t$ | $\hat{m}_t$ |
|-----|-------|--------------|------------|
| 1 | 0.1 | 0.1 | 1.0 |
| 2 | 0.19 | 0.19 | 1.0 |
| 5 | 0.4095 | 0.4095 | 1.0 |
| 10 | 0.6513 | 0.6513 | 1.0 |
| 100 | 0.9999 | 0.9999 | 1.0 |

**不修正的影响**：

早期学习率会偏大（因为分母 $v_t$ 也偏小），导致训练不稳定。

**修正后的效果**：

早期估计准确，训练稳定。

**Adam的实现**：

```python
# 偏差修正
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

</details>

### 问题3：为什么SGD泛化性能通常比Adam好？

<details>
<summary>点击查看答案</summary>

**答案：SGD的噪声有正则化效果，帮助找到更平坦的极小值。**

**观察现象**：

在图像分类等任务上，SGD + Momentum通常比Adam泛化更好。

**可能原因**：

**1. 噪声正则化**

SGD的梯度噪声来自mini-batch采样，相当于添加了随机扰动：

$$\mathbf{g} = \nabla L(\mathbf{w}) + \boldsymbol{\epsilon}$$

这种噪声使优化器在极小值附近游走，倾向于找到平坦极小值。

Adam通过自适应学习率和动量，减少了这种噪声，可能收敛到尖锐极小值。

**2. 平坦极小值**

平坦极小值对参数扰动不敏感，泛化更好。

SGD的噪声帮助逃离尖锐极小值，收敛到平坦极小值。

Adam的稳定更新可能收敛到尖锐极小值。

**3. 学习率衰减**

SGD通常配合学习率衰减策略，而Adam的自适应机制可能在训练后期学习率偏大。

**实验证据**：

论文《On the Importance of Single Directions for Generalization》发现：
- SGD找到的极小值更平坦
- Adam找到的极小值更尖锐

**解决方案**：

**AdamW**：将权重衰减从梯度更新中分离，改善泛化。

**SGDR**：使用余弦退火重启，在SGD和Adam之间折中。

**SWATS**：先用Adam快速收敛，后用SGD精细调优。

**实践建议**：

1. **快速原型**：Adam（收敛快）
2. **最终训练**：SGD + Momentum + 学习率衰减（泛化好）
3. **混合策略**：Adam前期，SGD后期

**代码示例**：

```python
# SGD + Momentum
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.1, 
                            momentum=0.9, 
                            weight_decay=1e-4)

# 学习率衰减
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1)

# AdamW
optimizer = torch.optim.AdamW(model.parameters(), 
                               lr=0.001, 
                               weight_decay=0.01)
```

</details>

### 问题4：如何选择学习率？

<details>
<summary>点击查看答案</summary>

**答案：使用学习率发现器，或遵循经验规则并配合学习率衰减。**

**方法1：学习率发现器**

**步骤**：
1. 从很小的学习率开始（如 $10^{-6}$）
2. 每个batch增大学习率（如乘以1.1）
3. 记录损失和学习率
4. 绘制损失-学习率曲线
5. 选择损失快速下降前的学习率

**代码示例**：

```python
def find_learning_rate(model, train_loader, optimizer, 
                       init_lr=1e-6, final_lr=10, num_iter=100):
    lrs = []
    losses = []
    lr = init_lr
    
    for i, (X, y) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        optimizer.param_groups[0]['lr'] = lr
        
        # 前向传播
        output = model(X)
        loss = criterion(output, y)
        
        # 记录
        lrs.append(lr)
        losses.append(loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 增大学习率
        lr *= (final_lr / init_lr) ** (1 / num_iter)
    
    return lrs, losses

# 使用
lrs, losses = find_learning_rate(model, train_loader, optimizer)
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.show()
```

**选择规则**：
- 选择损失开始快速下降的学习率
- 通常是损失下降最快处的 $1/10$ 到 $1$

**方法2：经验规则**

| 优化器 | 典型学习率 |
|--------|-----------|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam | 0.0001 - 0.001 |
| AdamW | 0.0001 - 0.001 |
| RMSprop | 0.001 - 0.01 |

**方法3：网格搜索**

尝试多个学习率，选择验证误差最小的：

```python
learning_rates = [0.1, 0.01, 0.001, 0.0001]
for lr in learning_rates:
    # 训练模型
    # 评估验证误差
    # 选择最佳
```

**方法4：学习率衰减策略**

选择初始学习率后，配合衰减策略：

**阶梯衰减**：
```python
scheduler = MultiStepLR(optimizer, 
                        milestones=[30, 60, 90], 
                        gamma=0.1)
```

**余弦衰减**：
```python
scheduler = CosineAnnealingLR(optimizer, 
                               T_max=100, 
                               eta_min=0)
```

**实践建议**：

1. **快速实验**：先用 Adam，lr=0.001
2. **精细调优**：使用学习率发现器
3. **最终训练**：SGD + Momentum + lr衰减
4. **监控**：绘制学习率-损失曲线

</details>

---

## 今日要点

1. **损失地形**：高维空间中鞍点比局部最优多，SGD噪声帮助逃离鞍点

2. **优化器演进**：SGD → Momentum → Adam，自适应学习率是关键

3. **学习率调度**：热身、衰减、周期性策略各有适用场景

4. **泛化之谜**：隐式正则化、平坦极小值、双下降现象

5. **优化器选择**：快速原型用Adam，最终训练用SGD+Momentum

---

## 明日预告

Day 10 我们将深入卷积神经网络（CNN）：

- 卷积操作的数学定义
- 为什么CNN适合图像
- 感受野与特征提取
- 经典架构演进

CNN是深度学习在计算机视觉领域的基石，理解它对理解整个深度学习至关重要。
