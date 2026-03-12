# Day 6: 决策树与集成理论

## 核心问题：非参数化方法如何学习？

前五天我们学习了参数化模型（线性模型、逻辑回归），它们有一个共同特点：**模型形式预先固定**，学习的是参数值。

今天我们转向非参数化方法——**决策树**。它不预设模型形式，而是从数据中"学习"如何分裂。

更精彩的是，决策树是集成学习的基础。多个"弱学习器"如何组合成"强学习器"？这是今天要回答的核心问题。

---

## 第一部分：决策树的信息论基础

### 1.1 决策树的本质

**决策树是什么？**

一棵决策树就是一系列"如果...那么..."规则的组合：

```
                年龄 > 30?
               /          \
             是            否
            /                \
      收入 > 50k?          学生?
       /    \              /    \
      是    否            是     否
     /      \            /       \
   贷款?   不贷款      不贷款    贷款?
```

**核心思想**：
1. 选择一个特征
2. 根据特征值分裂数据
3. 对每个子集递归

**关键问题**：
- 选择哪个特征分裂？
- 如何确定分裂点？

### 1.2 熵：不确定性的度量

**熵的定义**：

对于离散随机变量 $X$，取值为 $\{x_1, x_2, ..., x_n\}$，概率分布为 $P$：

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

**直觉理解**：

熵衡量"不确定性"或"混乱程度"：
- 熵高：不确定性强，分布均匀
- 熵低：不确定性强，分布集中
- 熵为0：完全确定

**例子**：

**情况1**：公平硬币
$$H = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = 1 \text{ bit}$$

**情况2**：不公平硬币（P(正面)=0.9）
$$H = -0.9 \log_2 0.9 - 0.1 \log_2 0.1 \approx 0.47 \text{ bits}$$

**情况3**：完全确定（P(正面)=1）
$$H = -1 \log_2 1 = 0 \text{ bits}$$

**为什么用 $\log$？**

**信息论视角**：

信息量定义为 $I(x) = -\log P(x)$，表示"惊讶程度"：
- 发生概率低的事件 → 信息量大
- 发生概率高的事件 → 信息量小

熵是信息量的期望：
$$H(X) = \mathbb{E}[I(X)] = \sum P(x) \cdot (-\log P(x))$$

**为什么用 $\log_2$？**

单位是"比特"（bits），与计算机存储相关。也可以用自然对数 $\ln$，单位是"纳特"（nats）。

### 1.3 信息增益：选择最优特征

**问题**：给定一个特征，分裂后不确定性降低了多少？

**条件熵**：

给定特征 $A$，类别 $Y$ 的条件熵：

$$H(Y|A) = \sum_{a \in A} P(A=a) \cdot H(Y|A=a)$$

**信息增益**：

$$IG(Y, A) = H(Y) - H(Y|A)$$

**含义**：知道特征 $A$ 后，对类别 $Y$ 的不确定性减少了多少。

**决策树的分裂准则**：

选择使信息增益最大的特征：

$$A^* = \arg\max_{A} IG(Y, A)$$

**例子**：

假设有14个样本，类别分布：
- 正类：9个
- 负类：5个

**分裂前的熵**：
$$H(Y) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} \approx 0.94 \text{ bits}$$

**考虑特征A（如"天气"）**：

| 天气 | 正类 | 负类 | 总数 |
|------|------|------|------|
| 晴天 | 2 | 3 | 5 |
| 阴天 | 4 | 0 | 4 |
| 雨天 | 3 | 2 | 5 |

**条件熵**：
$$H(Y|A) = \frac{5}{14}H(Y|晴) + \frac{4}{14}H(Y|阴) + \frac{5}{14}H(Y|雨)$$

其中：
- $H(Y|晴) = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} \approx 0.97$
- $H(Y|阴) = -\frac{4}{4}\log_2\frac{4}{4} = 0$（完全纯净！）
- $H(Y|雨) = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} \approx 0.97$

$$H(Y|A) = \frac{5}{14} \times 0.97 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.97 \approx 0.69$$

**信息增益**：
$$IG(Y, A) = 0.94 - 0.69 = 0.25 \text{ bits}$$

**选择特征**：计算所有特征的信息增益，选择最大的。

### 1.4 信息增益比：解决偏向问题

**问题**：信息增益偏向取值多的特征。

**极端例子**：

考虑"样本ID"这个特征，每个样本有唯一ID。分裂后每个子集只有一个样本，熵为0。

$$IG(Y, ID) = H(Y) - 0 = H(Y)$$

信息增益最大，但这个分裂毫无意义！

**信息增益比**：

$$IGR(Y, A) = \frac{IG(Y, A)}{H(A)}$$

其中 $H(A)$ 是特征 $A$ 的熵：

$$H(A) = -\sum_{a \in A} P(A=a) \log_2 P(A=a)$$

**直觉**：$H(A)$ 衡量特征取值的"分散程度"。取值越多，$H(A)$ 越大，分母越大，惩罚取值多的特征。

**例子**：

对于"样本ID"特征：
- 每个样本一个ID，$H(ID)$ 很大
- 信息增益比很小，避免了偏向

### 1.5 基尼不纯度：另一种分裂准则

**定义**：

$$Gini(Y) = 1 - \sum_{i=1}^{n} P(y_i)^2$$

**直觉**：

从数据集中随机抽取两个样本，它们类别不同的概率。

**推导**：

两样本类别相同的概率：$\sum_i P(y_i)^2$

两样本类别不同的概率：$1 - \sum_i P(y_i)^2$

**例子**：

**情况1**：均匀分布（$P = 0.5, 0.5$）
$$Gini = 1 - 0.5^2 - 0.5^2 = 0.5$$

**情况2**：纯净（$P = 1, 0$）
$$Gini = 1 - 1^2 = 0$$

**情况3**：不平衡（$P = 0.9, 0.1$）
$$Gini = 1 - 0.9^2 - 0.1^2 = 0.18$$

**基尼增益**：

$$\Delta Gini = Gini(Y) - Gini(Y|A)$$

其中 $Gini(Y|A)$ 是条件基尼不纯度：

$$Gini(Y|A) = \sum_{a \in A} P(A=a) \cdot Gini(Y|A=a)$$

**CART算法**使用基尼不纯度。

### 1.6 熵 vs 基尼

| 特性 | 熵 | 基尼 |
|------|-----|------|
| 公式 | $-\sum p_i \log p_i$ | $1 - \sum p_i^2$ |
| 范围 | $[0, \log_2 n]$ | $[0, 1-1/n]$ |
| 计算复杂度 | 需要计算对数 | 只需平方 |
| 效果 | 类似 | 类似 |

**实践中**：两者效果接近，基尼计算更快，更常用。

### 1.7 决策树的构建算法

**ID3算法**（Iterative Dichotomiser 3）：

1. 计算所有特征的信息增益
2. 选择信息增益最大的特征作为分裂节点
3. 对每个子节点递归

**缺点**：
- 只能处理离散特征
- 偏向取值多的特征
- 容易过拟合

**C4.5算法**（ID3的改进）：

1. 使用信息增益比代替信息增益
2. 处理连续特征（二分法）
3. 处理缺失值
4. 剪枝

**CART算法**（Classification and Regression Trees）：

1. 使用基尼不纯度
2. 二叉树（每次分裂只分两支）
3. 可用于分类和回归

### 1.8 连续特征的处理

**问题**：连续特征如何选择分裂点？

**方法**：二分法

1. 将连续特征值排序
2. 考虑所有相邻值的中点作为候选分裂点
3. 选择使信息增益（或基尼增益）最大的分裂点

**例子**：

特征值：$[1, 3, 5, 7, 9]$

候选分裂点：$2, 4, 6, 8$

计算每个分裂点的信息增益，选择最优。

### 1.9 剪枝：控制复杂度

**问题**：决策树容易过拟合。

**原因**：可以无限分裂，直到每个叶节点只有一个样本。

**解决**：剪枝

#### 预剪枝（Pre-pruning）

在构建过程中停止分裂：

- 最大深度限制
- 叶节点最小样本数
- 分裂的最小信息增益

**优点**：计算快
**缺点**：可能过早停止，错过后续好的分裂

#### 后剪枝（Post-pruning）

先构建完整树，再剪掉不必要的分支：

**代价复杂度剪枝（Cost-Complexity Pruning）**：

定义树的代价：
$$R_\alpha(T) = R(T) + \alpha |T|$$

其中：
- $R(T)$：树的训练误差
- $|T|$：叶节点数（复杂度）
- $\alpha$：正则化参数

**算法**：
1. 构建完整树 $T_0$
2. 逐步增加 $\alpha$，每次剪掉使 $R_\alpha(T)$ 最小的分支
3. 得到一系列子树 $T_0, T_1, ..., T_n$
4. 用验证集选择最优子树

**优点**：更全面地考虑
**缺点**：计算量大

---

## 第二部分：集成学习的理论基础

### 2.1 "三个臭皮匠顶个诸葛亮"

**集成学习的核心思想**：组合多个模型，得到比任何单个模型都好的性能。

**为什么有效？**

**直觉1：降低方差**

单个模型可能因为数据的随机性而表现不稳定。多个模型的平均可以抵消这种随机性。

**直觉2：降低偏差**

单个模型可能有系统性偏差。多个模型从不同角度学习，可能抵消偏差。

**直觉3：扩大假设空间**

单个模型的假设空间有限。多个模型的组合覆盖更大的假设空间。

### 2.2 弱学习器与强学习器

**弱学习器（Weak Learner）**：

性能略好于随机猜测的学习器。

形式化：对于某个任务，准确率略高于 50%（二分类）。

**强学习器（Strong Learner）**：

性能很好的学习器。

形式化：准确率任意高（给定足够数据）。

**核心问题**：弱学习器能组合成强学习器吗？

**回答**：可以！这是集成学习的理论基础。

### 2.3 PAC学习框架

**PAC学习（Probably Approximately Correct）**：

一个概念类 $\mathcal{C}$ 是PAC可学习的，如果存在算法 $\mathcal{A}$，对于任意 $\epsilon > 0$（精度）和 $\delta > 0$（置信度），以至少 $1-\delta$ 的概率输出一个假设 $h$，使得：

$$\text{error}(h) \leq \epsilon$$

**弱PAC可学习**：

误差略低于 50%：
$$\text{error}(h) \leq \frac{1}{2} - \gamma$$

其中 $\gamma > 0$ 是某个小量。

**Schapire定理（1990）**：

**弱PAC可学习 $\Leftrightarrow$ 强PAC可学习**

**意义**：如果能找到弱学习器，就能组合成强学习器。

### 2.4 偏差-方差分解回顾

对于回归问题，单个模型的期望误差：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**集成模型**：

设集成由 $M$ 个模型 $f_1, f_2, ..., f_M$ 组成，采用平均：

$$\hat{f}_{ens}(x) = \frac{1}{M}\sum_{m=1}^{M} f_m(x)$$

假设每个模型的偏差和方差相同：
- 单个模型偏差：$b$
- 单个模型方差：$\sigma^2$
- 模型间相关性：$\rho$

**集成模型的偏差**：

$$\text{Bias}(\hat{f}_{ens}) = \mathbb{E}[\hat{f}_{ens}] - f = b$$

偏差不变（假设每个模型偏差相同）。

**集成模型的方差**：

$$\text{Var}(\hat{f}_{ens}) = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

**分析**：
- 第一项 $\rho \sigma^2$：模型间的共同方差（无法通过集成消除）
- 第二项 $\frac{1-\rho}{M}\sigma^2$：可以通过集成消除的部分

**关键洞察**：
- 模型越独立（$\rho$ 越小），集成效果越好
- 模型越多（$M$ 越大），方差越低

### 2.5 Bagging：降低方差

**Bagging（Bootstrap Aggregating）**：

**核心思想**：通过对数据重采样，训练多个独立的模型，然后平均/投票。

**算法**：

1. 对于 $m = 1, 2, ..., M$：
   - 从训练数据中有放回抽样，得到数据集 $D_m$
   - 在 $D_m$ 上训练模型 $f_m$

2. 集成预测：
   - 回归：$\hat{y} = \frac{1}{M}\sum_{m=1}^{M} f_m(x)$
   - 分类：$\hat{y} = \text{majority vote}(f_1(x), ..., f_M(x))$

**Bootstrap抽样**：

从 $n$ 个样本中有放回抽取 $n$ 个样本。

每个样本被抽中的概率：$1 - (1 - \frac{1}{n})^n \approx 1 - e^{-1} \approx 63.2\%$

未被抽中的概率：$e^{-1} \approx 36.8\%$

这些未被抽中的样本称为 **袋外（Out-of-Bag, OOB）样本**，可用于验证。

**为什么Bagging能降低方差？**

假设模型独立（$\rho = 0$）：

$$\text{Var}(\hat{f}_{ens}) = \frac{\sigma^2}{M}$$

方差降低为原来的 $\frac{1}{M}$！

**实际上**：模型不完全独立（$\rho > 0$），但方差仍显著降低。

**Bagging适合什么模型？**

**高方差、低偏差的模型**：
- 深决策树
- 不剪枝的决策树

**不适合**：
- 低方差的模型（如线性回归）
- 高偏差的模型（需要Boosting）

### 2.6 Boosting：降低偏差

**Boosting的核心思想**：

顺序训练模型，每个新模型专注于之前模型的错误。

**直觉**：

- 第一轮：训练一个模型，有一些错误
- 第二轮：训练一个模型，专注于第一轮的错误样本
- 第三轮：训练一个模型，专注于前两轮的错误样本
- ...

**最终预测**：加权组合所有模型。

**为什么能降低偏差？**

每个新模型都在纠正之前模型的偏差，逐步提升整体性能。

**Boosting的关键问题**：
1. 如何衡量"错误"？
2. 如何"专注"错误样本？
3. 如何组合模型？

### 2.7 AdaBoost：自适应Boosting

**算法**：

**初始化**：每个样本权重 $w_i = \frac{1}{n}$

**For** $m = 1, 2, ..., M$：

1. 用权重 $w$ 训练弱学习器 $f_m$

2. 计算加权误差：
   $$\epsilon_m = \frac{\sum_{i=1}^{n} w_i \cdot \mathbb{1}[f_m(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i}$$

3. 计算模型权重：
   $$\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$$

4. 更新样本权重：
   $$w_i \leftarrow w_i \cdot \exp(-\alpha_m y_i f_m(x_i))$$

5. 归一化权重

**最终预测**：
$$\hat{y} = \text{sign}\left(\sum_{m=1}^{M} \alpha_m f_m(x)\right)$$

**为什么 $\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$？**

$\alpha_m$ 是模型权重，误差越低，权重越大。

- $\epsilon_m = 0.5$：随机猜测，$\alpha_m = 0$（权重为0，不参与）
- $\epsilon_m < 0.5$：好于随机，$\alpha_m > 0$
- $\epsilon_m = 0$：完美，$\alpha_m = \infty$（主导决策）

**为什么权重更新是 $\exp(-\alpha_m y_i f_m(x_i))$？**

- 如果预测正确（$y_i f_m(x_i) > 0$）：权重降低
- 如果预测错误（$y_i f_m(x_i) < 0$）：权重增加

这确保下一轮模型专注于错误样本。

**AdaBoost的训练误差界**：

$$\text{Training Error} \leq \prod_{m=1}^{M} 2\sqrt{\epsilon_m(1-\epsilon_m)}$$

如果每个弱学习器略好于随机（$\epsilon_m \leq 0.5 - \gamma$），训练误差指数下降！
---

## 第三部分：随机森林——Bagging的杰作

### 3.1 随机森林的核心创新

**Bagging + 决策树**：

对数据进行Bootstrap抽样，训练多棵决策树，最后投票/平均。

**问题**：树之间相关性较高（$\rho$ 较大），方差降低有限。

**随机森林的创新**：

在Bagging基础上，**随机选择特征子集**。

**算法**：

对于每棵树：
1. Bootstrap抽样得到数据集
2. 在每个分裂节点：
   - 随机选择 $k$ 个特征
   - 在这 $k$ 个特征中选择最优分裂
3. 生长到最大深度（或不剪枝）

**特征子集大小 $k$ 的选择**：

- 分类：$k \approx \sqrt{d}$（$d$ 是总特征数）
- 回归：$k \approx d/3$

### 3.2 为什么随机选择特征能降低相关性？

**分析**：

假设有 $d$ 个特征，最优分裂特征是 $j^*$。

**普通Bagging**：

所有树都会优先选择特征 $j^*$，树之间高度相关。

**随机森林**：

特征 $j^*$ 被选中的概率是 $\frac{k}{d}$。

如果 $k = \sqrt{d}$，概率是 $\frac{1}{\sqrt{d}}$。

树之间选择不同特征的概率增加，降低相关性。

**数学分析**：

设单个树的方差为 $\sigma^2$，树之间的相关性为 $\rho$。

随机森林的方差：
$$\text{Var}_{RF} = \rho_{RF} \sigma^2 + \frac{1-\rho_{RF}}{M}\sigma^2$$

由于 $\rho_{RF} < \rho_{Bagging}$，随机森林的方差更低。

### 3.3 随机森林的特性

**优点**：
- 训练和预测都高效（可并行）
- 不容易过拟合（袋外误差可估计）
- 可处理高维数据
- 可评估特征重要性

**缺点**：
- 解释性不如单棵树
- 对某些数据可能不如Boosting

**袋外（OOB）误差估计**：

对于每个样本，使用未包含该样本的树进行预测，计算误差。

这提供了无偏的泛化误差估计，无需单独验证集。

**特征重要性**：

两种方法：

1. **基于不纯度**：特征在所有树中降低的不纯度之和
2. **基于排列**：随机打乱特征值，观察误差增加多少

### 3.4 随机森林为什么不容易过拟合？

**直觉**：树很多，但每棵树都不同。

**数学解释**：

单棵树过拟合：深度很大，完美拟合训练数据，但泛化差。

随机森林：
- 每棵树只看到部分数据（Bootstrap）
- 每个分裂只考虑部分特征
- 树之间高度多样化
- 平均/投票消除过拟合的影响

**关键**：多样性 + 平均 = 降低方差，不过拟合。

---

## 第四部分：梯度提升决策树（GBDT）——Boosting的巅峰

### 4.1 梯度提升的核心思想

**回顾AdaBoost**：

- 重新加权样本
- 模型权重由误差决定
- 适用于分类

**梯度提升**：

将提升视为**梯度下降**在函数空间中的优化。

**目标**：

最小化损失函数：
$$\min_F \sum_{i=1}^{n} L(y_i, F(x_i))$$

其中 $F$ 是模型（函数）。

**梯度下降视角**：

在参数空间中：
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

在函数空间中：
$$F_{t+1} = F_t - \eta \nabla_F L(F_t)$$

**关键问题**：$\nabla_F L$ 是什么？

**负梯度方向**：

对于第 $i$ 个样本，损失关于预测的梯度：

$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

这个负梯度 $r_i$ 称为**伪残差（pseudo-residual）**。

**算法**：

**初始化**：$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$

**For** $m = 1, 2, ..., M$：

1. 计算伪残差：
   $$r_i = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F(x_i)}$$

2. 训练弱学习器 $h_m$ 拟合残差 $(x_i, r_i)$

3. 寻找最优步长：
   $$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

4. 更新模型：
   $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

### 4.2 不同损失函数的伪残差

**回归：均方误差**

$$L(y, F) = \frac{1}{2}(y - F)^2$$

伪残差：
$$r = -\frac{\partial L}{\partial F} = y - F$$

**解释**：伪残差就是真实残差！GBDT拟合残差。

**回归：绝对误差**

$$L(y, F) = |y - F|$$

伪残差：
$$r = -\frac{\partial L}{\partial F} = \text{sign}(F - y)$$

**解释**：伪残差是符号，表示残差的方向。

**分类：指数损失（AdaBoost）**

$$L(y, F) = e^{-yF}$$

伪残差：
$$r = -\frac{\partial L}{\partial F} = y e^{-yF}$$

**分类：对数损失（Log Loss）**

$$L(y, F) = \log(1 + e^{-yF})$$

伪残差：
$$r = \frac{y}{1 + e^{yF}}$$

### 4.3 GBDT的完整算法

**以回归为例**：

**输入**：训练数据 $\{(x_i, y_i)\}_{i=1}^{n}$，损失函数 $L(y, F)$，迭代次数 $M$

**Step 1**：初始化
$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

对于MSE损失，$F_0(x) = \bar{y}$（均值）。

**Step 2**：For $m = 1, 2, ..., M$：

2a. 计算伪残差：
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

2b. 拟合回归树 $h_m$ 到 $\{(x_i, r_{im})\}$

2c. 对于每个叶节点 $j$，计算最优输出：
$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

2d. 更新模型：
$$F_m(x) = F_{m-1}(x) + \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{1}(x \in R_{jm})$$

其中 $J_m$ 是第 $m$ 棵树的叶节点数，$R_{jm}$ 是第 $j$ 个叶节点区域。

**输出**：$F_M(x)$

### 4.4 GBDT vs AdaBoost

| 特性 | AdaBoost | GBDT |
|------|----------|------|
| 损失函数 | 指数损失 | 任意可微损失 |
| 权重调整 | 样本权重 | 拟合残差 |
| 模型权重 | 由误差决定 | 线搜索最优 |
| 异常值 | 敏感 | 可选择鲁棒损失 |
| 适用任务 | 主要是分类 | 分类和回归 |

**GBDT的优势**：
- 可以使用任意可微损失函数
- 对异常值更鲁棒（使用Huber损失等）
- 更灵活

### 4.5 GBDT的正则化

**GBDT容易过拟合**，需要正则化。

**方法1：学习率（Shrinkage）**

$$F_m(x) = F_{m-1}(x) + \nu \cdot \gamma_m h_m(x)$$

其中 $\nu \in (0, 1]$ 是学习率。

**直觉**：每一步走得小一点，需要更多步（更多树），但每一步更稳健。

**方法2：子采样**

每轮只使用部分样本训练树（类似Bagging）。

**方法3：树的深度限制**

限制每棵树的最大深度，防止过拟合。

**方法4：早停**

监控验证误差，在验证误差上升时停止。

### 4.6 XGBoost：GBDT的工程优化

**XGBoost**（eXtreme Gradient Boosting）是对GBDT的全面优化。

**目标函数**：

$$\mathcal{L} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)$$

其中正则化项：
$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$：叶节点数
- $w_j$：叶节点输出
- $\gamma$：叶节点数惩罚
- $\lambda$：叶节点输出惩罚

**二阶泰勒展开**：

XGBoost对损失函数进行二阶泰勒展开：

$$L(y_i, \hat{y}_i^{(t-1)} + h_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i h_t(x_i) + \frac{1}{2} h_i h_t^2(x_i)$$

其中：
- $g_i = \frac{\partial L}{\partial \hat{y}}$：一阶导数（梯度）
- $h_i = \frac{\partial^2 L}{\partial \hat{y}^2}$：二阶导数（海森）

**优势**：
- 更快的收敛
- 可以自定义损失函数（只需提供一阶和二阶导数）

**分裂准则**：

对于每个分裂，计算增益：

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

其中：
- $G_L, G_R$：左右子节点的梯度之和
- $H_L, H_R$：左右子节点的海森之和
- $\gamma$：新叶节点的复杂度代价

**XGBoost的其他优化**：
- 列采样（类似随机森林）
- 稀疏数据处理
- 并行化
- 缓存优化

---

## 第五部分：Bagging vs Boosting 对比

### 5.1 本质区别

| 特性 | Bagging | Boosting |
|------|---------|----------|
| 模型关系 | 并行独立 | 串行依赖 |
| 目标 | 降低方差 | 降低偏差 |
| 数据采样 | Bootstrap | 权重调整/残差拟合 |
| 模型权重 | 相等 | 不等（基于误差） |
| 过拟合风险 | 低 | 高 |
| 异常值敏感度 | 低 | 高 |

### 5.2 偏差-方差视角

**Bagging**：

$$\text{Var}_{Bagging} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

- 主要降低方差
- 偏差基本不变
- 适合高方差、低偏差的基学习器

**Boosting**：

- 主要降低偏差
- 方差可能增加（串行依赖）
- 适合低方差、高偏差的基学习器

### 5.3 选择指南

**选择Bagging（随机森林）当**：
- 数据噪声大
- 异常值多
- 需要并行训练
- 不想调太多参数

**选择Boosting（GBDT/XGBoost）当**：
- 数据质量好
- 追求最高精度
- 可以调参
- 计算资源允许串行训练

### 5.4 为什么Boosting对异常值敏感？

**AdaBoost**：

异常值被错误分类后，权重指数增长，后续模型过度关注异常值。

**GBDT**：

使用MSE损失时，异常值的残差大，后续模型过度拟合异常值。

**解决方法**：

使用鲁棒损失函数：
- Huber损失
- Quantile损失
- 对数损失

---

## 第六部分：代码示例

### 6.1 决策树可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# 训练决策树
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=['x1', 'x2'], class_names=['Class 0', 'Class 1'],
          filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')

plt.figure(figsize=(15, 5))

# 不同深度的决策树
for i, depth in enumerate([1, 3, 10]):
    plt.subplot(1, 3, i+1)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X, y)
    plot_decision_boundary(tree, X, y, f'Depth={depth}')
    
plt.tight_layout()
plt.show()
```

### 6.2 信息增益计算

```python
import numpy as np

def entropy(y):
    """计算熵"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(X, y, feature_idx, threshold):
    """计算信息增益"""
    # 分裂前的熵
    H_before = entropy(y)
    
    # 分裂
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    n_total = len(y)
    
    if n_left == 0 or n_right == 0:
        return 0
    
    # 分裂后的熵
    H_left = entropy(y[left_mask])
    H_right = entropy(y[right_mask])
    H_after = (n_left / n_total) * H_left + (n_right / n_total) * H_right
    
    return H_before - H_after

# 示例
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

print(f"原始熵: {entropy(y):.4f}")

# 计算不同分裂点的信息增益
for threshold in [1.5, 2.5, 3.5, 4.5]:
    ig = information_gain(X, y, 0, threshold)
    print(f"特征0, 阈值{threshold}: 信息增益 = {ig:.4f}")
```

### 6.3 随机森林 vs GBDT

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 随机森林
rf_train_scores = []
rf_test_scores = []
n_estimators_range = range(1, 101, 10)

for n_est in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    rf_test_scores.append(accuracy_score(y_test, rf.predict(X_test)))

# GBDT
gb_train_scores = []
gb_test_scores = []

for n_est in n_estimators_range:
    gb = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
    gb.fit(X_train, y_train)
    gb_train_scores.append(accuracy_score(y_train, gb.predict(X_train)))
    gb_test_scores.append(accuracy_score(y_test, gb.predict(X_test)))

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n_estimators_range, rf_train_scores, 'b-', label='训练集')
plt.plot(n_estimators_range, rf_test_scores, 'r-', label='测试集')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('随机森林')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(n_estimators_range, gb_train_scores, 'b-', label='训练集')
plt.plot(n_estimators_range, gb_test_scores, 'r-', label='测试集')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('GBDT')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"随机森林最佳测试准确率: {max(rf_test_scores):.4f}")
print(f"GBDT最佳测试准确率: {max(gb_test_scores):.4f}")
```

### 6.4 XGBoost示例

```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 转换为DMatrix（XGBoost的数据结构）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 参数设置
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}

# 训练
evals_result = {}
model = xgb.train(params, dtrain, num_boost_round=100,
                   evals=[(dtrain, 'train'), (dtest, 'test')],
                   evals_result=evals_result,
                   verbose_eval=20)

# 预测
y_pred = (model.predict(dtest) > 0.5).astype(int)
print(f"\n测试准确率: {accuracy_score(y_test, y_pred):.4f}")

# 特征重要性
importance = model.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\n前5个重要特征:")
for i, (feat, imp) in enumerate(sorted_importance[:5]):
    print(f"{i+1}. {feat}: {imp:.2f}")
```
---

## 第七部分：核心直觉总结

### 7.1 决策树的本质

**分裂准则的本质**：选择让子节点"最纯"的特征。

- 信息增益：降低不确定性（熵）
- 基尼增益：降低不纯度（基尼）

**剪枝的本质**：控制模型复杂度，防止过拟合。

### 7.2 集成学习的本质

**Bagging的本质**：降低方差。

- 通过Bootstrap降低模型相关性
- 通过平均/投票消除随机误差
- 适合高方差模型（如深决策树）

**Boosting的本质**：降低偏差。

- 通过序列学习纠正错误
- 每个新模型专注于残差
- 适合高偏差模型（如浅决策树）

### 7.3 随机森林 vs GBDT

| 特性 | 随机森林 | GBDT |
|------|---------|------|
| 结构 | 并行 | 串行 |
| 基学习器 | 深树 | 浅树 |
| 优化目标 | 方差 | 偏差 |
| 异常值 | 鲁棒 | 敏感 |
| 调参 | 简单 | 复杂 |
| 精度上限 | 较低 | 较高 |

### 7.4 实践指南

**决策树**：
- 深度控制防止过拟合
- 适合可解释性要求高的场景

**随机森林**：
- 默认参数效果好
- 适合快速建模
- 特征重要性评估

**GBDT/XGBoost**：
- 精度最高
- 需要调参
- 适合竞赛和生产环境

---

## 思考题

### 问题1：为什么随机森林不容易过拟合？

<details>
<summary>点击查看提示</summary>

从方差的角度思考。随机森林如何降低方差？树之间的相关性如何影响方差？
</details>

<details>
<summary>点击查看答案</summary>

**答案：随机森林通过多样性和平均来降低方差，过拟合的单棵树被平均抵消。**

**单棵树为什么过拟合？**

决策树可以无限分裂，直到每个叶节点只有一个样本。这时：
- 训练误差 = 0
- 测试误差很高

过拟合的原因：树太深，学习了噪声。

**随机森林为什么不过拟合？**

**1. Bootstrap抽样降低相关性**

每棵树只看到约63.2%的样本，不同树看到不同的样本。

**2. 随机特征选择进一步降低相关性**

每个分裂只考虑 $\sqrt{d}$ 个随机特征，不同树选择不同的分裂特征。

**3. 数学分析**

设单棵树的方差为 $\sigma^2$，树之间的相关性为 $\rho$，共 $M$ 棵树。

随机森林的方差：
$$\text{Var}_{RF} = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$$

**关键**：$\rho$ 很小（由于Bootstrap和特征随机），方差主要由第一项 $\rho \sigma^2$ 决定，而不是 $\sigma^2$。

**4. 过拟合被平均抵消**

假设有100棵树，每棵树在某个噪声特征上学到了不同的"规律"。平均后，这些随机噪声相互抵消。

**5. 袋外误差估计**

随机森林的袋外误差是泛化误差的无偏估计。随着树增加：
- 袋外误差趋于稳定，不会持续下降
- 说明不会过拟合

**实验验证**：

```python
# 随机森林：增加树的数量
for n_trees in [10, 50, 100, 500, 1000]:
    rf = RandomForestClassifier(n_estimators=n_trees)
    rf.fit(X_train, y_train)
    print(f"n_trees={n_trees}: train={train_acc:.4f}, test={test_acc:.4f}")

# 结果：测试误差趋于稳定，不会上升
```

**对比单棵树**：

```python
# 单棵树：增加深度
for depth in [1, 3, 5, 10, 20, None]:
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    # ...

# 结果：深度增加，训练误差下降，测试误差先降后升（过拟合）
```

**总结**：

随机森林不过拟合的原因：
1. **多样性**：Bootstrap + 特征随机 → 低相关性
2. **平均**：方差降低，噪声抵消
3. **自助聚合**：每棵树只看到部分数据

</details>

### 问题2：Boosting为何对异常值敏感？如何缓解？

<details>
<summary>点击查看答案</summary>

**答案：Boosting会让异常值权重增加，后续模型过度关注异常值。可通过鲁棒损失函数缓解。**

**AdaBoost的敏感机制**：

**权重更新**：
$$w_i \leftarrow w_i \cdot \exp(-\alpha_m y_i f_m(x_i))$$

- 预测正确：权重降低
- 预测错误：权重增加

**异常值的影响**：

假设有一个异常样本，被所有模型都预测错误：

1. 第1轮：权重增加
2. 第2轮：权重继续增加（因为模型专注于高权重样本）
3. ...
4. 第M轮：权重指数增长

最终，所有模型都专注于这一个异常样本，严重过拟合。

**GBDT的敏感机制**：

**伪残差**（MSE损失）：
$$r_i = y_i - F(x_i)$$

异常值（$|y_i - F(x_i)|$ 大）的残差大，后续树会过度拟合这些大残差。

**直观理解**：

假设正常样本残差在 [-1, 1] 范围，异常样本残差为 100。

GBDT会花很多树去拟合这个 100 的残差，而忽略其他样本。

**缓解方法**：

**1. 使用鲁棒损失函数**

**Huber损失**：
$$L(y, F) = \begin{cases}
\frac{1}{2}(y - F)^2 & \text{if } |y - F| \leq \delta \\
\delta |y - F| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

- 小残差：MSE（快速收敛）
- 大残差：绝对误差（限制影响）

**Quantile损失**：
$$L(y, F) = \begin{cases}
\alpha(y - F) & \text{if } y \geq F \\
(1-\alpha)(F - y) & \text{otherwise}
\end{cases}$$

用于分位数回归，对异常值不敏感。

**2. 限制树的深度**

浅树（深度=3-5）不会过度拟合异常值。

**3. 降低学习率**

$$F_m = F_{m-1} + \nu \cdot h_m$$

小的学习率 $\nu$ 减缓对异常值的过拟合。

**4. 子采样**

每轮只用部分样本，减少异常值被选中概率。

**5. 早停**

监控验证误差，在验证误差上升时停止。

**代码示例**：

```python
# 使用Huber损失的GBDT
from sklearn.ensemble import GradientBoostingRegressor

# 敏感版本（MSE损失）
gb_mse = GradientBoostingRegressor(loss='squared_error')

# 鲁棒版本（Huber损失）
gb_huber = GradientBoostingRegressor(loss='huber', alpha=0.9)

# 在有异常值的数据上比较
# gb_huber 表现更稳定
```

**总结**：

Boosting对异常值敏感是因为：
- AdaBoost：指数权重增长
- GBDT：大残差主导

缓解方法：
- 鲁棒损失函数（Huber、Quantile）
- 限制树深度
- 降低学习率
- 子采样
- 早停

</details>

### 问题3：信息增益和信息增益比的区别是什么？为什么需要增益比？

<details>
<summary>点击查看答案</summary>

**答案：信息增益偏向取值多的特征，增益比通过除以特征熵来惩罚。**

**信息增益的定义**：

$$IG(Y, A) = H(Y) - H(Y|A)$$

**问题：偏向取值多的特征**

**极端例子**：

考虑"样本ID"特征，每个样本有唯一ID。

分裂后每个子节点只有一个样本，熵为0：
$$H(Y|ID) = 0$$

信息增益：
$$IG(Y, ID) = H(Y) - 0 = H(Y)$$

这是最大可能的信息增益！

但这个分裂毫无意义——每个叶节点只有一个样本，无法泛化。

**为什么偏向取值多的特征？**

信息增益衡量的是"分裂后的纯度提升"。

取值越多，分裂越多，子节点越容易纯净（每个子节点样本少）。

但这也意味着：
- 每个子节点样本少，统计不可靠
- 容易过拟合
- 泛化能力差

**信息增益比的定义**：

$$IGR(Y, A) = \frac{IG(Y, A)}{H(A)} = \frac{H(Y) - H(Y|A)}{H(A)}$$

其中 $H(A)$ 是特征 $A$ 的熵：

$$H(A) = -\sum_{a \in A} P(A=a) \log_2 P(A=a)$$

**惩罚机制**：

- 取值均匀分布：$H(A)$ 大，惩罚大
- 取值集中：$H(A)$ 小，惩罚小

**"样本ID"特征**：

$$H(ID) = -\sum_{i=1}^{n} \frac{1}{n} \log_2 \frac{1}{n} = \log_2 n$$

当 $n$ 很大时，$H(ID)$ 很大，信息增益比趋近于0。

**比较**：

| 特性 | 信息增益 | 信息增益比 |
|------|---------|-----------|
| 偏向 | 取值多的特征 | 无明显偏向 |
| 适用场景 | 特征取值数相近 | 特征取值数差异大 |
| 算法 | ID3 | C4.5 |

**实践建议**：

- 特征取值数相近时，两者效果类似
- 特征取值数差异大时，使用信息增益比
- C4.5默认使用信息增益比

**代码示例**：

```python
import numpy as np
from collections import Counter

def entropy(y):
    counts = Counter(y)
    probs = [c/len(y) for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def information_gain(X, y, feature_idx):
    """信息增益"""
    H_Y = entropy(y)
    values = set(X[:, feature_idx])
    H_Y_given_A = 0
    for v in values:
        mask = X[:, feature_idx] == v
        weight = sum(mask) / len(y)
        H_Y_given_A += weight * entropy(y[mask])
    return H_Y - H_Y_given_A

def information_gain_ratio(X, y, feature_idx):
    """信息增益比"""
    ig = information_gain(X, y, feature_idx)
    H_A = entropy(X[:, feature_idx])  # 特征的熵
    if H_A == 0:
        return 0
    return ig / H_A

# 示例：比较信息增益和信息增益比
# 特征A：2个取值，均匀分布
# 特征B：10个取值，均匀分布（偏向）

# 特征B的信息增益可能更大，但增益比会惩罚
```

**总结**：

信息增益偏向取值多的特征，因为分裂越多越容易纯净。

信息增益比通过除以特征熵来惩罚取值多的特征，避免选择无意义的分裂。

</details>

### 问题4：GBDT为什么用负梯度而不是真实残差？

<details>
<summary>点击查看提示</summary>

思考不同损失函数下的"残差"。MSE损失的残差是什么？其他损失函数呢？
</details>

<details>
<summary>点击查看答案</summary>

**答案：负梯度是残差的一般化，适用于任意可微损失函数。**

**MSE损失下的残差**：

损失函数：
$$L(y, F) = \frac{1}{2}(y - F)^2$$

负梯度：
$$r = -\frac{\partial L}{\partial F} = y - F$$

**这恰好是真实残差！**

所以，对于MSE损失，GBDT拟合的就是残差。

**其他损失函数呢？**

**绝对误差损失**：
$$L(y, F) = |y - F|$$

负梯度：
$$r = -\frac{\partial L}{\partial F} = \text{sign}(F - y)$$

**这不是残差，而是残差的符号！**

**解释**：

绝对误差对异常值更鲁棒。负梯度是符号，表示"往哪个方向调整"，而不是"调整多少"。

**对数损失（二分类）**：
$$L(y, F) = \log(1 + e^{-yF})$$

负梯度：
$$r = \frac{y}{1 + e^{yF}}$$

**这不是残差，而是概率误差！**

**为什么用负梯度？**

**1. 一般化**

负梯度是残差的一般化：
- MSE损失：负梯度 = 残差
- 其他损失：负梯度是"广义残差"

**2. 优化视角**

梯度下降要求沿着负梯度方向更新：
$$F_{t+1} = F_t - \eta \nabla_F L$$

在函数空间中，梯度就是 $\frac{\partial L}{\partial F}$，负梯度是最速下降方向。

**3. 灵活性**

使用负梯度，GBDT可以用于任意可微损失函数：
- 回归：MSE、MAE、Huber
- 分类：对数损失、指数损失
- 排序：LambdaRank

**为什么不用真实残差？**

**真实残差只对MSE有意义！**

对于其他损失函数，"真实残差"没有定义。负梯度才是统一的优化方向。

**类比**：

| 损失函数 | 真实残差 | 负梯度 | 含义 |
|---------|---------|--------|------|
| MSE | $y - F$ | $y - F$ | 残差 |
| MAE | $y - F$ | $\text{sign}(F - y)$ | 残差符号 |
| Log Loss | 无定义 | $\frac{y}{1+e^{yF}}$ | 概率误差 |

**代码示例**：

```python
import numpy as np

# 不同损失函数的负梯度
def negative_gradient(y, F, loss='mse'):
    """计算负梯度"""
    if loss == 'mse':
        return y - F
    elif loss == 'mae':
        return np.sign(y - F)
    elif loss == 'log':
        return y / (1 + np.exp(y * F))
    elif loss == 'exponential':
        return y * np.exp(-y * F)

# 示例
y = np.array([1, 1, -1, -1])
F = np.array([0.5, 0.8, -0.3, 0.2])

print("MSE负梯度:", negative_gradient(y, F, 'mse'))
print("MAE负梯度:", negative_gradient(y, F, 'mae'))
print("Log Loss负梯度:", negative_gradient(y, F, 'log'))
```

**总结**：

GBDT使用负梯度而不是真实残差，因为：

1. **一般化**：负梯度适用于任意可微损失
2. **优化理论**：负梯度是最速下降方向
3. **统一框架**：回归、分类、排序都可以用同一框架

对于MSE损失，负梯度恰好等于残差，这是特殊情况，不是一般规律。

</details>

---

## 今日要点

1. **决策树的核心**：选择最优分裂特征，信息增益/增益比/基尼增益

2. **集成学习的本质**：
   - Bagging：降低方差（并行、独立）
   - Boosting：降低偏差（串行、依赖）

3. **随机森林**：Bagging + 特征随机，降低模型相关性

4. **GBDT**：梯度下降在函数空间，拟合负梯度（广义残差）

5. **XGBoost**：GBDT + 二阶近似 + 正则化 + 工程优化

6. **选择指南**：
   - 噪声大、异常值多 → 随机森林
   - 追求精度、数据干净 → GBDT/XGBoost

---

## 明日预告

Day 7 是Week 1的综合实践，我们将：

- 回顾ML理论框架
- 端到端完成一个ML项目
- 将本周学习的概念融会贯通

Week 1 总结了机器学习的核心思想，Week 2 我们将深入深度学习的世界。
