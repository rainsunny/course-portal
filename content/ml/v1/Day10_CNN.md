# Day 10: 卷积神经网络 - 让机器"看见"世界

## 核心问题

**为什么全连接网络处理图像会遇到困难？为什么卷积神经网络能够成为计算机视觉的核心架构？卷积操作的本质是什么？**

---

## 一、图像处理的核心挑战

### 1.1 全连接网络的困境

假设我们处理一张 224×224 的彩色图像（3个通道）：

**输入维度问题：**
- 输入神经元数量：224 × 224 × 3 = 150,528
- 如果第一隐藏层有 1000 个神经元
- 参数数量：150,528 × 1000 + 1000 = 150,529,000 个参数！

**三大困境：**

```
困境1：参数爆炸
├── 参数量随图像尺寸平方增长
├── 训练困难，容易过拟合
└── 计算资源需求巨大

困境2：空间结构丢失
├── 图像被展平为一维向量
├── 相邻像素的关系被破坏
└── 空间局部性信息丢失

困境3：平移不变性缺失
├── 猫在图像左上角 vs 右下角
├── 全连接网络需要分别学习
└── 无法共享特征检测能力
```

### 1.2 人类的视觉启发

人类视觉系统的特点：

**1. 局部感知**
- 视网膜上的神经元只感受视野的一小部分
- 每个神经元有有限的"感受野"
- 通过层级组合构建整体理解

**2. 空间不变性**
- 无论猫在图像哪个位置，我们都能识别
- 特征检测器应该在空间上共享
- "边缘检测器"在图像任何位置都做同样的工作

**3. 层级抽象**
```
视网膜 → V1（边缘检测）→ V2（形状组合）→ V4（复杂形状）→ IT（物体识别）
   ↓           ↓              ↓               ↓              ↓
 像素      简单特征        中级特征        高级特征        语义理解
```

---

## 二、卷积操作的本质

### 2.1 卷积的数学定义

**一维离散卷积：**

$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \cdot g[n-m]$$

**二维离散卷积（图像处理中常用）：**

$$(I * K)[i,j] = \sum_{m}\sum_{n} I[i-m, j-n] \cdot K[m, n]$$

**注意：在深度学习中，我们通常使用"互相关"而非严格卷积：**

$$S[i,j] = (I \star K)[i,j] = \sum_{m}\sum_{n} I[i+m, j+n] \cdot K[m, n]$$

**区别在于是否翻转核：**
- 严格卷积：核需要翻转（flip）
- 互相关：核不翻转
- 深度学习中统称为"卷积"，但实际是互相关

### 2.2 卷积的直观理解

**卷积核（Filter/Kernel）的本质：**

```
边缘检测核（水平）：
┌────┬────┬────┐
│ -1 │ -1 │ -1 │
├────┼────┼────┤
│  0 │  0 │  0 │
├────┼────┼────┤
│  1 │  1 │  1 │
└────┴────┴────┘

作用：检测图像中的水平边缘
数学本质：计算局部区域的梯度
```

**卷积操作过程：**

```
输入图像（5×5）        核（3×3）        输出特征图（3×3）
┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐   ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │   │ 1 │ 0 │-1 │   │ 2 │ 1 │-1 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤   ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │   │ 1 │ 0 │-1 │   │ 3 │ 0 │-2 │
├───┼───┼───┼───┼───┤   │ 1 │ 0 │-1 │   ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │   └───┴───┴───┘   │ 1 │ 2 │ 0 │
├───┼───┼───┼───┼───┤                   ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │                   │-1 │ 1 │ 1 │
├───┼───┼───┼───┼───┤                   └───┴───┴───┘
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘

计算示例（左上角）：
1×1 + 0×0 + 1×(-1) + 
0×1 + 1×0 + 0×(-1) + 
1×1 + 0×0 + 1×(-1) = 2
```

### 2.3 卷积核学习的本质

**核心洞察：核的权重是可学习的！**

```
传统计算机视觉：
├── 手工设计核（Sobel、Gabor、Haar等）
├── 需要领域专家知识
└── 特征表达能力有限

深度学习：
├── 核的权重通过反向传播学习
├── 网络自动发现最有用的特征
└── 端到端优化，无需人工设计
```

**学习的特征示例：**

| 层级 | 学习的特征 | 语义级别 |
|------|-----------|---------|
| 第1层 | 边缘、颜色斑点 | 底层特征 |
| 第2层 | 纹理、简单形状 | 中低层特征 |
| 第3层 | 眼睛、耳朵等部件 | 中层特征 |
| 第4层 | 人脸、车等物体部件 | 中高层特征 |
| 第5层 | 完整物体 | 高层语义 |

---

## 三、CNN的核心架构组件

### 3.1 卷积层（Convolutional Layer）

**关键参数：**

**1. 核大小（Kernel Size）：**
- 常用：3×3、5×5、7×7
- 奇数尺寸便于中心对称
- 小核堆叠可替代大核（两个3×3 ≈ 一个5×5）

**2. 步长（Stride）：**
- 步长为1：每次移动1个像素
- 步长为2：每次移动2个像素，输出尺寸减半
- 步长>1实现下采样

**3. 填充（Padding）：**
- 保持输出尺寸不变
- 常用："same"填充（四周补0）
- 计算：padding = (kernel_size - 1) / 2

**输出尺寸计算公式：**

$$H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} \rfloor + 1$$

其中：
- $H_{in}$：输入高度
- $P$：填充大小
- $K$：核大小
- $S$：步长

**代码示例：**

```python
import torch
import torch.nn as nn

# 基本卷积层
conv_layer = nn.Conv2d(
    in_channels=3,      # 输入通道数
    out_channels=64,    # 输出通道数（核的数量）
    kernel_size=3,      # 核大小
    stride=1,           # 步长
    padding=1           # 填充
)

# 输入：batch_size × 3 × 224 × 224
# 输出：batch_size × 64 × 224 × 224

x = torch.randn(1, 3, 224, 224)
output = conv_layer(x)
print(f"输入形状: {x.shape}")    # [1, 3, 224, 224]
print(f"输出形状: {output.shape}")  # [1, 64, 224, 224]
print(f"参数数量: {conv_layer.weight.numel() + conv_layer.bias.numel()}")
# 参数 = 3 × 3 × 3 × 64 + 64 = 1792
```

### 3.2 多通道卷积

**单通道 → 多通道：**

```
输入：H × W × C_in（C_in个通道）
核：C_in × K × K × C_out（C_out个核，每个核有C_in个通道）
输出：H' × W' × C_out

每个输出通道 = 所有输入通道卷积结果之和
```

**示意图：**

```
输入（3通道）         核组（64个核，每个3通道）    输出（64通道）
┌─────────┐          ┌───────────────────┐       ┌──────┐
│ R通道   │──┐       │ 核1（3通道）       │       │ 通道1│
├─────────┤  │       │  ┌───┐            │       ├──────┤
│ G通道   │──┼──────▶│  │R核│            │       │ ...  │
├─────────┤  │       │  │G核│ ──求和──▶  │       ├──────┤
│ B通道   │──┘       │  │B核│            │       │通道64│
└─────────┘          │  └───┘            │       └──────┘
                     │  ...（共64组）     │
                     └───────────────────┘
```

**为什么有效？**
- 每个输出通道学习一种特征
- 64个核 = 学习64种不同的特征
- 不同通道组合响应不同的模式

### 3.3 池化层（Pooling Layer）

**作用：降低空间维度，提取主要特征**

**最大池化（Max Pooling）：**

```
输入（4×4）           2×2最大池化          输出（2×2）
┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 4 │     ┌───────────┐      ┌───┬───┐
├───┼───┼───┼───┤     │ max(1,3,  │      │ 6 │ 8 │
│ 5 │ 6 │ 7 │ 8 │     │     5,6)=6 │  ──▶├───┼───┤
├───┼───┼───┼───┤     ├───────────┤      │10 │12 │
│ 2 │ 4 │ 1 │ 3 │     │ max(...)  │      └───┴───┘
├───┼───┼───┼───┤     └───────────┘
│ 9 │10 │11 │12 │
└───┴───┴───┴───┘
```

**平均池化（Average Pooling）：**
- 计算局部区域的平均值
- 保留更多信息，但不如最大池化突出显著特征

**全局池化（Global Pooling）：**
- 将整个特征图压缩为一个值
- 常用于替代全连接层
- 减少参数，防止过拟合

```python
# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 输入：1 × 64 × 112 × 112
# 输出：1 × 64 × 56 × 56

# 全局平均池化
global_pool = nn.AdaptiveAvgPool2d(1)

# 输入：1 × 64 × 112 × 112
# 输出：1 × 64 × 1 × 1
```

---

## 四、CNN的关键设计原理

### 4.1 参数共享（Parameter Sharing）

**核心思想：同一个核在不同位置共享权重**

```
全连接层：
├── 每个位置需要独立的权重
├── 参数量 = 输入维度 × 输出维度
└── 无法利用空间结构

卷积层：
├── 同一个核滑过整个图像
├── 参数量 = 核大小 × 输入通道 × 输出通道
└── 显著减少参数
```

**对比示例：**

```
输入：224 × 224 × 3
输出：224 × 224 × 64

全连接参数：
150,528 × 64 × 224 × 224 ≈ 485亿参数

卷积参数（3×3核）：
3 × 3 × 3 × 64 = 1,728 参数

减少倍数：约2800万倍！
```

### 4.2 局部连接（Local Connectivity）

**核心思想：每个神经元只连接局部区域**

```
感受野（Receptive Field）：
├── 每个输出神经元只"看到"输入的一小部分
├── 浅层：小感受野，检测局部特征
└── 深层：大感受野，检测全局特征
```

**感受野的计算：**

对于第 $l$ 层，其感受野大小：

$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

**示例计算：**

```
Layer 1: 3×3卷积，stride=1  → RF = 3
Layer 2: 3×3卷积，stride=1  → RF = 3 + (3-1)×1 = 5
Layer 3: 3×3卷积，stride=1  → RF = 5 + (3-1)×1 = 7

堆叠3个3×3卷积 = 1个7×7卷积的感受野
但参数量：3×(3×3) = 27 vs 7×7 = 49
```

### 4.3 平移等变性与不变性

**平移等变性（Translation Equivariance）：**

$$f(g(x)) = g(f(x))$$

如果输入平移，输出相应平移

**直觉：**
- 猫的特征在左上角 → 相应的激活也在左上角
- 猫的特征移到右下角 → 激活也移到右下角
- 卷积操作保持空间对应关系

**平移不变性（Translation Invariance）：**

通过池化实现一定程度的平移不变性：
- 无论猫在哪里，池化后都能检测到
- 最大池化保留最显著的特征

```
卷积层：平移等变性
    ↓
池化层：引入平移不变性
    ↓
组合效果：对位置变化有鲁棒性
```

---

## 五、经典CNN架构演进

### 5.1 LeNet-5（1998）：开创者

```
输入：32×32灰度图
    ↓
C1: 卷积 6@28×28 (5×5核)
    ↓
S2: 池化 6@14×14 (2×2)
    ↓
C3: 卷积 16@10×10 (5×5核)
    ↓
S4: 池化 16@5×5 (2×2)
    ↓
C5: 卷积 120@1×1 (5×5核)
    ↓
F6: 全连接 84
    ↓
输出：10类（数字0-9）

总参数：约6万
```

**核心贡献：**
- 确立了CNN的基本架构模式
- 成功应用于手写数字识别
- 证明了反向传播训练CNN的可行性

### 5.2 AlexNet（2012）：深度学习复兴

```
输入：227×227×3
    ↓
Conv1: 96@55×55 (11×11, stride=4) + ReLU + MaxPool
    ↓
Conv2: 256@27×27 (5×5) + ReLU + MaxPool
    ↓
Conv3: 384@13×13 (3×3) + ReLU
    ↓
Conv4: 384@13×13 (3×3) + ReLU
    ↓
Conv5: 256@13×13 (3×3) + ReLU + MaxPool
    ↓
FC6: 4096 + Dropout
    ↓
FC7: 4096 + Dropout
    ↓
FC8: 1000（ImageNet类别）

总参数：约6000万
```

**关键创新：**
1. **ReLU激活函数**：解决梯度消失，加速训练
2. **Dropout**：防止过拟合
3. **数据增强**：随机裁剪、水平翻转
4. **GPU训练**：使用双GPU并行

### 5.3 VGG（2014）：小核堆叠

**核心思想：用小核堆叠替代大核**

```
VGG-16架构：
┌─────────────────────────────────┐
│ 输入 224×224×3                  │
├─────────────────────────────────┤
│ 2×Conv3-64  + MaxPool          │ → 112×112×64
│ 2×Conv3-128 + MaxPool          │ → 56×56×128
│ 3×Conv3-256 + MaxPool          │ → 28×28×256
│ 3×Conv3-512 + MaxPool          │ → 14×14×512
│ 3×Conv3-512 + MaxPool          │ → 7×7×512
├─────────────────────────────────┤
│ FC-4096 + Dropout              │
│ FC-4096 + Dropout              │
│ FC-1000                        │
└─────────────────────────────────┘

参数量：约1.38亿
```

**为什么3×3更好？**

| 对比项 | 一个7×7核 | 三个3×3核堆叠 |
|--------|----------|--------------|
| 感受野 | 7×7 | 7×7 |
| 参数量 | C×C×49 | 3×C×C×9 = 27C² |
| 非线性 | 1次 | 3次（更强表达力）|
| 参数比 | 100% | 55% |

**优势：**
- 参数更少
- 非线性变换更多，表达力更强
- 更容易优化

### 5.4 ResNet（2015）：残差学习的革命

**核心问题：网络越深越好吗？**

```
实验发现：
网络深度 ↑ → 训练误差 ↑（退化问题）
这不是过拟合！（测试误差也上升）
而是优化困难：深层网络更难训练
```

**残差学习解决方案：**

$$\mathcal{F}(x) = H(x) - x$$
$$H(x) = \mathcal{F}(x) + x$$

**残差块结构：**

```
输入 x ─────────────────────┐
    ↓                        │
Conv → BN → ReLU            │
    ↓                        │
Conv → BN                   │
    ↓                        │
    (+) ←────────────────────┘
    ↓
  ReLU
    ↓
  输出
```

**为什么有效？**

```
普通网络：学习 H(x)
├── 如果恒等映射是最优解：H(x) = x
├── 网络需要学习：把权重推向0来得到x
└── 困难！多层变换后很难"什么都不做"

残差网络：学习 F(x) = H(x) - x
├── 如果恒等映射是最优解：H(x) = x
├── 只需要学习：F(x) = 0
├── 把残差块权重推向0即可
└── 容易！初始权重就接近0
```

**梯度传播优势：**

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot (1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F(x_i))$$

恒为1的梯度通道确保梯度可以直接传播到浅层！

**ResNet架构变体：**

| 模型 | 层数 | 参数量 | Top-5错误率 |
|------|------|--------|------------|
| ResNet-18 | 18 | 11.7M | 10.76% |
| ResNet-34 | 34 | 21.8M | 8.74% |
| ResNet-50 | 50 | 25.6M | 7.13% |
| ResNet-101 | 101 | 44.5M | 6.44% |
| ResNet-152 | 152 | 60.2M | 5.71% |

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x  # 保存输入作为残差
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = self.relu(out)
        return out
```


### 5.5 现代架构：DenseNet、EfficientNet

**DenseNet：密集连接**

每一层与之前所有层连接：

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

```
Layer 0: 输入
Layer 1: 接收 Layer 0
Layer 2: 接收 Layer 0, 1
Layer 3: 接收 Layer 0, 1, 2
...
```

**优势：**
- 特征重用，参数效率高
- 梯度流动更顺畅
- 深层网络更容易训练

**EfficientNet：复合缩放**

同时缩放深度、宽度、分辨率：

$$\text{depth}: d = \alpha^\phi$$
$$\text{width}: w = \beta^\phi$$
$$\text{resolution}: r = \gamma^\phi$$

约束条件：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$, $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$

---

## 六、CNN的理论分析

### 6.1 为什么CNN适合图像？

**归纳偏置（Inductive Bias）：**

CNN内置了三个关键假设：

| 假设 | 实现 | 效果 |
|------|------|------|
| 局部性 | 小核卷积 | 相邻像素相关性高 |
| 平移等变性 | 权重共享 | 特征位置无关 |
| 层级性 | 深层网络 | 底层→高层抽象 |

```
归纳偏置的作用：
├── 限制假设空间
├── 引入先验知识
├── 减少所需数据量
└── 提高泛化能力

代价：
└── 如果假设不成立，性能受限
```

**与全连接网络对比：**

```
全连接网络：
├── 归纳偏置：几乎无
├── 假设空间：极大
├── 数据需求：巨大
└── 图像表现：差（丢失空间结构）

CNN：
├── 归纳偏置：局部性、平移等变
├── 假设空间：受限但合理
├── 数据需求：相对较少
└── 图像表现：优秀
```

### 6.2 感受野的深入理解

**有效感受野 vs 理论感受野：**

理论感受野：神经元理论上能"看到"的输入范围

有效感受野：实际影响神经元的输入区域（呈高斯分布）

**计算有效感受野：**

对输出神经元 $y$ 关于输入 $x_{i,j}$ 的梯度：

$$\frac{\partial y}{\partial x_{i,j}}$$

分布近似高斯，中心区域贡献最大！

**增大感受野的策略：**

```python
# 策略1：大核卷积
conv_large = nn.Conv2d(64, 64, kernel_size=7, padding=3)
# 感受野：7×7，参数：64×64×49

# 策略2：堆叠小核（更优）
conv_stack = nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1)
)
# 感受野：7×7，参数：3×64×64×9，更多非线性

# 策略3：空洞卷积（Dilated Convolution）
conv_dilated = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
# 感受野：5×5，参数：64×64×9
# 在核元素间插入空洞，扩大感受野
```

**空洞卷积示意：**

```
普通3×3卷积：        空洞率=2的3×3卷积：
┌───┬───┬───┐        ┌───┬───┬───┬───┬───┐
│ 1 │ 1 │ 1 │        │ 1 │ 0 │ 1 │ 0 │ 1 │
├───┼───┼───┤        ├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 1 │   →    │ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┤        ├───┼───┼───┼───┼───┤
│ 1 │ 1 │ 1 │        │ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┘        ├───┼───┼───┼───┼───┤
感受野：3×3           │ 0 │ 0 │ 0 │ 0 │ 0 │
                      ├───┼───┼───┼───┼───┤
                      │ 1 │ 0 │ 1 │ 0 │ 1 │
                      └───┴───┴───┴───┴───┘
                      感受野：5×5
```

### 6.3 特征可视化

**第一层卷积核可视化：**

```python
import matplotlib.pyplot as plt

def visualize_first_layer(conv_layer, num_kernels=16):
    """可视化第一层卷积核"""
    kernels = conv_layer.weight.data
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            # 取第i个核，转为RGB格式
            kernel = kernels[i].permute(1, 2, 0).cpu().numpy()
            # 归一化到[0,1]
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            ax.imshow(kernel)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 使用示例
visualize_first_layer(model.conv1)
```

**典型学习结果：**

- **第一层**：边缘检测器（不同方向、频率）
- **第二层**：纹理模式、角点
- **第三层**：简单形状（圆形、椭圆形）
- **第四层**：部件特征（眼睛、轮子）
- **第五层**：完整物体部件

**特征图可视化：**

```python
def visualize_feature_maps(feature_map, num_channels=16):
    """可视化特征图"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.imshow(feature_map[0, i].detach().cpu().numpy(), cmap='viridis')
            ax.set_title(f'Channel {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## 七、CNN的完整实现

### 7.1 从零实现简单CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """简单的CNN分类器"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 输出: 32×32×32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 输出: 64×16×16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 输出: 128×8×8
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32×32 → 16×16
        
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16×16 → 8×8
        
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8×8 → 4×4
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# 测试模型
model = SimpleCNN(num_classes=10)
x = torch.randn(16, 3, 32, 32)
output = model(x)
print(f'输入形状: {x.shape}')
print(f'输出形状: {output.shape}')
print(f'参数数量: {sum(p.numel() for p in model.parameters()):,}')
```

### 7.2 实现ResNet

```python
class BasicBlock(nn.Module):
    """ResNet基础块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果维度不匹配，需要调整残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """简化版ResNet"""
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
```

### 7.3 迁移学习实践

**预训练模型使用：**

```python
import torchvision.models as models

# 加载预训练ResNet
model = models.resnet50(pretrained=True)

# 方法1：特征提取（冻结卷积层）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10类分类

# 方法2：微调（解冻部分层）
for param in model.parameters():
    param.requires_grad = True

# 冻结前几层（只训练后面的层）
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False
```

**迁移学习最佳实践：**

```python
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # 使用预训练骨干网络
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 不同学习率策略
def get_parameter_groups(model):
    """为不同层设置不同学习率"""
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'features' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': 1e-4},   # 骨干网络用小学习率
        {'params': classifier_params, 'lr': 1e-3}   # 分类头用大学习率
    ]
```

---

## 八、思考题

<details>
<summary>思考题1：为什么CNN的参数共享假设对图像有效，但对其他类型数据可能不合适？</summary>

**答案：**

**参数共享的核心假设：**
CNN假设"相同特征可能出现在图像的任何位置"，这基于图像的特殊性质：

1. **平移不变性的物理意义**
   - 图像中的"边缘"无论出现在左上角还是右下角，本质上都是边缘
   - 物体可以在图像中任意位置出现
   - 同一特征检测器（如边缘检测）适用于全图

2. **图像数据的特殊性**
   ```
   图像数据特点：
   ├── 空间局部性：相邻像素高度相关
   ├── 平移对称性：物体位置变化不应改变识别结果
   └── 层级结构：局部特征→全局模式
   ```

**对其他数据不合适的原因：**

1. **序列数据（如文本）**
   - "猫"在句首和句尾的含义可能不同
   - 位置信息本身就很重要
   - 解决方案：位置编码（Transformer）

2. **图数据（如社交网络）**
   - 节点的"邻居"结构各不相同
   - 没有规则的网格结构
   - 解决方案：图卷积网络（GCN）

3. **时间序列数据**
   - 时间位置有特殊含义（季节性、趋势）
   - 早期和晚期数据重要性不同
   - 解决方案：带位置权重的网络

**本质洞察：**
- CNN的归纳偏置适合具有**局部相关性**和**平移对称性**的数据
- 对于不满足这些假设的数据，需要设计合适的归纳偏置
- 这就是为什么不同领域发展出了不同的架构（RNN、GCN、Transformer等）

</details>

<details>
<summary>思考题2：ResNet解决了深层网络的退化问题，但为什么简单地增加层数会导致退化？退化现象和过拟合有什么本质区别？</summary>

**答案：**

**退化现象（Degradation Problem）：**

实验观察：随着网络加深，**训练误差**和**测试误差**都上升

```
网络深度    训练误差    测试误差
20层        1.5%        2.0%
56层        2.5%        3.0%  ← 训练和测试都变差！
```

**与过拟合的区别：**

| 现象 | 训练误差 | 测试误差 | 原因 |
|------|---------|---------|------|
| 过拟合 | 低 | 高 | 模型过于复杂，记住训练数据 |
| 退化 | 高 | 高 | 优化困难，无法找到好的解 |

**为什么加深会导致退化？**

1. **恒等映射的困难**
   - 如果浅层网络已经很好，深层网络至少应该能达到相同性能
   - 理论上，后加的层可以学习"恒等映射"（什么都不做）
   - 但实际训练中，很难让普通网络学习恒等映射

2. **梯度传播问题**
   $$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_n} \prod_{i=0}^{n-1} \frac{\partial x_{i+1}}{\partial x_i}$$
   
   - 每层的梯度是小于1的数（ReLU会截断负值）
   - 多层连乘导致梯度指数衰减
   - 浅层几乎收不到梯度信号

3. **优化景观的复杂性**
   ```
   深层网络的损失景观：
   ├── 更多局部最小值
   ├── 更平坦的区域
   ├── 梯度方向更不明确
   └── 初始化的影响更大
   ```

**ResNet的解决方案：**

$$H(x) = F(x) + x$$

1. **恒等映射变得简单**
   - 只需要学习 F(x) = 0（权重初始化接近0）
   - 而不是让普通网络学习 H(x) = x

2. **梯度传播通道**
   $$x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$$
   $$\frac{\partial x_L}{\partial x_l} = 1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}F(x_i)$$
   
   - "1"确保梯度至少可以无损传播
   - 即使 F(x) 的梯度很小，梯度也不会消失

**关键洞察：**
- 退化问题不是能力不足，而是优化困难
- ResNet不是增加了网络的能力（已经足够），而是改善了优化的难度
- 这是一个"让网络更容易学习"的问题，而不是"让网络更强大"的问题

</details>

<details>
<summary>思考题3：感受野大小和网络层数的关系是什么？如何设计网络来最大化感受野同时保持参数效率？</summary>

**答案：**

**感受野计算公式：**

对于第 $l$ 层：
$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

其中 $K_l$ 是核大小，$S_i$ 是第 $i$ 层的步长。

**感受野增长模式：**

```python
def compute_receptive_field(layers):
    """
    layers: [(kernel_size, stride), ...]
    返回每层感受野大小
    """
    rf = 1  # 输入层感受野为1
    stride_product = 1
    rf_history = [rf]
    
    for k, s in layers:
        rf = rf + (k - 1) * stride_product
        stride_product *= s
        rf_history.append(rf)
    
    return rf_history

# 示例1：3个3×3卷积，步长均为1
layers1 = [(3, 1), (3, 1), (3, 1)]
print(compute_receptive_field(layers1))
# [1, 3, 5, 7]

# 示例2：交替使用池化
layers2 = [(3, 1), (2, 2), (3, 1), (2, 2)]
print(compute_receptive_field(layers2))
# [1, 3, 4, 6, 8]
```

**最大化感受野的策略：**

1. **使用池化层（下采样）**
   - 池化后的卷积，每个像素对应输入更大区域
   - 代价：空间分辨率降低

2. **增大步长**
   - Stride > 1 加速感受野增长
   - 代价：信息损失

3. **空洞卷积（Dilated Convolution）**
   ```python
   # 不增加参数，扩大感受野
   conv_dilated = nn.Conv2d(64, 64, 3, dilation=2, padding=2)
   # 空洞率2 → 实际感受野5×5，但参数仍是3×3
   ```

4. **深度可分离卷积 + 大核**
   - 先逐深度卷积（参数少）
   - 再逐点卷积（1×1）
   - 可以使用更大的核

**参数效率对比：**

| 方法 | 感受野 | 参数量 | 计算量 |
|------|--------|--------|--------|
| 单个7×7卷积 | 7×7 | 49×C² | 49×H×W×C² |
| 3个3×3卷积堆叠 | 7×7 | 3×9×C² | 3×9×H×W×C² |
| 空洞率=3的3×3卷积 | 7×7 | 9×C² | 9×H×W×C² |

**最优设计原则：**

```
感受野最大化 + 参数效率：
├── 优先使用小核堆叠（3×3）
│   └── 更多非线性，参数更少
├── 适当使用空洞卷积
│   └── 在不增加参数的情况下扩大感受野
├── 下采样要适度
│   └── 保留足够空间信息
└── 深度可分离卷积
    └── 大核也可接受
```

**实际案例：**

```python
# 高效感受野设计
class EfficientRFBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        # 逐深度卷积（大核，少参数）
        self.depthwise = nn.Conv2d(
            channels, channels, 3, 
            padding=dilation, dilation=dilation, groups=channels
        )
        # 逐点卷积
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pointwise(out)
        return out
```

</details>

<details>
<summary>思考题4：卷积神经网络中的池化层有什么作用？为什么现代网络架构（如ResNet）倾向于减少或替代池化层？</summary>

**答案：**

**池化层的作用：**

1. **降维（下采样）**
   - 减少特征图尺寸，降低计算量
   - 后续层的感受野间接增大

2. **平移不变性**
   ```
   输入图像轻微平移 → 池化后输出可能不变
   最大池化保留最显著特征，位置信息被弱化
   ```

3. **扩大感受野**
   - 池化后，下一层每个神经元看到更大输入区域
   - 间接实现感受野增长

4. **减少过拟合**
   - 减少参数量
   - 提供一种正则化效果

**为什么现代网络减少池化？**

1. **信息丢失**
   - 最大池化：只保留最大值，丢弃其他信息
   - 平均池化：信息被稀释
   - 可能丢失重要细节

2. **步长卷积可以替代**
   ```python
   # 传统方法：卷积 + 池化
   nn.Sequential(
       nn.Conv2d(64, 128, 3, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2)  # 尺寸减半
   )
   
   # 现代方法：步长卷积
   nn.Sequential(
       nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 尺寸直接减半
       nn.ReLU()
   )
   ```
   - 步长卷积可学习，更灵活
   - 不需要额外操作

3. **任务需要精确定位**
   - 语义分割：需要像素级预测
   - 目标检测：需要精确边界框
   - 池化丢失位置信息

4. **全局平均池化替代全连接**
   ```python
   # 传统：展平 + 全连接
   nn.Sequential(
       nn.Flatten(),
       nn.Linear(512 * 7 * 7, 4096),  # 大量参数
       nn.Linear(4096, 1000)
   )
   
   # 现代：全局平均池化
   nn.Sequential(
       nn.AdaptiveAvgPool2d(1),  # (512, 7, 7) → (512, 1, 1)
       nn.Flatten(),
       nn.Linear(512, 1000)  # 参数减少很多
   )
   ```

**池化的现代角色：**

```
传统CNN（如VGG）：
├── 多次池化（5次下采样）
├── 最后特征图：7×7
└── 适合：图像分类

现代网络（如ResNet）：
├── 步长卷积替代中间池化
├── 全局平均池化替代全连接
└── 保留更多空间信息

分割/检测网络：
├── 空洞卷积保持分辨率
├── 上采样恢复尺寸
└── 最小化池化
```

**总结：**
- 池化层在早期CNN中至关重要
- 现代网络通过可学习的步长卷积实现下采样
- 全局平均池化仍然常用（减少参数）
- 任务需要精确定位时，池化要谨慎使用

</details>

---

## 九、今日要点

1. **卷积的本质**：可学习的特征提取器，通过权重共享和局部连接高效处理图像

2. **三大设计原理**：
   - 参数共享：同一核应用于全图
   - 局部连接：小感受野提取局部特征
   - 平移等变性：保持空间对应关系

3. **架构演进**：
   - LeNet → AlexNet → VGG → ResNet
   - 趋势：更深、更高效、更好优化

4. **关键技术**：
   - 残差连接解决退化问题
   - 批归一化加速训练
   - 迁移学习利用预训练知识

5. **设计原则**：
   - 小核堆叠优于大核
   - 平衡感受野和参数效率
   - 根据任务选择架构

---

## 十、明日预告

**第11天：循环神经网络（RNN）**

我们将探讨：
- 序列数据的特点和挑战
- RNN的基本原理和数学形式
- LSTM和GRU的设计思想
- 序列到序列模型
- 注意力机制的雏形

CNN处理空间数据，RNN处理时间数据——两者是深度学习的两大支柱架构。
