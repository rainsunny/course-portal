import { CourseInfo, CourseMeta, CourseVersion } from './types';

// 课程元数据 - 用于门户首页展示
export const coursesMeta: CourseMeta[] = [
  {
    id: 'management',
    title: '管理学基础',
    subtitle: '从第一性原理到实践技能',
    description: '为小团队管理者提供系统性的管理知识和实践技能培训课程，涵盖角色认知、目标管理、团队建设、沟通影响力等核心内容。',
    duration: '10-14天',
    icon: 'Users',
    color: 'primary',
    path: '/courses/management',
    hasVersions: true,
    versions: [
      { id: 'v2', name: '第二版 (经典)', description: '经典版本，内容简洁实用', path: '/courses/management/v2', default: true },
      { id: 'v3', name: '第三版 (进阶)', description: '进阶版本，内容更加丰富深入', path: '/courses/management/v3' },
      { id: 'v4', name: '第四版 (最新)', description: '最新版本，精简核心内容', path: '/courses/management/v4' },
    ]
  },
  {
    id: 'business',
    title: '商业本质与思维',
    subtitle: '从第一性原理到实战应用',
    description: '从商业世界的底层规律出发，逐层构建认知框架，掌握价值创造、竞争优势、网络效应、商业模式创新等核心概念。',
    duration: '14天',
    icon: 'TrendingUp',
    color: 'emerald',
    path: '/courses/business',
    hasVersions: true,
    versions: [
      { id: 'v1', name: '第一版', description: '完整版本，内容全面', path: '/courses/business/v1', default: true },
      { id: 'v2', name: '第二版', description: '14天系统课程', path: '/courses/business/v2' },
      { id: 'v3', name: '第三版', description: '精简版本，重点突出', path: '/courses/business/v3' },
    ]
  },
  {
    id: 'investment',
    title: '投资的本质',
    subtitle: '从第一性原理到投资体系',
    description: '面向有实战经验但缺乏系统学习的投资者，从第一性原理出发，构建完整的投资知识体系和决策框架。',
    duration: '14天',
    icon: 'BarChart3',
    color: 'amber',
    path: '/courses/investment',
  },
  {
    id: 'ml',
    title: '机器学习与深度学习',
    subtitle: '从数学原理到实践应用',
    description: '从第一性原理出发，建立对ML/DL本质的理论理解，掌握核心概念与数学原理，从经典机器学习到现代深度学习。',
    duration: '14天',
    icon: 'Brain',
    color: 'violet',
    path: '/courses/ml',
  },
  {
    id: 'rl',
    title: '强化学习',
    subtitle: '从基础理论到前沿应用',
    description: '系统学习强化学习的核心概念、经典算法与前沿技术，从马尔可夫决策过程到深度强化学习、RLHF与大模型对齐。',
    duration: '10天',
    icon: 'Zap',
    color: 'orange',
    path: '/courses/rl',
  },
];

// 管理学课程详细数据
export const managementCourse: CourseInfo = {
  id: 'management',
  title: '管理学基础',
  subtitle: '从第一性原理到实践技能',
  description: '为小团队管理者提供系统性的管理知识和实践技能培训课程',
  duration: '10-14天',
  icon: 'Users',
  color: 'primary',
  hasVersions: true,
  versions: [
    { id: 'v2', name: '第二版 (经典)', description: '经典版本，内容简洁实用', path: '/courses/management/v2', default: true },
    { id: 'v3', name: '第三版 (进阶)', description: '进阶版本，内容更加丰富深入', path: '/courses/management/v3' },
    { id: 'v4', name: '第四版 (最新)', description: '最新版本，精简核心内容', path: '/courses/management/v4' },
  ],
  modules: [
    {
      id: 'module-1',
      title: '管理者的角色认知',
      duration: '第1天',
      topics: ['管理定义', '三项核心职能', '角色转变', '时间管理']
    },
    {
      id: 'module-2',
      title: '目标管理与计划制定',
      duration: '第2天',
      topics: ['SMART原则', 'OKR', 'PDCA循环', '计划制定']
    },
    {
      id: 'module-3',
      title: '人员管理与激励',
      duration: '第3-4天',
      topics: ['激励理论', '人才盘点', '绩效管理', '非物质激励']
    },
    {
      id: 'module-4',
      title: '团队建设与协作',
      duration: '第5天',
      topics: ['团队发展阶段', '团队角色', '信任建设', '心理安全感']
    },
    {
      id: 'module-5',
      title: '沟通与影响力',
      duration: '第6天',
      topics: ['沟通模型', '倾听技巧', '向上管理', '非暴力沟通']
    },
    {
      id: 'module-6',
      title: '领导力与情境管理',
      duration: '第7天',
      topics: ['情境领导', '情绪智力', '教练式领导']
    },
    {
      id: 'module-7',
      title: '决策与问题解决',
      duration: '第8天',
      topics: ['决策陷阱', '根因分析', '数据驱动']
    },
    {
      id: 'module-8',
      title: '组织设计与流程管理',
      duration: '第9天',
      topics: ['组织结构', '权责对等', '流程优化']
    },
    {
      id: 'module-9',
      title: '组织文化与价值观',
      duration: '第10天',
      topics: ['文化模型', '价值观建设', '文化塑造']
    },
    {
      id: 'module-10',
      title: '自我管理与持续成长',
      duration: '第11-12天',
      topics: ['自我觉察', '情绪管理', '学习系统']
    },
    {
      id: 'module-11',
      title: '综合实践：管理案例分析',
      duration: '第13-14天',
      topics: ['案例研讨', '综合应用']
    }
  ]
};

// 商业课程详细数据
export const businessCourse: CourseInfo = {
  id: 'business',
  title: '商业本质与思维',
  subtitle: '从第一性原理到实战应用',
  description: '从商业世界的底层规律出发，逐层构建认知框架',
  duration: '14天',
  icon: 'TrendingUp',
  color: 'emerald',
  hasVersions: true,
  versions: [
    { id: 'v1', name: '第一版', description: '完整版本，内容全面', path: '/courses/business/v1', default: true },
    { id: 'v2', name: '第二版', description: '14天系统课程', path: '/courses/business/v2' },
    { id: 'v3', name: '第三版', description: '精简版本，重点突出', path: '/courses/business/v3' },
  ],
  modules: [
    {
      id: 'day-1',
      title: '商业的第一性原理',
      duration: 'Day 1',
      topics: ['价值创造', '价值传递', '价值捕获', 'WTP-WTS框架']
    },
    {
      id: 'day-2',
      title: '竞争优势的本质-护城河理论',
      duration: 'Day 2',
      topics: ['7种竞争优势', '规模经济', '网络效应', '转换成本']
    },
    {
      id: 'day-3',
      title: '网络效应与平台战略',
      duration: 'Day 3',
      topics: ['直接网络效应', '间接网络效应', '平台商业模式', '冷启动']
    },
    {
      id: 'day-4',
      title: '商业模式的九大要素',
      duration: 'Day 4',
      topics: ['商业模式画布', '价值主张', '收入来源', '成本结构']
    },
    {
      id: 'day-5',
      title: '颠覆式创新理论',
      duration: 'Day 5',
      topics: ['持续创新vs颠覆式创新', '创新者困境', '边缘市场']
    },
    {
      id: 'day-6',
      title: '竞争优势理论-波特五力',
      duration: 'Day 6',
      topics: ['五力模型', '三大通用战略', '行业分析']
    },
    {
      id: 'day-7',
      title: '价值链与规模效应',
      duration: 'Day 7',
      topics: ['价值链理论', '规模经济', '飞轮效应']
    },
    {
      id: 'day-8',
      title: '商业分析的思维框架',
      duration: 'Day 8',
      topics: ['SWOT分析', 'PESTEL分析', 'VRIO框架']
    },
    {
      id: 'day-9',
      title: '第一性原理思维',
      duration: 'Day 9',
      topics: ['拆解问题', '从基本原理构建', '验证逻辑']
    },
    {
      id: 'day-10',
      title: '收益递增与路径依赖',
      duration: 'Day 10',
      topics: ['收益递增理论', '路径依赖', '锁定效应']
    },
    {
      id: 'day-11',
      title: '商业模式创新',
      duration: 'Day 11',
      topics: ['重新定义价值', '改变收入模式', '平台化转型']
    },
    {
      id: 'day-12',
      title: '案例分析-综合应用',
      duration: 'Day 12',
      topics: ['瑞幸咖啡', '拼多多', 'SHEIN']
    },
    {
      id: 'day-13',
      title: '商业决策的思维陷阱',
      duration: 'Day 13',
      topics: ['认知偏差', '幸存者偏差', '确认偏差', '沉没成本']
    },
    {
      id: 'day-14',
      title: '综合复盘与实践规划',
      duration: 'Day 14',
      topics: ['知识体系复盘', '实践规划', '学习路线']
    }
  ]
};

// 投资课程详细数据
export const investmentCourse: CourseInfo = {
  id: 'investment',
  title: '投资的本质',
  subtitle: '从第一性原理到投资体系',
  description: '面向有实战经验但缺乏系统学习的投资者，构建完整的投资知识体系',
  duration: '14天',
  icon: 'BarChart3',
  color: 'amber',
  modules: [
    {
      id: 'day-1',
      title: '投资的第一性原理',
      duration: 'Day 1',
      topics: ['投资vs投机', '时间价值', '复利', '风险收益权衡']
    },
    {
      id: 'day-2',
      title: '资产定价的底层逻辑',
      duration: 'Day 2',
      topics: ['价格与价值', '有效市场假说', '行为金融学']
    },
    {
      id: 'day-3',
      title: '市场的本质',
      duration: 'Day 3',
      topics: ['市场机制', '参与者画像', '市场周期']
    },
    {
      id: 'day-4',
      title: '资产类别深度解析',
      duration: 'Day 4',
      topics: ['股票', '债券', '商品', '另类资产']
    },
    {
      id: 'day-5',
      title: '估值方法论（上）DCF',
      duration: 'Day 5',
      topics: ['现金流折现', '自由现金流', '终值估计']
    },
    {
      id: 'day-6',
      title: '估值方法论（下）相对估值',
      duration: 'Day 6',
      topics: ['PE/PB/PS', 'EV/EBITDA', '估值倍数']
    },
    {
      id: 'day-7',
      title: '财务报表分析',
      duration: 'Day 7',
      topics: ['三表分析', '财务比率', '识别造假']
    },
    {
      id: 'day-8',
      title: '宏观与行业分析',
      duration: 'Day 8',
      topics: ['宏观经济', '行业周期', '竞争格局']
    },
    {
      id: 'day-9',
      title: '投资哲学与目标设定',
      duration: 'Day 9',
      topics: ['投资哲学', '目标设定', '风险承受', '资产配置']
    },
    {
      id: 'day-10',
      title: '投资策略选择',
      duration: 'Day 10',
      topics: ['主动vs被动', '因子投资', '定投策略']
    },
    {
      id: 'day-11',
      title: '投资决策流程',
      duration: 'Day 11',
      topics: ['机会筛选', '深度研究', '买入卖出决策']
    },
    {
      id: 'day-12',
      title: '风险管理体系',
      duration: 'Day 12',
      topics: ['风险类型', '分散化原理', '仓位管理']
    },
    {
      id: 'day-13',
      title: '行为金融学与心理建设',
      duration: 'Day 13',
      topics: ['认知偏差', '心理弱点', '投资日记']
    },
    {
      id: 'day-14',
      title: '综合应用与学习路径',
      duration: 'Day 14',
      topics: ['构建投资组合', '持续学习', '进阶方向']
    }
  ]
};

// ML课程详细数据
export const mlCourse: CourseInfo = {
  id: 'ml',
  title: '机器学习与深度学习',
  subtitle: '从数学原理到实践应用',
  description: '从第一性原理出发，建立对ML/DL本质的理论理解',
  duration: '14天',
  icon: 'Brain',
  color: 'violet',
  modules: [
    {
      id: 'day-1',
      title: '学习的本质',
      duration: 'Day 1',
      topics: ['学习问题形式化', '偏差-方差分解', '过拟合几何直觉']
    },
    {
      id: 'day-2',
      title: '线性模型与优化理论',
      duration: 'Day 2',
      topics: ['损失函数', '梯度下降', '凸优化']
    },
    {
      id: 'day-3',
      title: '分类与概率建模',
      duration: 'Day 3',
      topics: ['Sigmoid函数', '交叉熵', 'Softmax']
    },
    {
      id: 'day-4',
      title: '模型评估与统计推断',
      duration: 'Day 4',
      topics: ['训练验证测试', '评估指标', '显著性检验']
    },
    {
      id: 'day-5',
      title: '正则化理论',
      duration: 'Day 5',
      topics: ['L1/L2正则', '贝叶斯视角', 'Dropout']
    },
    {
      id: 'day-6',
      title: '决策树与集成理论',
      duration: 'Day 6',
      topics: ['信息增益', 'Bagging', 'Boosting', '随机森林']
    },
    {
      id: 'day-7',
      title: 'Week 1 综合与实战',
      duration: 'Day 7',
      topics: ['ML理论回顾', '端到端流程', '经典案例']
    },
    {
      id: 'day-8',
      title: '神经网络数学基础',
      duration: 'Day 8',
      topics: ['感知机', '万能近似定理', '反向传播']
    },
    {
      id: 'day-9',
      title: '优化与训练动力学',
      duration: 'Day 9',
      topics: ['损失地形', '优化算法', '梯度问题']
    },
    {
      id: 'day-10',
      title: '卷积神经网络（CNN）',
      duration: 'Day 10',
      topics: ['卷积操作', '感受野', '经典架构']
    },
    {
      id: 'day-11',
      title: '序列建模与循环网络',
      duration: 'Day 11',
      topics: ['RNN', 'LSTM/GRU', '长期依赖']
    },
    {
      id: 'day-12',
      title: 'Attention机制',
      duration: 'Day 12',
      topics: ['注意力权重', 'Self-Attention', '多头注意力']
    },
    {
      id: 'day-13',
      title: 'Transformer架构',
      duration: 'Day 13',
      topics: ['Encoder-Decoder', '位置编码', '残差连接']
    },
    {
      id: 'day-14',
      title: '现代深度学习范式',
      duration: 'Day 14',
      topics: ['预训练', '语言模型', '大模型时代']
    }
  ]
};

// RL课程详细数据
export const rlCourse: CourseInfo = {
  id: 'rl',
  title: '强化学习',
  subtitle: '从基础理论到前沿应用',
  description: '系统学习强化学习的核心概念、经典算法与前沿技术',
  duration: '10天',
  icon: 'Zap',
  color: 'orange',
  modules: [
    {
      id: 'day-1',
      title: '强化学习基础',
      duration: 'Day 1',
      topics: ['强化学习定义', 'MDP框架', '智能体-环境交互', '奖励设计']
    },
    {
      id: 'day-2',
      title: '价值函数与贝尔曼方程',
      duration: 'Day 2',
      topics: ['状态价值函数', '动作价值函数', '贝尔曼期望方程', '贝尔曼最优方程']
    },
    {
      id: 'day-3',
      title: '探索与利用',
      duration: 'Day 3',
      topics: ['多臂老虎机', 'ε-贪心策略', 'UCB算法', 'Thompson采样']
    },
    {
      id: 'day-4',
      title: '动态规划',
      duration: 'Day 4',
      topics: ['策略评估', '策略改进', '值迭代', '策略迭代']
    },
    {
      id: 'day-5',
      title: '无模型学习',
      duration: 'Day 5',
      topics: ['蒙特卡洛方法', '时序差分学习', 'Q-Learning', 'SARSA']
    },
    {
      id: 'day-6',
      title: '策略梯度方法',
      duration: 'Day 6',
      topics: ['策略梯度定理', 'REINFORCE', 'Actor-Critic', 'A3C']
    },
    {
      id: 'day-7',
      title: '深度Q网络',
      duration: 'Day 7',
      topics: ['DQN架构', '经验回放', '目标网络', 'Double DQN']
    },
    {
      id: 'day-8',
      title: 'PPO与策略优化',
      duration: 'Day 8',
      topics: ['TRPO', 'PPO算法', '重要性采样', 'GAE']
    },
    {
      id: 'day-9',
      title: 'RLHF与LLM对齐',
      duration: 'Day 9',
      topics: ['RLHF框架', '奖励建模', 'PPO应用', 'DPO']
    },
    {
      id: 'day-10',
      title: '多智能体与前沿',
      duration: 'Day 10',
      topics: ['多智能体RL', '博弈论基础', '前沿方向', '实践总结']
    }
  ]
};

// 根据课程ID获取课程信息
export function getCourseById(id: string): CourseInfo | undefined {
  const courses: Record<string, CourseInfo> = {
    ml: mlCourse,
    rl: rlCourse,
  };
  return courses[id];
}

// 获取课程模块路径
export function getModulePath(courseId: string, moduleId: string): string {
  return `/courses/${courseId}/module/${moduleId}`;
}

// 获取默认版本
export function getDefaultVersion(courseId: string): CourseVersion | undefined {
  const course = getCourseById(courseId);
  if (!course?.versions) return undefined;
  return course.versions.find(v => v.default) || course.versions[0];
}
