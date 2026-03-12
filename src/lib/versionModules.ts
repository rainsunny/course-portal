import { ModuleInfo } from './types';

// 每个版本的模块信息
// key 格式: courseId_versionId
export const versionModules: Record<string, ModuleInfo[]> = {
  // 管理学 - 第二版 (默认)
  'management_v2': [
    { id: 'module-1', title: '管理者的角色认知', duration: '第1天', topics: ['管理定义', '三项核心职能', '角色转变', '时间管理'] },
    { id: 'module-2', title: '目标管理与计划制定', duration: '第2天', topics: ['SMART原则', 'OKR', 'PDCA循环', '计划制定'] },
    { id: 'module-3', title: '人员管理与激励', duration: '第3-4天', topics: ['激励理论', '人才盘点', '绩效管理', '非物质激励'] },
    { id: 'module-4', title: '团队建设与协作', duration: '第5天', topics: ['团队发展阶段', '团队角色', '信任建设', '心理安全感'] },
    { id: 'module-5', title: '沟通与影响力', duration: '第6天', topics: ['沟通模型', '倾听技巧', '向上管理', '非暴力沟通'] },
    { id: 'module-6', title: '领导力与情境管理', duration: '第7天', topics: ['情境领导', '情绪智力', '教练式领导'] },
    { id: 'module-7', title: '决策与问题解决', duration: '第8天', topics: ['决策陷阱', '根因分析', '数据驱动'] },
    { id: 'module-8', title: '组织设计与流程管理', duration: '第9天', topics: ['组织结构', '权责对等', '流程优化'] },
    { id: 'module-9', title: '组织文化与价值观', duration: '第10天', topics: ['文化模型', '价值观建设', '文化塑造'] },
    { id: 'module-10', title: '自我管理与持续成长', duration: '第11-12天', topics: ['自我觉察', '情绪管理', '学习系统'] },
    { id: 'module-11', title: '综合实践：管理案例分析', duration: '第13-14天', topics: ['案例研讨', '综合应用'] },
  ],
  
  // 管理学 - 第三版
  'management_v3': [
    { id: 'module-1', title: '管理者的角色认知', duration: '第1天', topics: ['管理定义', '三项核心职能', '角色转变', '时间管理'] },
    { id: 'module-2', title: '目标管理与计划制定', duration: '第2天', topics: ['SMART原则', 'OKR', 'PDCA循环', '计划制定'] },
    { id: 'module-3', title: '人员管理与激励', duration: '第3-4天', topics: ['激励理论', '人才盘点', '绩效管理', '非物质激励'] },
    { id: 'module-4', title: '团队建设与协作', duration: '第5天', topics: ['团队发展阶段', '团队角色', '信任建设', '心理安全感'] },
    { id: 'module-5', title: '沟通与影响力', duration: '第6天', topics: ['沟通模型', '倾听技巧', '向上管理', '非暴力沟通'] },
    { id: 'module-6', title: '领导力与情境管理', duration: '第7天', topics: ['情境领导', '情绪智力', '教练式领导'] },
    { id: 'module-7', title: '决策与问题解决', duration: '第8天', topics: ['决策陷阱', '根因分析', '数据驱动'] },
    { id: 'module-8', title: '组织设计与流程管理', duration: '第9天', topics: ['组织结构', '权责对等', '流程优化'] },
    { id: 'module-9', title: '组织文化与价值观', duration: '第10天', topics: ['文化模型', '价值观建设', '文化塑造'] },
    { id: 'module-10', title: '自我管理与持续成长', duration: '第11-12天', topics: ['自我觉察', '情绪管理', '学习系统'] },
    { id: 'module-11', title: '综合实践：管理案例分析', duration: '第13-14天', topics: ['案例研讨', '综合应用'] },
  ],
  
  // 管理学 - 第四版
  'management_v4': [
    { id: 'module-1', title: '管理者的角色认知', duration: '第1天', topics: ['管理定义', '三项核心职能', '角色转变', '时间管理'] },
    { id: 'module-2', title: '目标管理与计划制定', duration: '第2天', topics: ['SMART原则', 'OKR', 'PDCA循环', '计划制定'] },
    { id: 'module-3', title: '人员管理与激励', duration: '第3-4天', topics: ['激励理论', '人才盘点', '绩效管理', '非物质激励'] },
    { id: 'module-4', title: '团队建设与协作', duration: '第5天', topics: ['团队发展阶段', '团队角色', '信任建设', '心理安全感'] },
    { id: 'module-5', title: '沟通与影响力', duration: '第6天', topics: ['沟通模型', '倾听技巧', '向上管理', '非暴力沟通'] },
    { id: 'module-6', title: '领导力与情境管理', duration: '第7天', topics: ['情境领导', '情绪智力', '教练式领导'] },
    { id: 'module-7', title: '决策与问题解决', duration: '第8天', topics: ['决策陷阱', '根因分析', '数据驱动'] },
    { id: 'module-8', title: '组织设计与流程管理', duration: '第9天', topics: ['组织结构', '权责对等', '流程优化'] },
    { id: 'module-9', title: '组织文化与价值观', duration: '第10天', topics: ['文化模型', '价值观建设', '文化塑造'] },
    { id: 'module-10', title: '自我管理与持续成长', duration: '第11-12天', topics: ['自我觉察', '情绪管理', '学习系统'] },
    { id: 'module-11', title: '综合实践：管理案例分析', duration: '第13-14天', topics: ['案例研讨', '综合应用'] },
  ],
  
  // 商业 - 第一版 (默认)
  'business_v1': [
    { id: 'day-1', title: '商业本质与思维', duration: 'Day 1-3', topics: ['价值创造', '价值传递', '价值捕获'] },
    { id: 'day-4', title: '商业模式与分析', duration: 'Day 4-6', topics: ['商业模式', '竞争分析', '战略思维'] },
    { id: 'day-7', title: '商业模式与实践', duration: 'Day 7-10', topics: ['案例分析', '实践应用'] },
  ],
  
  // 商业 - 第二版
  'business_v2': [
    { id: 'day-1', title: '商业的第一性原理', duration: 'Day 1', topics: ['价值创造', '价值传递', '价值捕获', 'WTP-WTS框架'] },
    { id: 'day-2', title: '竞争优势的本质-护城河理论', duration: 'Day 2', topics: ['7种竞争优势', '规模经济', '网络效应', '转换成本'] },
    { id: 'day-3', title: '网络效应与平台战略', duration: 'Day 3', topics: ['直接网络效应', '间接网络效应', '平台商业模式', '冷启动'] },
    { id: 'day-4', title: '商业模式的九大要素', duration: 'Day 4', topics: ['商业模式画布', '价值主张', '收入来源', '成本结构'] },
    { id: 'day-5', title: '颠覆式创新理论', duration: 'Day 5', topics: ['持续创新vs颠覆式创新', '创新者困境', '边缘市场'] },
    { id: 'day-6', title: '竞争优势理论-波特五力', duration: 'Day 6', topics: ['五力模型', '三大通用战略', '行业分析'] },
    { id: 'day-7', title: '价值链与规模效应', duration: 'Day 7', topics: ['价值链理论', '规模经济', '飞轮效应'] },
    { id: 'day-8', title: '商业分析的思维框架', duration: 'Day 8', topics: ['SWOT分析', 'PESTEL分析', 'VRIO框架'] },
    { id: 'day-9', title: '第一性原理思维', duration: 'Day 9', topics: ['拆解问题', '从基本原理构建', '验证逻辑'] },
    { id: 'day-10', title: '收益递增与路径依赖', duration: 'Day 10', topics: ['收益递增理论', '路径依赖', '锁定效应'] },
    { id: 'day-11', title: '商业模式创新', duration: 'Day 11', topics: ['重新定义价值', '改变收入模式', '平台化转型'] },
    { id: 'day-12', title: '案例分析-综合应用', duration: 'Day 12', topics: ['瑞幸咖啡', '拼多多', 'SHEIN'] },
    { id: 'day-13', title: '商业决策的思维陷阱', duration: 'Day 13', topics: ['认知偏差', '幸存者偏差', '确认偏差', '沉没成本'] },
    { id: 'day-14', title: '综合复盘与实践规划', duration: 'Day 14', topics: ['知识体系复盘', '实践规划', '学习路线'] },
  ],
  
  // 商业 - 第三版
  'business_v3': [
    { id: 'day-1', title: '商业思维基础', duration: 'Day 1-2', topics: ['商业本质', '价值创造'] },
    { id: 'day-3', title: '竞争与战略', duration: 'Day 3-4', topics: ['竞争优势', '战略分析'] },
    { id: 'day-5', title: '商业模式', duration: 'Day 5-6', topics: ['商业模式画布', '创新模式'] },
    { id: 'day-7', title: '商业分析框架', duration: 'Day 7-8', topics: ['分析工具', '决策框架'] },
    { id: 'day-9', title: '综合实践', duration: 'Day 9-10', topics: ['案例分析', '实践应用'] },
  ],
};

// 获取版本特定的模块列表
export function getVersionModules(courseId: string, versionId: string): ModuleInfo[] | undefined {
  return versionModules[`${courseId}_${versionId}`];
}
