import fs from 'fs';
import path from 'path';

// 课程内容路径配置 - 支持多版本
// 内容文件存放在项目的 content 目录下
export const courseContentPaths: Record<string, {
  basePath: string;
  defaultVersion: string;
  versions: Record<string, {
    name: string;
    path: string;
    moduleFiles: Record<string, string>;
  }>;
}> = {
  management: {
    basePath: 'content/management',
    defaultVersion: 'v2',
    versions: {
      v2: {
        name: '第二版 (经典)',
        path: 'v2',
        moduleFiles: {
          'module-1': '模块一_管理者的角色认知.md',
          'module-2': '模块二_目标管理与计划制定.md',
          'module-3': '模块三_人员管理与激励.md',
          'module-4': '模块四_团队建设与协作.md',
          'module-5': '模块五_沟通与影响力.md',
          'module-6': '模块六_领导力与情境管理.md',
          'module-7': '模块七_决策与问题解决.md',
          'module-8': '模块八_组织设计与流程管理.md',
          'module-9': '模块九_组织文化与价值观.md',
          'module-10': '模块十_自我管理与持续成长.md',
          'module-11': '综合实践_管理案例分析.md',
        }
      },
      v3: {
        name: '第三版 (进阶)',
        path: 'v3',
        moduleFiles: {
          'module-1': '模块一_管理者的角色认知.md',
          'module-2': '模块二_目标管理与计划制定.md',
          'module-3': '模块三_人员管理与激励.md',
          'module-4': '模块四_团队建设与协作.md',
          'module-5': '模块五_沟通与影响力.md',
          'module-6': '模块六_领导力与情境管理.md',
          'module-7': '模块七_决策与问题解决.md',
          'module-8': '模块八_组织设计与流程管理.md',
          'module-9': '模块九_组织文化与价值观.md',
          'module-10': '模块十_自我管理与持续成长.md',
          'module-11': '综合实践_管理案例分析.md',
        }
      },
      v4: {
        name: '第四版 (最新)',
        path: 'v4',
        moduleFiles: {
          'module-1': '模块一_管理者的角色认知.md',
          'module-2': '模块二_目标管理与计划制定.md',
          'module-3': '模块三_人员管理与激励.md',
          'module-4': '模块四_团队建设与协作.md',
          'module-5': '模块五_沟通与影响力.md',
          'module-6': '模块六_领导力与情境管理.md',
          'module-7': '模块七_决策与问题解决.md',
          'module-8': '模块八_组织设计与流程管理.md',
          'module-9': '模块九_组织文化与价值观.md',
          'module-10': '模块十_自我管理与持续成长.md',
          'module-11': '综合实践_管理案例分析.md',
        }
      }
    }
  },
  business: {
    basePath: 'content/business',
    defaultVersion: 'v1',
    versions: {
      v1: {
        name: '第一版',
        path: 'v1',
        moduleFiles: {
          'day-1': '商业本质与思维_两周速成课_课程内容_上.md',
          'day-2': '商业本质与思维_两周速成课_课程内容_上.md',
          'day-3': '商业本质与思维_两周速成课_课程内容_上.md',
          'day-4': '商业本质与思维_两周速成课_课程内容_中.md',
          'day-5': '商业本质与思维_两周速成课_课程内容_中.md',
          'day-6': '商业本质与思维_两周速成课_课程内容_中.md',
          'day-7': '商业本质与思维_两周速成课_课程内容_下.md',
          'day-8': '商业本质与思维_两周速成课_课程内容_下.md',
          'day-9': '商业本质与思维_两周速成课_课程内容_下.md',
          'day-10': '商业本质与思维_两周速成课_课程内容_下.md',
        }
      },
      v2: {
        name: '第二版 (推荐)',
        path: 'v2',
        moduleFiles: {
          'day-1': 'Day01_商业的第一性原理.md',
          'day-2': 'Day02_竞争优势的本质-护城河理论.md',
          'day-3': 'Day03_网络效应与平台战略.md',
          'day-4': 'Day04_商业模式的九大要素.md',
          'day-5': 'Day05_颠覆式创新理论.md',
          'day-6': 'Day06_竞争优势理论-波特五力.md',
          'day-7': 'Day07_价值链与规模效应.md',
          'day-8': 'Day08_商业分析的思维框架.md',
          'day-9': 'Day09_第一性原理思维.md',
          'day-10': 'Day10_收益递增与路径依赖.md',
          'day-11': 'Day11_商业模式创新.md',
          'day-12': 'Day12_案例分析-综合应用.md',
          'day-13': 'Day13_商业决策的思维陷阱.md',
          'day-14': 'Day14_综合复盘与实践规划.md',
        }
      },
      v3: {
        name: '第三版',
        path: 'v3',
        moduleFiles: {
          'day-1': '商业思维课程-Day1-2.md',
          'day-2': '商业思维课程-Day1-2.md',
          'day-3': '商业思维课程-Day3-4.md',
          'day-4': '商业思维课程-Day3-4.md',
          'day-5': '商业思维课程-Day5-6.md',
          'day-6': '商业思维课程-Day5-6.md',
          'day-7': '商业思维课程-Day7-8.md',
          'day-8': '商业思维课程-Day7-8.md',
          'day-9': '商业思维课程-Day9-10.md',
          'day-10': '商业思维课程-Day9-10.md',
        }
      }
    }
  },
  investment: {
    basePath: 'content/investment',
    defaultVersion: 'v1',
    versions: {
      v1: {
        name: '完整版',
        path: 'v1',
        moduleFiles: {
          'day-1': 'day-1-first-principles.md',
          'day-2': 'day-2-asset-pricing.md',
          'day-3': 'day-3-market-nature.md',
          'day-4': 'day-4-asset-classes.md',
          'day-5': 'day-5-dcf-valuation.md',
          'day-6': 'day-6-relative-valuation.md',
          'day-7': 'day-7-financial-statement.md',
          'day-8': 'day-8-macro-industry.md',
          'day-9': 'day-9-investment-philosophy.md',
          'day-10': 'day-10-investment-strategies.md',
          'day-11': 'day-11-decision-process.md',
          'day-12': 'day-12-risk-management.md',
          'day-13': 'day-13-behavioral-finance.md',
          'day-14': 'day-14-comprehensive-application.md',
        }
      }
    }
  },
  ml: {
    basePath: 'content/ml',
    defaultVersion: 'v1',
    versions: {
      v1: {
        name: '完整版',
        path: 'v1',
        moduleFiles: {
          'day-1': 'Day1_Learning_Essence.md',
          'day-2': 'Day2_Linear_Model_Optimization.md',
          'day-3': 'Day3_Classification_Probability.md',
          'day-4': 'Day4_Model_Evaluation.md',
          'day-5': 'Day5_Regularization_Theory.md',
          'day-6': 'Day6_Decision_Tree_Ensemble.md',
          'day-7': 'Day7_ML_Synthesis_Practice.md',
          'day-8': 'Day8_Neural_Network_Foundation.md',
          'day-9': 'Day9_Optimization_Training.md',
          'day-10': 'Day10_CNN.md',
          'day-11': 'Day11_RNN.md',
          'day-12': 'Day12_Attention.md',
          'day-13': 'Day13_Transformer.md',
          'day-14': 'Day14_Modern_DL.md',
        }
      }
    }
  },
  rl: {
    basePath: 'content/rl',
    defaultVersion: 'v1',
    versions: {
      v1: {
        name: '完整版',
        path: 'v1',
        moduleFiles: {
          'day-1': '01_强化学习基础.md',
          'day-2': '02_价值函数与贝尔曼方程.md',
          'day-3': '03_探索与利用.md',
          'day-4': '04_动态规划.md',
          'day-5': '05_无模型学习.md',
          'day-6': '06_策略梯度方法.md',
          'day-7': '07_深度Q网络.md',
          'day-8': '08_PPO与策略优化.md',
          'day-9': '09_RLHF与LLM对齐.md',
          'day-10': '10_多智能体与前沿.md',
        }
      }
    }
  },
  agent: {
    basePath: 'content/agent',
    defaultVersion: 'v1',
    versions: {
      v1: {
        name: '完整版',
        path: 'v1',
        moduleFiles: {
          'day-1': 'Day1_Agent_Basics.md',
          'day-2': 'Day2_Workflows.md',
          'day-3': 'Day3_Full_Agent.md',
          'day-4': 'Day4_Tools_MCP.md',
          'day-5': 'Day5_Memory_Context.md',
          'day-6': 'Day6_Multi_Agent.md',
          'day-7': 'Day7_Project_Practice.md',
        }
      }
    }
  }
};

// 获取模块内容
export function getModuleContent(courseId: string, moduleId: string, version?: string): string | null {
  try {
    const courseConfig = courseContentPaths[courseId];
    if (!courseConfig) return null;
    
    // 获取版本配置
    const versionId = version || courseConfig.defaultVersion;
    const versionConfig = courseConfig.versions[versionId];
    if (!versionConfig) return null;
    
    const filePath = versionConfig.moduleFiles[moduleId];
    if (!filePath) return null;
    
    const fullPath = path.join(process.cwd(), courseConfig.basePath, versionConfig.path, filePath);
    
    if (!fs.existsSync(fullPath)) {
      console.log(`File not found: ${fullPath}`);
      return null;
    }
    
    return fs.readFileSync(fullPath, 'utf-8');
  } catch (error) {
    console.error(`Error reading module content: ${error}`);
    return null;
  }
}

// 获取所有课程模块路径（用于静态生成）
export function getAllModulePaths(): { courseId: string; moduleId: string; version?: string }[] {
  const paths: { courseId: string; moduleId: string; version?: string }[] = [];
  
  Object.keys(courseContentPaths).forEach(courseId => {
    const courseConfig = courseContentPaths[courseId];
    Object.keys(courseConfig.versions).forEach(versionId => {
      const versionConfig = courseConfig.versions[versionId];
      Object.keys(versionConfig.moduleFiles).forEach(moduleId => {
        // 默认版本不需要版本参数
        if (versionId === courseConfig.defaultVersion) {
          paths.push({ courseId, moduleId });
        } else {
          paths.push({ courseId, moduleId, version: versionId });
        }
      });
    });
  });
  
  return paths;
}

// 获取所有版本化的模块路径（用于静态生成）
// 获取所有版本化的模块路径（用于静态生成）
// 包含所有版本，包括默认版本
export function getAllVersionedModulePaths(): { courseId: string; version: string; moduleId: string }[] {
  const paths: { courseId: string; version: string; moduleId: string }[] = [];
  
  Object.keys(courseContentPaths).forEach(courseId => {
    const courseConfig = courseContentPaths[courseId];
    Object.keys(courseConfig.versions).forEach(versionId => {
      const versionConfig = courseConfig.versions[versionId];
      Object.keys(versionConfig.moduleFiles).forEach(moduleId => {
        paths.push({ courseId, version: versionId, moduleId });
      });
    });
  });
  
  return paths;
}


// 获取课程的所有版本
export function getCourseVersions(courseId: string): { id: string; name: string; default?: boolean }[] {
  const courseConfig = courseContentPaths[courseId];
  if (!courseConfig) return [];
  
  return Object.entries(courseConfig.versions).map(([id, config]) => ({
    id,
    name: config.name,
    default: id === courseConfig.defaultVersion
  }));
}
