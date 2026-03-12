# 课程学习门户

一个多课程学习平台，提供统一入口访问管理学、商业思维、投资体系和机器学习四门课程。

## 功能特点

- 🎓 **统一门户首页** - 一目了然查看所有课程
- 📚 **四门完整课程** - 管理学、商业思维、投资、机器学习
- 📖 **Markdown 内容渲染** - 美观的课程内容展示
- 🎨 **一致性主题设计** - 基于 ManagementCourse 的设计风格
- 📱 **响应式设计** - 支持桌面和移动设备
- ⚡ **静态导出** - 可部署到任何静态托管服务

## 课程内容

### 1. 管理学基础
从第一性原理到实践技能，为小团队管理者提供系统性的管理知识。

### 2. 商业本质与思维
从商业世界的底层规律出发，逐层构建认知框架。

### 3. 投资的本质
面向有实战经验的投资者，构建完整的投资知识体系和决策框架。

### 4. 机器学习与深度学习
从第一性原理出发，建立对 ML/DL 本质的理论理解。

## 技术栈

- **框架**: Next.js 14 (App Router)
- **样式**: Tailwind CSS
- **Markdown**: react-markdown + remark-gfm + rehype-raw
- **图标**: Lucide React
- **语言**: TypeScript

## 项目结构

```
course-portal/
├── src/
│   ├── app/                    # Next.js App Router 页面
│   │   ├── page.tsx           # 首页
│   │   ├── layout.tsx         # 根布局
│   │   ├── courses/
│   │   │   ├── page.tsx       # 课程列表页
│   │   │   └── [courseId]/
│   │   │       ├── page.tsx   # 课程详情页
│   │   │       └── module/[moduleId]/
│   │   │           └── page.tsx  # 模块内容页
│   │   └── not-found.tsx      # 404 页面
│   ├── components/             # React 组件
│   │   ├── Layout.tsx
│   │   ├── Navigation.tsx
│   │   ├── Footer.tsx
│   │   └── MarkdownRenderer.tsx
│   ├── lib/                    # 数据和工具函数
│   │   ├── courseData.ts      # 课程数据配置
│   │   ├── contentFiles.ts    # 文件路径配置
│   │   └── types.ts           # TypeScript 类型定义
│   └── styles/
│       └── globals.css        # 全局样式
├── package.json
├── tailwind.config.ts
└── next.config.js
```

## 开始使用

### 安装依赖

```bash
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:3000

### 构建静态站点

```bash
npm run build
```

静态文件将输出到 `out/` 目录。

### 预览构建结果

```bash
npm run start
```

## 课程内容路径配置

课程内容文件路径在 `src/lib/contentFiles.ts` 中配置，指向各课程目录下的 Markdown 文件：

- **管理学**: `../ManagementCourse/讲义_第三版/`
- **商业思维**: `../BusinessCourse/business_course_v2/lectures/`
- **投资**: `../investment-course/modules/`
- **机器学习**: `../ML_course/`

## 自定义主题

主题颜色在 `tailwind.config.ts` 中定义：

- `primary`: 蓝色系（管理学）
- `emerald`: 绿色系（商业思维）
- `amber`: 橙色系（投资）
- `violet`: 紫色系（机器学习）

每门课程使用不同的主题色来区分。

## 部署

构建后的静态文件可以部署到：

- Vercel
- Netlify
- GitHub Pages
- 任何静态文件服务器

## 许可

仅供学习使用。
