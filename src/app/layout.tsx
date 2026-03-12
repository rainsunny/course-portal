import type { Metadata } from 'next'
import '@/styles/globals.css'




export const metadata: Metadata = {
  title: '课程学习平台 - 系统化知识学习',
  description: '从第一性原理出发，系统学习管理、商业、投资、机器学习等核心知识',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
