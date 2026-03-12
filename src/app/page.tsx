import Link from 'next/link';
import { BookOpen, Clock, Users, Target, TrendingUp, Award, GraduationCap, BarChart3, Brain, ChevronRight } from 'lucide-react';
import Layout from '@/components/Layout';
import { coursesMeta } from '@/lib/courseData';

const iconMap: Record<string, any> = {
  Users,
  TrendingUp,
  BarChart3,
  Brain,
};

const colorMap: Record<string, { bg: string; text: string; gradient: string }> = {
  primary: { 
    bg: 'bg-primary-100', 
    text: 'text-primary-600',
    gradient: 'from-primary-600 via-primary-700 to-primary-800'
  },
  emerald: { 
    bg: 'bg-emerald-100', 
    text: 'text-emerald-600',
    gradient: 'from-emerald-600 via-emerald-700 to-emerald-800'
  },
  amber: { 
    bg: 'bg-amber-100', 
    text: 'text-amber-600',
    gradient: 'from-amber-600 via-amber-700 to-amber-800'
  },
  violet: { 
    bg: 'bg-violet-100', 
    text: 'text-violet-600',
    gradient: 'from-violet-600 via-violet-700 to-violet-800'
  },
};

export default function Home() {
  const features = [
    {
      icon: BookOpen,
      title: '系统完整',
      description: '覆盖各领域核心知识和技能，从基础到进阶',
    },
    {
      icon: Target,
      title: '第一性原理',
      description: '从本质出发，而非简单的技巧堆砌，建立深层理解',
    },
    {
      icon: Users,
      title: '实践导向',
      description: '大量案例和练习，理论与实践结合，学以致用',
    },
    {
      icon: TrendingUp,
      title: '循序渐进',
      description: '从基础到高级，逐步深入，适合不同水平学习者',
    },
  ];

  return (
    <Layout>
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              课程学习平台
            </h1>
            <p className="text-xl md:text-2xl text-primary-100 mb-8">
              从第一性原理出发，构建系统化知识体系
            </p>
            <p className="text-lg text-primary-200 mb-8 max-w-3xl mx-auto">
              涵盖管理学、商业思维、投资体系、机器学习与深度学习四大领域，
              为您提供系统、深入、实践导向的学习体验
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/courses"
                className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-lg text-primary-700 bg-white hover:bg-primary-50 transition-colors"
              >
                <BookOpen className="mr-2 h-5 w-5" />
                浏览全部课程
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">学习特色</h2>
            <p className="text-lg text-gray-600">系统化、实用化的知识学习体系</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="text-center p-6"
              >
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 text-primary-600 mb-4">
                  <feature.icon className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Courses Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">全部课程</h2>
            <p className="text-lg text-gray-600">
              四大领域的系统化学习路径
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {coursesMeta.map((course) => {
              const IconComponent = iconMap[course.icon] || BookOpen;
              const colors = colorMap[course.color] || colorMap.primary;
              
              // 多版本课程：显示多个版本入口
              if (course.hasVersions && course.versions && course.versions.length > 0) {
                return (
                  <div key={course.id} className={`h-full bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden`}>
                    <div className="p-6 pb-4">
                      <div className="flex items-start space-x-4">
                        <div className={`flex-shrink-0 w-14 h-14 rounded-xl ${colors.bg} ${colors.text} flex items-center justify-center`}>
                          <IconComponent className="h-7 w-7" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold text-gray-900 mb-1">
                            {course.title}
                          </h3>
                          <p className="text-sm text-gray-500">{course.subtitle}</p>
                        </div>
                      </div>
                      <p className="text-gray-600 text-sm mt-4 line-clamp-2">
                        {course.description}
                      </p>
                    </div>
                    <div className="px-6 pb-6">
                      <p className="text-sm font-medium text-gray-700 mb-3">选择版本：</p>
                      <div className="space-y-2">
                        {course.versions.map((version) => (
                          <Link
                            key={version.id}
                            href={version.path}
                            className={`flex items-center justify-between p-3 rounded-lg border transition-all group ${
                              version.default 
                                ? 'border-primary-500 bg-primary-50 hover:bg-primary-100' 
                                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                            }`}
                          >
                            <div>
                              <div className="flex items-center gap-2">
                                <span className={`font-medium ${version.default ? 'text-primary-700' : 'text-gray-900'}`}>
                                  {version.name}
                                </span>
                                {version.default && (
                                  <span className="text-xs px-2 py-0.5 rounded-full bg-primary-600 text-white">
                                    默认
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-gray-500 mt-0.5">{version.description}</p>
                            </div>
                            <ChevronRight className={`h-5 w-5 ${version.default ? 'text-primary-600' : 'text-gray-400'} group-hover:translate-x-1 transition-transform`} />
                          </Link>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              }
              
              // 单版本课程：直接显示进入按钮
              return (
                <Link key={course.id} href={course.path}>
                  <div className={`h-full bg-white rounded-xl shadow-sm hover:shadow-lg transition-all p-8 cursor-pointer border border-gray-200 group`}>
                    <div className="flex items-start space-x-4">
                      <div className={`flex-shrink-0 w-14 h-14 rounded-xl ${colors.bg} ${colors.text} flex items-center justify-center`}>
                        <IconComponent className="h-7 w-7" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600 transition-colors mb-1">
                          {course.title}
                        </h3>
                        <p className="text-sm text-gray-500 mb-3">{course.subtitle}</p>
                        <p className="text-gray-600 text-sm line-clamp-2 mb-4">
                          {course.description}
                        </p>
                        <div className="flex items-center text-sm text-gray-500">
                          <Clock className="h-4 w-4 mr-1" />
                          <span>{course.duration}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">准备好开始学习了吗？</h2>
          <p className="text-xl text-primary-100 mb-8">
            选择一门课程，开启您的学习之旅
          </p>
          <Link
            href="/courses"
            className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-lg text-primary-700 bg-white hover:bg-primary-50 transition-colors"
          >
            <Award className="mr-2 h-5 w-5" />
            开始学习
          </Link>
        </div>
      </section>
    </Layout>
  );
}
