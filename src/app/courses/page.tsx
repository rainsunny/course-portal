import Link from 'next/link';
import { BookOpen, Clock, Users, Target, TrendingUp, BarChart3, Brain, ChevronRight } from 'lucide-react';
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

export default function CoursesPage() {
  return (
    <Layout>
      {/* Header */}
      <div className="bg-gradient-to-br from-primary-600 via-primary-700 to-primary-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">全部课程</h1>
            <p className="text-xl text-primary-100 max-w-2xl mx-auto">
              四大领域的系统化学习路径，从第一性原理出发，构建完整的知识体系
            </p>
          </div>
        </div>
      </div>

      {/* Courses Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 gap-8">
          {coursesMeta.map((course) => {
            const IconComponent = iconMap[course.icon] || BookOpen;
            const colors = colorMap[course.color] || colorMap.primary;
            
            // 多版本课程：显示版本入口
            if (course.hasVersions && course.versions && course.versions.length > 0) {
              return (
                <div key={course.id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                  <div className="p-8 pb-4">
                    <div className="flex flex-col md:flex-row md:items-start gap-6">
                      {/* Icon */}
                      <div className={`flex-shrink-0 w-16 h-16 rounded-xl ${colors.bg} ${colors.text} flex items-center justify-center`}>
                        <IconComponent className="h-8 w-8" />
                      </div>
                      
                      {/* Content */}
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 mb-1">
                          {course.title}
                        </h2>
                        <p className="text-gray-500 mb-3">{course.subtitle}</p>
                        <p className="text-gray-600 mb-4">
                          {course.description}
                        </p>
                        <div className="flex items-center text-sm text-gray-500">
                          <Clock className="h-4 w-4 mr-1" />
                          <span>{course.duration}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Version Buttons */}
                  <div className="px-8 pb-8">
                    <p className="text-sm font-medium text-gray-700 mb-3">选择版本：</p>
                    <div className="flex flex-wrap gap-3">
                      {course.versions.map((version) => (
                        <Link
                          key={version.id}
                          href={version.path}
                          className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            version.default 
                              ? 'bg-primary-600 text-white hover:bg-primary-700' 
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                        >
                          {version.name}
                          {version.default && (
                            <span className="ml-2 text-xs opacity-80">(默认)</span>
                          )}
                        </Link>
                      ))}
                    </div>
                  </div>
                </div>
              );
            }
            
            // 单版本课程：直接跳转
            return (
              <Link key={course.id} href={`/courses/${course.id}`}>
                <div className="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all p-8 cursor-pointer border border-gray-200 group">
                  <div className="flex flex-col md:flex-row md:items-start gap-6">
                    {/* Icon */}
                    <div className={`flex-shrink-0 w-16 h-16 rounded-xl ${colors.bg} ${colors.text} flex items-center justify-center`}>
                      <IconComponent className="h-8 w-8" />
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1">
                      <div className="flex items-start justify-between">
                        <div>
                          <h2 className="text-2xl font-semibold text-gray-900 group-hover:text-primary-600 transition-colors mb-1">
                            {course.title}
                          </h2>
                          <p className="text-gray-500 mb-3">{course.subtitle}</p>
                        </div>
                        <ChevronRight className="h-6 w-6 text-gray-400 group-hover:text-primary-600 transition-colors hidden md:block" />
                      </div>
                      
                      <p className="text-gray-600 mb-4">
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
    </Layout>
  );
}
