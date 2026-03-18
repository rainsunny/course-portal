import { notFound } from 'next/navigation';
import Link from 'next/link';
import { BookOpen, Users, TrendingUp, BarChart3, Brain } from 'lucide-react';
import Layout from '@/components/Layout';
import CourseHeader from '@/components/CourseHeader';
import { getCourseById, coursesMeta, getDefaultVersion } from '@/lib/courseData';

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

// 生成静态参数
export async function generateStaticParams() {
  return coursesMeta.map((course) => ({
    courseId: course.id,
  }));
}

export default function CoursePage({ params }: { params: { courseId: string } }) {
  const course = getCourseById(params.courseId);

  if (!course) {
    notFound();
  }

  const IconComponent = iconMap[course.icon] || BookOpen;
  const colors = colorMap[course.color] || colorMap.primary;
  const hasVersions = course.hasVersions && course.versions && course.versions.length > 0;

  return (
    <Layout>
      {/* Header - 客户端组件，支持滚动隐藏 */}
      <CourseHeader
        courseId={params.courseId}
        title={course.title}
        subtitle={course.subtitle}
        description={course.description}
        duration={course.duration}
        icon={course.icon}
        colorGradient={colors.gradient}
        hasVersions={hasVersions}
        versions={course.versions}
      />

      {/* Course Modules */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="text-center mb-8 sm:mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-2 sm:mb-4">课程大纲</h2>
          <p className="text-base sm:text-lg text-gray-600">
            共 {course.modules.length} 个学习模块
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {course.modules.map((module, index) => (
            <Link
              key={module.id}
              href={`/courses/${params.courseId}/module/${module.id}`}
            >
              <div className="h-full bg-white rounded-lg shadow-sm hover:shadow-lg transition-all p-4 sm:p-6 cursor-pointer border border-gray-200 group">
                <div className="flex items-start justify-between mb-3 sm:mb-4">
                  <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-primary-100 text-primary-700 font-bold text-base sm:text-lg">
                    {index + 1}
                  </div>
                  <span className="text-xs sm:text-sm text-gray-500 bg-gray-100 px-2 sm:px-3 py-1 rounded-full">
                    {module.duration}
                  </span>
                </div>
                <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2 sm:mb-3 group-hover:text-primary-600 transition-colors">
                  {module.title}
                </h3>
                <div className="space-y-1.5 sm:space-y-2">
                  {module.topics.slice(0, 3).map((topic, topicIndex) => (
                    <div key={topicIndex} className="flex items-center text-xs sm:text-sm text-gray-600">
                      <div className={`w-1.5 h-1.5 rounded-full ${colors.bg.replace('100', '500')} mr-2`} />
                      {topic}
                    </div>
                  ))}
                  {module.topics.length > 3 && (
                    <div className="text-xs sm:text-sm text-gray-400">
                      +{module.topics.length - 3} 更多...
                    </div>
                  )}
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Start Learning CTA */}
      <section className={`py-12 sm:py-16 bg-gradient-to-br ${colors.gradient}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3 sm:mb-4">开始学习</h2>
          <p className="text-base sm:text-xl text-white/90 mb-6 sm:mb-8">
            从第一个模块开始您的学习之旅
          </p>
          <Link
            href={`/courses/${params.courseId}/module/${course.modules[0]?.id}`}
            className="inline-flex items-center justify-center px-6 sm:px-8 py-2.5 sm:py-3 border border-transparent text-sm sm:text-base font-medium rounded-lg bg-white text-gray-900 hover:bg-gray-100 transition-colors"
          >
            <BookOpen className="mr-2 h-4 w-4 sm:h-5 sm:w-5" />
            开始学习
          </Link>
        </div>
      </section>
    </Layout>
  );
}
