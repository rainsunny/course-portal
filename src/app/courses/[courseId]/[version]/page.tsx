import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ChevronLeft, Clock, BookOpen, Users, TrendingUp, BarChart3, Brain } from 'lucide-react';
import Layout from '@/components/Layout';
import { getCourseById, coursesMeta } from '@/lib/courseData';
import { courseContentPaths } from '@/lib/contentFiles';
import { getVersionModules } from '@/lib/versionModules';

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

// Generate static params for version routes
export async function generateStaticParams() {
  const params: { courseId: string; version: string }[] = [];
  
  coursesMeta.forEach(course => {
    if (course.hasVersions && course.versions) {
      course.versions.forEach(version => {
        params.push({
          courseId: course.id,
          version: version.id,
        });
      });
    }
  });
  
  return params;
}

export default function VersionCoursePage({ params }: { params: { courseId: string; version: string } }) {
  const { courseId, version } = params;
  const course = getCourseById(courseId);
  
  if (!course) {
    notFound();
  }

  // Verify this version exists for this course
  const courseConfig = courseContentPaths[courseId];
  if (!courseConfig || !courseConfig.versions[version]) {
    notFound();
  }

  const versionInfo = courseConfig.versions[version];
  const currentVersionData = course.versions?.find(v => v.id === version);
  
  // 获取版本特定的模块列表
  const modules = getVersionModules(courseId, version) || course.modules;
  
  const IconComponent = iconMap[course.icon] || BookOpen;
  const colors = colorMap[course.color] || colorMap.primary;

  return (
    <Layout>
      {/* Header */}
      <div className={`bg-gradient-to-br ${colors.gradient} text-white`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <Link 
            href="/courses" 
            className="inline-flex items-center text-white/80 hover:text-white mb-6 transition-colors"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            返回课程列表
          </Link>
          
          <div className="flex items-start gap-6">
            <div className="flex-shrink-0 w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <IconComponent className="h-8 w-8 text-white" />
            </div>
            
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-2">{course.title}</h1>
              <p className="text-xl text-white/90 mb-4">{course.subtitle}</p>
              <p className="text-white/80 max-w-2xl">{course.description}</p>
              
              <div className="flex items-center text-white/80 text-sm mt-4">
                <Clock className="h-4 w-4 mr-1" />
                <span>建议学习周期：{course.duration}</span>
                <span className="mx-2">·</span>
                <span className="font-medium">{versionInfo.name}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Course Modules */}
      {/* Course Modules */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">课程大纲</h2>
          <p className="text-lg text-gray-600">
            共 {modules.length} 个学习模块
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((module, index) => (
            <Link 
              key={module.id} 
              href={`/courses/${courseId}/${version}/module/${module.id}`}
            >
              <div className="h-full bg-white rounded-lg shadow-sm hover:shadow-lg transition-all p-6 cursor-pointer border border-gray-200 group">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center justify-center w-12 h-12 rounded-full bg-primary-100 text-primary-700 font-bold text-lg">
                    {index + 1}
                  </div>
                  <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                    {module.duration}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3 group-hover:text-primary-600 transition-colors">
                  {module.title}
                </h3>
                <div className="space-y-2">
                  {module.topics.slice(0, 3).map((topic, topicIndex) => (
                    <div key={topicIndex} className="flex items-center text-sm text-gray-600">
                      <div className={`w-1.5 h-1.5 rounded-full ${colors.bg.replace('100', '500')} mr-2`} />
                      {topic}
                    </div>
                  ))}
                  {module.topics.length > 3 && (
                    <div className="text-sm text-gray-400">
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
      <section className={`py-16 bg-gradient-to-br ${colors.gradient}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">开始学习</h2>
          <p className="text-xl text-white/90 mb-8">
            从第一个模块开始您的学习之旅
          </p>
          <Link
            href={`/courses/${courseId}/${version}/module/${modules[0]?.id}`}
            className="inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-lg bg-white text-gray-900 hover:bg-gray-100 transition-colors"
          >
            <BookOpen className="mr-2 h-5 w-5" />
            开始学习
          </Link>
        </div>
      </section>
    </Layout>
  );
}
