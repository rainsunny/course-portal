import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import Layout from '@/components/Layout';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import ModuleHeader from '@/components/ModuleHeader';
import { getCourseById, coursesMeta } from '@/lib/courseData';
import { getModuleContent, getAllVersionedModulePaths, courseContentPaths } from '@/lib/contentFiles';

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

// Generate static params for versioned module routes
export async function generateStaticParams() {
  return getAllVersionedModulePaths();
}

export default function VersionedModulePage({ params }: { params: { courseId: string; version: string; moduleId: string } }) {
  const { courseId, version, moduleId } = params;
  const course = getCourseById(courseId);

  if (!course) {
    notFound();
  }

  // Verify this version exists for this course
  const courseConfig = courseContentPaths[courseId];
  if (!courseConfig || !courseConfig.versions[version]) {
    notFound();
  }

  const currentIndex = course.modules.findIndex(m => m.id === moduleId);
  const currentModule = course.modules[currentIndex];
  const prevModule = currentIndex > 0 ? course.modules[currentIndex - 1] : null;
  const nextModule = currentIndex < course.modules.length - 1 ? course.modules[currentIndex + 1] : null;

  const content = getModuleContent(courseId, moduleId, version);
  const colors = colorMap[course.color] || colorMap.primary;
  const versionName = courseConfig.versions[version]?.name || version;

  if (!content || !currentModule) {
    return (
      <Layout>
        <div className="bg-gray-50 min-h-screen">
          {/* Header */}
          <div className={`bg-gradient-to-br ${colors.gradient} text-white`}>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3 sm:space-x-4">
                  <Link
                    href={`/courses/${courseId}/${version}`}
                    className="text-white/80 hover:text-white transition-colors"
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </Link>
                  <div className="flex items-center space-x-3 sm:space-x-4">
                    <div className="flex items-center justify-center w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-white/20 text-white font-bold text-sm sm:text-base">
                      {currentIndex + 1}
                    </div>
                    <div>
                      <h1 className="text-lg sm:text-xl font-bold">{currentModule?.title || '模块'}</h1>
                      <p className="text-xs sm:text-sm text-white/80">{currentModule?.duration || ''} · {versionName}</p>
                    </div>
                  </div>
                </div>

                {/* 桌面端导航按钮 */}
                <div className="hidden sm:flex items-center space-x-2">
                  {prevModule && (
                    <Link
                      href={`/courses/${courseId}/${version}/module/${prevModule.id}`}
                      className="flex items-center px-4 py-2 text-sm font-medium text-white/80 hover:text-white border border-white/30 rounded-lg hover:border-white/50 transition-colors"
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" />
                      上一模块
                    </Link>
                  )}
                  {nextModule && (
                    <Link
                      href={`/courses/${courseId}/${version}/module/${nextModule.id}`}
                      className="flex items-center px-4 py-2 text-sm font-medium bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
                    >
                      下一模块
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Link>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Content Placeholder */}
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <div className="text-center py-12">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">内容正在准备中</h2>
                <p className="text-gray-600 mb-8">该模块的内容尚未完成，请稍后再来查看。</p>
                <Link
                  href={`/courses/${courseId}`}
                  className="inline-flex items-center px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                >
                  返回课程首页
                </Link>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="bg-gray-50 min-h-screen">
        {/* Header - 客户端组件，支持滚动隐藏 */}
        <ModuleHeader
          courseId={courseId}
          moduleId={moduleId}
          version={version}
          currentIndex={currentIndex}
          title={currentModule.title}
          duration={currentModule.duration}
          versionName={versionName}
          colorGradient={colors.gradient}
          prevModule={prevModule ? { id: prevModule.id, title: prevModule.title } : null}
          nextModule={nextModule ? { id: nextModule.id, title: nextModule.title } : null}
        />

        {/* Content */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="p-4 sm:p-8">
              <MarkdownRenderer content={content} />
            </div>
          </div>

          {/* Bottom Navigation */}
          <div className="mt-6 sm:mt-8 flex items-center justify-between text-sm sm:text-base">
            {prevModule ? (
              <Link
                href={`/courses/${courseId}/${version}/module/${prevModule.id}`}
                className="flex items-center text-gray-700 hover:text-primary-600 transition-colors"
              >
                <ChevronLeft className="h-4 w-4 sm:h-5 sm:w-5 mr-1" />
                <span className="font-medium">上一模块：{prevModule.title}</span>
              </Link>
            ) : (
              <div />
            )}

            {nextModule && (
              <Link
                href={`/courses/${courseId}/${version}/module/${nextModule.id}`}
                className={`flex items-center transition-colors ${colors.text} hover:opacity-80`}
              >
                <span className="font-medium">下一模块：{nextModule.title}</span>
                <ChevronRight className="h-4 w-4 sm:h-5 sm:w-5 ml-1" />
              </Link>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
