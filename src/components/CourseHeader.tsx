'use client';

import Link from 'next/link';
import { ChevronLeft, Clock, BookOpen, Users, TrendingUp, BarChart3, Brain, Check } from 'lucide-react';

interface CourseHeaderProps {
  courseId: string;
  title: string;
  subtitle: string;
  description: string;
  duration: string;
  icon: string;
  colorGradient: string;
  versionName?: string;
  hasVersions?: boolean;
  versions?: Array<{
    id: string;
    name: string;
    path: string;
    default?: boolean;
  }>;
}

const iconMap: Record<string, any> = {
  Users,
  TrendingUp,
  BarChart3,
  Brain,
  BookOpen,
};

/**
 * 课程大纲页面头部 - 普通页面内容，不sticky，滚动时自然消失
 */
export default function CourseHeader({
  courseId,
  title,
  subtitle,
  description,
  duration,
  icon,
  colorGradient,
  versionName,
  hasVersions,
  versions,
}: CourseHeaderProps) {
  const IconComponent = iconMap[icon] || BookOpen;

  return (
    <div className={`bg-gradient-to-br ${colorGradient} text-white`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-16">
        <Link
          href="/courses"
          className="inline-flex items-center text-white/80 hover:text-white mb-4 sm:mb-6 transition-colors"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          返回课程列表
        </Link>

        <div className="flex items-start gap-4 sm:gap-6">
          <div className="flex-shrink-0 w-12 h-12 sm:w-16 sm:h-16 rounded-xl bg-white/20 flex items-center justify-center">
            <IconComponent className="h-6 w-6 sm:h-8 sm:w-8 text-white" />
          </div>

          <div className="flex-1 min-w-0">
            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold mb-1 sm:mb-2">{title}</h1>
            <p className="text-base sm:text-xl text-white/90 mb-2 sm:mb-4">{subtitle}</p>
            <p className="text-sm sm:text-base text-white/80 max-w-2xl hidden sm:block">{description}</p>

            <div className="flex items-center text-white/80 text-xs sm:text-sm mt-2 sm:mt-4">
              <Clock className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />
              <span>建议学习周期：{duration}</span>
              {versionName && (
                <>
                  <span className="mx-2">·</span>
                  <span className="font-medium">{versionName}</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* 版本选择 */}
        {hasVersions && versions && versions.length > 0 && (
          <div className="mt-4 sm:mt-6 pt-4 sm:pt-6 border-t border-white/20">
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
              <span className="text-xs sm:text-sm font-medium text-white/90">选择版本：</span>
              <div className="flex flex-wrap gap-2">
                {versions.map((version) => (
                  <Link
                    key={version.id}
                    href={version.path}
                    className={`px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors flex items-center gap-1 ${
                      version.default
                        ? 'bg-white text-gray-900'
                        : 'bg-white/20 text-white hover:bg-white/30'
                    }`}
                  >
                    {version.name}
                    {version.default && (
                      <Check className="h-3 w-3 sm:h-4 sm:w-4" />
                    )}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
