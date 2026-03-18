'use client';

import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useScrollHide } from '@/hooks/useScrollHide';

interface ModuleHeaderProps {
  courseId: string;
  moduleId: string;
  version?: string;
  currentIndex: number;
  title: string;
  duration?: string;
  versionName?: string;
  colorGradient: string;
  prevModule?: { id: string; title: string } | null;
  nextModule?: { id: string; title: string } | null;
}

/**
 * 课程章节页面头部 - sticky，和导航栏同步隐藏/显示
 */
export default function ModuleHeader({
  courseId,
  moduleId,
  version,
  currentIndex,
  title,
  duration,
  versionName,
  colorGradient,
  prevModule,
  nextModule,
}: ModuleHeaderProps) {
  const { isHidden, isMobile } = useScrollHide({ threshold: 50 });

  // 构建模块链接
  const getModuleLink = (targetModuleId: string) => {
    if (version) {
      return `/courses/${courseId}/${version}/module/${targetModuleId}`;
    }
    return `/courses/${courseId}/module/${targetModuleId}`;
  };

  // 返回链接
  const backLink = version ? `/courses/${courseId}/${version}` : `/courses/${courseId}`;

  // 移动端：导航栏隐藏时 top-0，显示时 top-14
  // 桌面端：始终 top-16
  const topClass = isMobile
    ? (isHidden ? 'top-0' : 'top-14')
    : 'top-16';

  // 隐藏时用 -translate-y-full 完全隐藏（向上移动自身高度）
  const hideTransform = isHidden && isMobile ? '-translate-y-full' : 'translate-y-0';

  return (
    <div
      className={`bg-gradient-to-br ${colorGradient} text-white sticky z-40 transition-transform duration-300 ease-in-out ${topClass} ${hideTransform}`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 sm:py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 sm:space-x-4 min-w-0 flex-1">
            <Link
              href={backLink}
              className="text-white/80 hover:text-white transition-colors flex-shrink-0"
            >
              <ChevronLeft className="h-5 w-5" />
            </Link>
            <div className="flex items-center justify-center w-7 h-7 sm:w-10 sm:h-10 rounded-lg bg-white/20 text-white font-bold text-xs sm:text-base flex-shrink-0">
              {currentIndex + 1}
            </div>
            <div className="min-w-0">
              <h1 className="text-sm sm:text-xl font-bold truncate">{title}</h1>
              <p className="text-xs sm:text-sm text-white/80 hidden sm:block">
                {duration}
                {versionName && ` · ${versionName}`}
              </p>
            </div>
          </div>

          {/* 桌面端导航按钮 */}
          <div className="hidden sm:flex items-center space-x-2">
            {prevModule && (
              <Link
                href={getModuleLink(prevModule.id)}
                className="flex items-center px-4 py-2 text-sm font-medium text-white/80 hover:text-white border border-white/30 rounded-lg hover:border-white/50 transition-colors"
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                上一模块
              </Link>
            )}
            {nextModule && (
              <Link
                href={getModuleLink(nextModule.id)}
                className="flex items-center px-4 py-2 text-sm font-medium bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
              >
                下一模块
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            )}
          </div>

          {/* 移动端导航按钮 */}
          <div className="flex sm:hidden items-center space-x-1 flex-shrink-0">
            {prevModule && (
              <Link
                href={getModuleLink(prevModule.id)}
                className="flex items-center justify-center w-8 h-8 text-white/80 hover:text-white border border-white/30 rounded-lg"
              >
                <ChevronLeft className="h-4 w-4" />
              </Link>
            )}
            {nextModule && (
              <Link
                href={getModuleLink(nextModule.id)}
                className="flex items-center justify-center w-8 h-8 bg-white/20 hover:bg-white/30 rounded-lg"
              >
                <ChevronRight className="h-4 w-4" />
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
