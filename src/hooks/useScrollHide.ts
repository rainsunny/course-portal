'use client';

import { useState, useEffect, useRef } from 'react';

interface UseScrollHideOptions {
  /** 触发隐藏的滚动距离阈值，默认 50 */
  threshold?: number;
  /** 是否启用（可用于条件控制），默认 true */
  enabled?: boolean;
}

interface ScrollHideResult {
  /** 是否隐藏（导航栏和子标题栏同步） */
  isHidden: boolean;
  /** 是否为移动设备场景 */
  isMobile: boolean;
}

/**
 * 滚动隐藏 Hook
 * 向下滚动时隐藏，向上滚动时显示
 *
 * 判断是否为移动设备场景：
 * 1. 触摸设备（pointer: coarse）
 * 2. 或屏幕宽度小于 768px
 * 3. 或屏幕高度小于 800px（覆盖所有手机横屏情况）
 */
export function useScrollHide(options: UseScrollHideOptions = {}): ScrollHideResult {
  const {
    threshold = 50,
    enabled = true
  } = options;

  const [isHidden, setIsHidden] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const lastScrollY = useRef(0);
  const ticking = useRef(false);

  // 检测是否为移动设备场景
  useEffect(() => {
    const checkMobile = () => {
      // 触摸设备
      const isTouchDevice = window.matchMedia('(pointer: coarse)').matches;
      // 窄屏幕
      const isNarrowWidth = window.innerWidth < 768;
      // 低高度（手机横屏）- 覆盖所有可能的手机尺寸
      const isShortHeight = window.innerHeight < 800;

      // 检测是否为移动设备或小屏幕设备
      const mobile = isTouchDevice || isNarrowWidth || isShortHeight;

      setIsMobile(mobile);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    if (!enabled || !isMobile) {
      setIsHidden(false);
      return;
    }

    const handleScroll = () => {
      if (!ticking.current) {
        requestAnimationFrame(() => {
          const currentScrollY = window.scrollY;

          if (currentScrollY < threshold) {
            setIsHidden(false);
          } else if (currentScrollY > lastScrollY.current) {
            setIsHidden(true);
          } else if (currentScrollY < lastScrollY.current - 5) {
            setIsHidden(false);
          }

          lastScrollY.current = currentScrollY;
          ticking.current = false;
        });
        ticking.current = true;
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [threshold, enabled, isMobile]);

  return { isHidden, isMobile };
}
