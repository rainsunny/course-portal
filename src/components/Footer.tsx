import Link from 'next/link';
import { GraduationCap, BookOpen } from 'lucide-react';
import { coursesMeta } from '@/lib/courseData';

export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <GraduationCap className="h-6 w-6 text-primary-600" />
              <span className="text-lg font-bold text-gray-900">课程学习平台</span>
            </div>
            <p className="text-sm text-gray-600">
              系统化的知识学习平台，从第一性原理出发，构建完整的认知框架。
            </p>
          </div>
          
          <div className="md:col-span-2">
            <h3 className="text-sm font-semibold text-gray-900 mb-4">课程列表</h3>
            <ul className="space-y-2 grid grid-cols-2 gap-2">
              {coursesMeta.map((course) => (
                <li key={course.id}>
                  <Link 
                    href={`/courses/${course.id}`} 
                    className="text-sm text-gray-600 hover:text-primary-600"
                  >
                    {course.title}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">平台特点</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• 第一性原理出发</li>
              <li>• 系统完整的知识体系</li>
              <li>• 实践导向的教学方法</li>
              <li>• 循序渐进的学习路径</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
          <p>© 2024 课程学习平台. 仅供学习使用.</p>
        </div>
      </div>
    </footer>
  );
}
