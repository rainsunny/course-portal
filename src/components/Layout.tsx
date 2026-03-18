import { ReactNode } from 'react';
import Navigation from './Navigation';
import Footer from './Footer';

interface LayoutProps {
  children: ReactNode;
  showSidebar?: boolean;
}

export default function Layout({ children, showSidebar = false }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navigation />
      {/* 为固定导航栏留出空间 */}
      <main className="flex-grow pt-14 sm:pt-16">
        {children}
      </main>
      <Footer />
    </div>
  );
}
