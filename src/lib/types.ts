export interface CourseMeta {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  duration: string;
  icon: string;
  color: string;
  path: string;
  hasVersions?: boolean;
  versions?: CourseVersion[];
}

export interface CourseVersion {
  id: string;
  name: string;
  description: string;
  path: string;
  default?: boolean;
  modules?: ModuleInfo[];  // 版本特定的模块列表
}

export interface ModuleInfo {
  id: string;
  title: string;
  duration: string;
  topics: string[];
}

export interface CourseInfo {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  duration: string;
  icon: string;
  color: string;
  modules: ModuleInfo[];
  hasVersions?: boolean;
  versions?: CourseVersion[];
}

export interface CourseModule {
  id: string;
  title: string;
  duration: string;
  description: string;
  sections: CourseSection[];
}

export interface CourseSection {
  id: string;
  title: string;
  content: string;
  subsections?: CourseSubsection[];
}

export interface CourseSubsection {
  id: string;
  title: string;
  content: string;
}

export interface ModuleData {
  id: string;
  title: string;
  duration: string;
  intro: string;
  sections: Section[];
}

export interface Section {
  id: string;
  title: string;
  content: ContentItem[];
}

export interface ContentItem {
  type: 'paragraph' | 'heading' | 'list' | 'quote' | 'table' | 'diagram' | 'case' | 'keypoint' | 'example';
  data: any;
}
