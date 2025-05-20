import path from 'path';

// Map of file extensions to languages
const EXTENSION_MAP = {
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.ts': 'typescript',
  '.tsx': 'tsx',
  '.py': 'python',
  '.rb': 'ruby',
  '.go': 'go',
  '.java': 'java',
  '.c': 'c',
  '.h': 'c',
  '.cpp': 'cpp',
  '.hpp': 'cpp',
  '.cc': 'cpp',
  '.cxx': 'cpp',
};

// Simple shebang detection for scripts
const SHEBANG_MAP = {
  'node': 'javascript',
  'python': 'python',
  'ruby': 'ruby',
};

export function detectLanguage(filename, content) {
  // Try to detect by file extension first
  if (filename && filename !== 'stdin') {
    const ext = path.extname(filename).toLowerCase();
    if (EXTENSION_MAP[ext]) {
      return EXTENSION_MAP[ext];
    }
  }
  
  // Try to detect by shebang
  const firstLine = content.split('\n')[0];
  if (firstLine.startsWith('#!')) {
    for (const [key, value] of Object.entries(SHEBANG_MAP)) {
      if (firstLine.includes(key)) {
        return value;
      }
    }
  }
  
  // Try to make an educated guess based on content
  if (content.includes('function') && (content.includes('var ') || content.includes('let ') || content.includes('const '))) {
    return 'javascript';
  }
  
  if (content.includes('def ') && content.includes(':')) {
    return 'python';
  }
  
  if (content.includes('class ') && content.includes('end')) {
    return 'ruby';
  }
  
  if (content.includes('package ') && content.includes('import ') && content.includes(';')) {
    return 'java';
  }
  
  // Unable to detect
  return null;
}