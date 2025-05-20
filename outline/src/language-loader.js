// Cache loaded languages
const languageCache = new Map();

export async function loadLanguage(languageName) {
  // Return from cache if already loaded
  if (languageCache.has(languageName)) {
    return languageCache.get(languageName);
  }
  
  try {
    let langModule;
    
    // Import the appropriate language module
    switch (languageName) {
      case 'javascript':
        langModule = await import('tree-sitter-javascript');
        break;
      case 'typescript':
        langModule = await import('tree-sitter-typescript').then(m => m.typescript);
        break;
      case 'tsx':
        langModule = await import('tree-sitter-typescript').then(m => m.tsx);
        break;
      case 'python':
        langModule = await import('tree-sitter-python');
        break;
      case 'ruby':
        langModule = await import('tree-sitter-ruby');
        break;
      case 'go':
        langModule = await import('tree-sitter-go');
        break;
      case 'java':
        langModule = await import('tree-sitter-java');
        break;
      case 'c':
        langModule = await import('tree-sitter-c');
        break;
      case 'cpp':
        langModule = await import('tree-sitter-cpp');
        break;
      default:
        throw new Error(`Unsupported language: ${languageName}`);
    }
    
    // Cache and return the language
    const language = langModule.default;
    languageCache.set(languageName, language);
    return language;
  } catch (error) {
    throw new Error(`Failed to load language ${languageName}: ${error.message}`);
  }
}