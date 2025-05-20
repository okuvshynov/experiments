import path from 'path';
import Parser from 'tree-sitter';
import { detectLanguage } from './language-detector.js';
import { loadLanguage } from './language-loader.js';
import { getOutlineForNode } from './outline-generator.js';

export async function generateOutline(content, filename, options = {}) {
  // Detect or use specified language
  const language = options.language || detectLanguage(filename, content);
  if (!language) {
    throw new Error('Unable to detect language and no language specified');
  }
  
  // Load Tree-sitter parser for the language
  const parser = new Parser();
  const treeLanguage = await loadLanguage(language);
  parser.setLanguage(treeLanguage);
  
  // Parse the content
  const tree = parser.parse(content);
  
  // Generate outline
  return getOutlineForNode(tree.rootNode, { language });
}