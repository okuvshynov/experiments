#!/usr/bin/env node

import { program } from 'commander';
import { readFileSync } from 'fs';
import { generateOutline } from '../src/index.js';

program
  .name('outline')
  .description('Generate code outline from source files')
  .version('1.0.0')
  .argument('[file]', 'Source file to process (optional, reads from stdin if not provided)')
  .option('-l, --language <language>', 'Specify the language explicitly')
  .option('-f, --format <format>', 'Output format (json, text)', 'text')
  .action(async (file, options) => {
    try {
      let content;
      let filename;
      
      if (file) {
        content = readFileSync(file, 'utf8');
        filename = file;
      } else {
        // Read from stdin
        const chunks = [];
        process.stdin.on('data', (chunk) => {
          chunks.push(chunk);
        });
        
        await new Promise((resolve) => {
          process.stdin.on('end', resolve);
        });
        
        content = Buffer.concat(chunks).toString('utf8');
        filename = 'stdin';
      }
      
      const outline = await generateOutline(content, filename, options);
      
      if (options.format === 'json') {
        console.log(JSON.stringify(outline, null, 2));
      } else {
        printOutline(outline);
      }
    } catch (error) {
      console.error(`Error: ${error.message}`);
      process.exit(1);
    }
  });

program.parse();

function printOutline(outline) {
  function printNode(node, depth = 0) {
    const indent = '  '.repeat(depth);
    console.log(`${indent}${node.type}: ${node.name}${node.params ? ` ${node.params}` : ''}`);
    
    if (node.children && node.children.length > 0) {
      for (const child of node.children) {
        printNode(child, depth + 1);
      }
    }
  }
  
  for (const node of outline) {
    printNode(node);
  }
}