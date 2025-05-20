// Configuration for different language-specific node types
const LANGUAGE_CONFIG = {
  javascript: {
    class: ['class_declaration'],
    function: ['function_declaration', 'method_definition', 'arrow_function', 'function'],
    variable: ['lexical_declaration', 'variable_declaration'],
    method: ['method_definition'],
    property: ['public_field_definition'],
    parameter: ['formal_parameters']
  },
  typescript: {
    class: ['class_declaration'],
    function: ['function_declaration', 'method_definition', 'arrow_function', 'function'],
    variable: ['lexical_declaration', 'variable_declaration'],
    method: ['method_definition'],
    property: ['public_field_definition', 'property_signature'],
    parameter: ['formal_parameters'],
    interface: ['interface_declaration'],
    type: ['type_alias_declaration']
  },
  python: {
    class: ['class_definition'],
    function: ['function_definition'],
    variable: ['assignment'],
    parameter: ['parameters']
  },
  ruby: {
    class: ['class', 'module'],
    function: ['method', 'singleton_method'],
    variable: ['assignment'],
    parameter: ['method_parameters']
  },
  go: {
    function: ['function_declaration', 'method_declaration'],
    struct: ['type_declaration'],
    interface: ['type_declaration'],
    variable: ['var_declaration', 'const_declaration'],
    parameter: ['parameter_list']
  },
  java: {
    class: ['class_declaration'],
    interface: ['interface_declaration'],
    function: ['method_declaration', 'constructor_declaration'],
    variable: ['field_declaration'],
    parameter: ['formal_parameters']
  },
  c: {
    function: ['function_definition', 'declaration', 'function_declarator'],
    struct: ['struct_specifier'],
    variable: ['declaration'],
    parameter: ['parameter_list', 'parameter_declaration']
  },
  cpp: {
    class: ['class_specifier'],
    function: ['function_definition', 'declaration'],
    struct: ['struct_specifier'],
    variable: ['declaration'],
    parameter: ['parameter_list']
  }
};

// For simplicity, defaulting to JavaScript if language not found
function getLanguageConfig(language) {
  return LANGUAGE_CONFIG[language] || LANGUAGE_CONFIG.javascript;
}

export function getOutlineForNode(node, options = {}) {
  const language = options.language || 'javascript';
  const langConfig = getLanguageConfig(language);
  
  return processNode(node, langConfig, language);
}

function processNode(node, langConfig, language) {
  const result = [];
  
  if (!node || !node.children) {
    return result;
  }
  
  // Process this node's children
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i);
    const outlineNode = createOutlineNode(child, langConfig, language);
    
    if (outlineNode) {
      // For nodes that contribute to the outline, add them
      result.push(outlineNode);
    } else {
      // For other nodes, recursively process them
      const childResult = processNode(child, langConfig, language);
      result.push(...childResult);
    }
  }
  
  return result;
}

function createOutlineNode(node, langConfig, language) {
  if (!node) return null;
  
  const nodeType = node.type;
  
  // Check if this node type contributes to the outline
  let outlineType = null;
  let childrenProcessor = null;
  
  if (langConfig.class && langConfig.class.includes(nodeType)) {
    outlineType = 'class';
    childrenProcessor = processMembersAndMethods;
  } else if (langConfig.function && langConfig.function.includes(nodeType)) {
    outlineType = 'function';
    childrenProcessor = processParameters;
  } else if (langConfig.variable && langConfig.variable.includes(nodeType)) {
    outlineType = 'variable';
  } else if (langConfig.method && langConfig.method.includes(nodeType)) {
    outlineType = 'method';
    childrenProcessor = processParameters;
  } else if (langConfig.interface && langConfig.interface.includes(nodeType)) {
    outlineType = 'interface';
    childrenProcessor = processMembersAndMethods;
  } else if (langConfig.struct && langConfig.struct.includes(nodeType)) {
    outlineType = 'struct';
    childrenProcessor = processMembersAndMethods;
  } else if (langConfig.type && langConfig.type.includes(nodeType)) {
    outlineType = 'type';
  }
  
  // Special handling for C declarations that might be functions
  if (!outlineType && nodeType === 'declaration' && language === 'c') {
    const declarator = findChildByType(node, 'function_declarator');
    if (declarator) {
      outlineType = 'function';
      childrenProcessor = processParameters;
    } else {
      // Could be a variable declaration
      outlineType = 'variable';
    }
  }
  
  if (!outlineType) {
    return null;
  }
  
  // Extract name
  const name = getNodeName(node, outlineType, language);
  if (!name) {
    return null;
  }
  
  // Create outline node
  const outlineNode = {
    type: outlineType,
    name: name,
    params: outlineType === 'function' || outlineType === 'method' ? getParameters(node, langConfig, language) : null,
    start: node.startPosition,
    end: node.endPosition,
    children: []
  };
  
  // Process children if needed
  if (childrenProcessor) {
    outlineNode.children = childrenProcessor(node, langConfig, language);
  }
  
  return outlineNode;
}

function getNodeName(node, outlineType, language) {
  // Common pattern to find name in different languages
  const nameNode = findNameNode(node, language);
  if (nameNode) {
    return nameNode.text;
  }
  
  // Handle language-specific cases
  if (language === 'javascript' || language === 'typescript') {
    if (outlineType === 'variable') {
      // For variable declarations, look for the first identifier
      const declarations = findChildrenByType(node, 'variable_declarator');
      if (declarations && declarations.length > 0) {
        const identifier = findChildByType(declarations[0], 'identifier');
        return identifier ? identifier.text : null;
      }
    }
  } else if (language === 'python') {
    if (outlineType === 'variable') {
      // For Python assignments, look for the first identifier
      const identifier = findChildByType(node, 'identifier');
      return identifier ? identifier.text : null;
    }
  } else if (language === 'c') {
    // For C function declarations
    if (outlineType === 'function') {
      const declarator = findChildByType(node, 'function_declarator') || 
                        findChildByType(node, 'declaration') || 
                        node;
      const identifier = findChildByType(declarator, 'identifier');
      return identifier ? identifier.text : null;
    }
    // For C struct definitions
    if (outlineType === 'struct') {
      const tagName = findChildByType(node, 'type_identifier');
      return tagName ? tagName.text : null;
    }
    // For C variable declarations
    if (outlineType === 'variable') {
      const declarator = findChildByType(node, 'init_declarator') || 
                        findChildByType(node, 'declarator');
      if (declarator) {
        const identifier = findChildByType(declarator, 'identifier');
        return identifier ? identifier.text : null;
      }
    }
  }
  
  return null;
}

function findNameNode(node, language) {
  // Get identifier node based on language and node type
  if (language === 'javascript' || language === 'typescript') {
    // For method_definition nodes, check for property_identifier first (method name)
    if (node.type === 'method_definition') {
      const propIdentifier = findDirectChildByType(node, 'property_identifier');
      if (propIdentifier) return propIdentifier;
    }
    
    return (
      findDirectChildByType(node, 'identifier') ||
      findDirectChildByType(node, 'property_identifier')
    );
  } else if (language === 'python') {
    return findDirectChildByType(node, 'identifier');
  } else if (language === 'ruby') {
    return (
      findDirectChildByType(node, 'identifier') ||
      findDirectChildByType(node, 'constant')
    );
  } else if (language === 'go') {
    return findDirectChildByType(node, 'identifier');
  } else if (language === 'java') {
    return findDirectChildByType(node, 'identifier');
  } else if (language === 'c' || language === 'cpp') {
    return findDirectChildByType(node, 'identifier');
  }
  
  return null;
}

function getParameters(node, langConfig, language) {
  let parametersNode = null;
  
  // Find the parameters node
  if (langConfig.parameter) {
    for (const paramType of langConfig.parameter) {
      parametersNode = findChildByType(node, paramType);
      if (parametersNode) break;
    }
  }
  
  if (!parametersNode) {
    return '';
  }
  
  // Extract parameter text
  return parametersNode.text;
}

function processMembersAndMethods(node, langConfig, language) {
  const result = [];
  
  // First, recursively find methods and properties
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i);
    
    const outlineNode = createOutlineNode(child, langConfig, language);
    if (outlineNode) {
      result.push(outlineNode);
      continue;
    }
    
    // Check for class body, which might contain methods and properties
    if (child.type === 'class_body' || child.type === 'block') {
      const childResult = processNode(child, langConfig, language);
      result.push(...childResult);
    }
  }
  
  return result;
}

function processParameters(node, langConfig, language) {
  // Functions typically don't have children in the outline
  return [];
}

// Helper functions to find child nodes
function findChildByType(node, type) {
  if (!node) return null;
  
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i);
    if (child.type === type) {
      return child;
    }
    
    const found = findChildByType(child, type);
    if (found) return found;
  }
  
  return null;
}

// Helper function to find a direct child by type (no recursion)
function findDirectChildByType(node, type) {
  if (!node) return null;
  
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i);
    if (child.type === type) {
      return child;
    }
  }
  
  return null;
}

function findChildrenByType(node, type) {
  const results = [];
  if (!node) return results;
  
  for (let i = 0; i < node.childCount; i++) {
    const child = node.child(i);
    if (child.type === type) {
      results.push(child);
    }
    
    const childResults = findChildrenByType(child, type);
    results.push(...childResults);
  }
  
  return results;
}