import sys
from tree_sitter import Language, Parser

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

parser = Parser(PY_LANGUAGE)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

query = PY_LANGUAGE.query(
    """
(function_definition
  name: (identifier) @function_name
  body: (block) @function_body)
"""
)

def run_query(source_code, function_name):
    # Parse the source code
    code = bytes(source_code, "utf8")
    tree = parser.parse(code)
    root_node = tree.root_node

    # Compile and run the query
    captures = query.captures(root_node)

    for capture_name, nodes in captures.items():
        if capture_name != "function_name":
            continue

        for node in nodes:
            if code[node.start_byte:node.end_byte].decode("utf8") == function_name:
                function_node = node.parent  # The parent node is the full function definition

                # Print out the entire function definition
                function_text = code[function_node.start_byte:function_node.end_byte].decode("utf8")
                print(function_text)
                return

def main():
    if len(sys.argv) != 3:
        print("Usage: python ts.py <file_path> function_name")
        return

    file_path = sys.argv[1]
    query_string = sys.argv[2]

    source_code = read_file(file_path)
    run_query(source_code, query_string)

if __name__ == "__main__":
    main()
