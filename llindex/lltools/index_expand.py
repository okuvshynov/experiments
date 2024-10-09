# will return as much of an expansion as we can fit into max tokens
# options are:
# list of directories and files (hierarchical)
# list of directories with summaries + list of files
# list of directories with summaries + list of files with summaries
class IndexExpandTool:
    def __init__(self, root, max_tokens):
        self.root = os.path.expanduser(root)

    def definition(self):
        return {
            "name": "git_grep",
            "description": "Executes git grep with the provided argument and returns file:line data. Use this tool to look up usage and definitions of symbols.",
            "input_schema": {
              "type": "object",
              "properties": {
                "needle": {
                  "type": "string",
                  "description": "A string to grep for."
                }
              },
              "required": ["needle"]
            }
        }

    # TODO: only return file names here, as we operate on file level anyway?
    def run(self, tool_use_args):
        command = ["git", "grep", "-n", shlex.quote(tool_use_args['needle'])]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, cwd=self.root)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"

if __name__ == '__main__':
    tool = GitGrepTool(sys.argv[1])
    print(tool.run({'needle': sys.argv[2]}))

