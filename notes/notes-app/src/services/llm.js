const https = require('https');
const http = require('http');

class LLMService {
    constructor(endpoint, apiKey, model = 'gpt-3.5-turbo') {
        this.endpoint = endpoint;
        this.apiKey = apiKey;
        this.model = model;
    }

    async makeRequest(messages) {
        const url = new URL(this.endpoint);
        const isHttps = url.protocol === 'https:';
        const client = isHttps ? https : http;

        const postData = JSON.stringify({
            model: this.model,
            messages: messages,
            temperature: 0.3,
            max_tokens: 1000
        });

        const options = {
            hostname: url.hostname,
            port: url.port || (isHttps ? 443 : 80),
            path: url.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Length': Buffer.byteLength(postData)
            }
        };

        return new Promise((resolve, reject) => {
            const req = client.request(options, (res) => {
                let data = '';
                res.on('data', (chunk) => data += chunk);
                res.on('end', () => {
                    try {
                        const response = JSON.parse(data);
                        if (response.choices && response.choices[0]) {
                            resolve(response.choices[0].message.content);
                        } else {
                            reject(new Error('Invalid response format'));
                        }
                    } catch (err) {
                        reject(err);
                    }
                });
            });

            req.on('error', reject);
            req.write(postData);
            req.end();
        });
    }

    extractJsonFromResult(responseText) {
        const resultMatch = responseText.match(/<result>([\s\S]*?)<\/result>/);
        if (resultMatch) {
            return resultMatch[1].trim();
        }
        // Fallback: try to find JSON in the response without tags
        return responseText.trim();
    }

    async selectProject(noteContent, existingProjects) {
        const projectList = existingProjects.map(p => `${p.id}: ${p.name} - ${p.description}`).join('\n');
        
        const messages = [
            {
                role: 'system',
                content: `You are a project organizer. Given a new note and a list of existing projects, decide which project the note belongs to, or if a new project should be created.

Respond with a JSON object wrapped in <result></result> tags in this format:
<result>
{"action": "existing", "project_id": 123}
</result>
for existing project

OR

<result>
{"action": "new", "name": "Project Name", "description": "Brief description"}
</result>
for new project`
            },
            {
                role: 'user',
                content: `New note: "${noteContent}"

Existing projects:
${projectList || 'No existing projects'}

Which project should this note belong to?`
            }
        ];

        const response = await this.makeRequest(messages);
        const jsonStr = this.extractJsonFromResult(response);
        return JSON.parse(jsonStr);
    }

    async updateProjectContent(noteContent, projectName, projectDescription, currentContent) {
        const messages = [
            {
                role: 'system',
                content: `You are updating a project with a new note. Create an updated version of the project content that incorporates the new note. Keep the content organized and maintain any existing structure like todo lists, notes, etc.

Respond with a JSON object wrapped in <result></result> tags:
<result>
{"name": "Updated Project Name", "description": "Updated brief description", "content": "Updated full content"}
</result>`
            },
            {
                role: 'user',
                content: `Project: ${projectName}
Description: ${projectDescription}

Current content:
${currentContent}

New note to incorporate:
${noteContent}

Please update the project.`
            }
        ];

        const response = await this.makeRequest(messages);
        const jsonStr = this.extractJsonFromResult(response);
        return JSON.parse(jsonStr);
    }

    async createNewProject(noteContent, projectName, projectDescription) {
        const messages = [
            {
                role: 'system',
                content: `Create initial content for a new project based on the first note. Structure it as a project with any relevant todos, goals, or organization.

Respond with a JSON object wrapped in <result></result> tags:
<result>
{"content": "Full project content including the note and any derived structure"}
</result>`
            },
            {
                role: 'user',
                content: `Project: ${projectName}
Description: ${projectDescription}

Initial note:
${noteContent}`
            }
        ];

        const response = await this.makeRequest(messages);
        const jsonStr = this.extractJsonFromResult(response);
        return JSON.parse(jsonStr);
    }
}

module.exports = LLMService;