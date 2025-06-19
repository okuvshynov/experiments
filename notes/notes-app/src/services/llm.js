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

        const requestBody = {
            model: this.model,
            messages: messages,
            temperature: 0.3,
            max_tokens: 8192
        };

        const postData = JSON.stringify(requestBody);

        console.log('\n=== LLM REQUEST ===');
        console.log('Endpoint:', this.endpoint);
        console.log('Model:', this.model);
        console.log('Messages:', JSON.stringify(messages, null, 2));
        console.log('Full request body:', JSON.stringify(requestBody, null, 2));
        console.log('==================\n');

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
                    console.log('\n=== LLM RAW RESPONSE ===');
                    console.log('Status:', res.statusCode);
                    console.log('Headers:', JSON.stringify(res.headers, null, 2));
                    console.log('Raw response body:', data);
                    console.log('=======================\n');

                    try {
                        const response = JSON.parse(data);
                        if (response.choices && response.choices[0]) {
                            const content = response.choices[0].message.content;
                            console.log('\n=== LLM EXTRACTED CONTENT ===');
                            console.log('Content:', content);
                            console.log('Content type:', typeof content);
                            console.log('Content length:', content.length);
                            console.log('=============================\n');
                            resolve(content);
                        } else {
                            console.error('Invalid response format - no choices found');
                            reject(new Error('Invalid response format'));
                        }
                    } catch (err) {
                        console.error('Failed to parse LLM response as JSON:', err);
                        console.error('Raw response was:', data);
                        reject(err);
                    }
                });
            });

            req.on('error', (err) => {
                console.error('LLM request failed:', err);
                reject(err);
            });
            req.write(postData);
            req.end();
        });
    }

    extractJsonFromResult(responseText) {
        console.log('\n=== JSON EXTRACTION ===');
        console.log('Input text:', responseText);
        
        const resultMatch = responseText.match(/<result>([\s\S]*?)<\/result>/);
        if (resultMatch) {
            const extracted = resultMatch[1].trim();
            console.log('Found <result> tags, extracted:', extracted);
            console.log('======================\n');
            return extracted;
        }
        
        // Fallback: try to find JSON in the response without tags
        const fallback = responseText.trim();
        console.log('No <result> tags found, using fallback:', fallback);
        console.log('======================\n');
        return fallback;
    }

    async selectProject(noteContent, existingProjects) {
        console.log('\n=== SELECT PROJECT CALL ===');
        console.log('Note content:', noteContent);
        console.log('Existing projects:', existingProjects);
        
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
        
        console.log('\n=== SELECT PROJECT PARSING ===');
        console.log('JSON string to parse:', jsonStr);
        
        try {
            const result = JSON.parse(jsonStr);
            console.log('Parsed result:', result);
            console.log('==============================\n');
            return result;
        } catch (err) {
            console.error('Failed to parse JSON:', err);
            console.error('JSON string was:', jsonStr);
            throw err;
        }
    }

    async updateProjectContent(noteContent, projectName, projectDescription, currentContent) {
        console.log('\n=== UPDATE PROJECT CALL ===');
        console.log('Note content:', noteContent);
        console.log('Project name:', projectName);
        console.log('Project description:', projectDescription);
        console.log('Current content:', currentContent);
        
        const messages = [
            {
                role: 'system',
                content: `You are updating a project with a new note. Create an updated version of the project content that incorporates the new note. Keep the content organized and maintain any existing structure like todo lists, notes, etc.

The content field should be plain text (not JSON formatted) - organize it as you would a normal document with line breaks, bullet points, etc.

Respond with a JSON object wrapped in <result></result> tags:
<result>
{"name": "Updated Project Name", "description": "Updated brief description", "content": "Updated full content as plain text with proper formatting"}
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
        
        console.log('\n=== UPDATE PROJECT PARSING ===');
        console.log('JSON string to parse:', jsonStr);
        
        try {
            const result = JSON.parse(jsonStr);
            console.log('Parsed result:', result);
            console.log('==============================\n');
            return result;
        } catch (err) {
            console.error('Failed to parse JSON:', err);
            console.error('JSON string was:', jsonStr);
            throw err;
        }
    }

    async createNewProject(noteContent, projectName, projectDescription) {
        console.log('\n=== CREATE PROJECT CALL ===');
        console.log('Note content:', noteContent);
        console.log('Project name:', projectName);
        console.log('Project description:', projectDescription);
        
        const messages = [
            {
                role: 'system',
                content: `Create initial content for a new project based on the first note. Structure it as a project with any relevant todos, goals, or organization.

The content should be plain text (not JSON formatted) - organize it as you would a normal document with line breaks, bullet points, todo lists, etc.

Respond with a JSON object wrapped in <result></result> tags:
<result>
{"content": "Full project content as plain text with proper formatting, line breaks, and structure"}
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
        
        console.log('\n=== CREATE PROJECT PARSING ===');
        console.log('JSON string to parse:', jsonStr);
        
        try {
            const result = JSON.parse(jsonStr);
            console.log('Parsed result:', result);
            console.log('==============================\n');
            return result;
        } catch (err) {
            console.error('Failed to parse JSON:', err);
            console.error('JSON string was:', jsonStr);
            throw err;
        }
    }
}

module.exports = LLMService;
