require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const Database = require('./db/database');
const LLMService = require('./services/llm');

const app = express();
const port = process.env.PORT || 3000;

// Initialize services
const db = new Database();
const llm = new LLMService(
    process.env.LLM_ENDPOINT || 'http://localhost:1234/v1/chat/completions',
    process.env.LLM_API_KEY || 'dummy-key',
    process.env.LLM_MODEL || 'gpt-3.5-turbo'
);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// API Routes
app.post('/api/notes', async (req, res) => {
    try {
        const { content } = req.body;
        if (!content || content.trim() === '') {
            return res.status(400).json({ error: 'Content is required' });
        }

        // Step 1: Save the note
        const noteId = await db.addNote(content.trim());

        // Step 2: Get existing projects summary
        const existingProjects = await db.getProjectsSummary();

        // Step 3: Ask LLM to select/create project
        const projectDecision = await llm.selectProject(content, existingProjects);

        let projectId;
        
        if (projectDecision.action === 'existing') {
            // Update existing project
            const projects = await db.getCurrentProjects();
            const project = projects.find(p => p.id === projectDecision.project_id);
            
            if (project) {
                const updatedProject = await llm.updateProjectContent(
                    content, 
                    project.name, 
                    project.description, 
                    project.content
                );
                
                projectId = await db.updateProject(
                    project.id,
                    updatedProject.name,
                    updatedProject.description,
                    updatedProject.content
                );
            } else {
                throw new Error('Selected project not found');
            }
        } else if (projectDecision.action === 'new') {
            // Create new project
            const newProjectContent = await llm.createNewProject(
                content,
                projectDecision.name,
                projectDecision.description
            );
            
            projectId = await db.createProject(
                projectDecision.name,
                projectDecision.description,
                newProjectContent.content
            );
        }

        // Step 4: Associate note with project
        if (projectId) {
            await db.associateNoteWithProject(noteId, projectId);
        }

        res.json({ 
            success: true, 
            noteId, 
            projectId,
            action: projectDecision.action 
        });
    } catch (error) {
        console.error('Error processing note:', error);
        res.status(500).json({ error: 'Failed to process note' });
    }
});

app.get('/api/notes', async (req, res) => {
    try {
        const notes = await db.getNotes();
        res.json(notes);
    } catch (error) {
        console.error('Error fetching notes:', error);
        res.status(500).json({ error: 'Failed to fetch notes' });
    }
});

app.get('/api/projects', async (req, res) => {
    try {
        const projects = await db.getCurrentProjects();
        res.json(projects);
    } catch (error) {
        console.error('Error fetching projects:', error);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});

app.get('/api/debug', async (req, res) => {
    try {
        const debugInfo = await db.getDebugInfo();
        
        let output = '';
        output += '=== NOTES APP DEBUG INFO ===\n\n';
        
        // Summary
        output += `📊 SUMMARY:\n`;
        output += `- Total Notes: ${debugInfo.notesCount}\n`;
        output += `- Current Projects: ${debugInfo.projects.length}\n`;
        output += `- Note-Project Associations: ${debugInfo.associations.length}\n\n`;
        
        // Recent Notes
        output += `📝 RECENT NOTES (Last 10):\n`;
        if (debugInfo.notes.length === 0) {
            output += '  No notes found.\n';
        } else {
            debugInfo.notes.forEach(note => {
                const date = new Date(note.created_at).toLocaleString();
                const preview = note.content.length > 50 ? 
                    note.content.substring(0, 50) + '...' : 
                    note.content;
                output += `  [${note.id}] ${date}\n`;
                output += `      "${preview}"\n\n`;
            });
        }
        
        // Current Projects
        output += `🗂️  CURRENT PROJECTS:\n`;
        if (debugInfo.projects.length === 0) {
            output += '  No projects found.\n';
        } else {
            debugInfo.projects.forEach(project => {
                const date = new Date(project.created_at).toLocaleString();
                output += `  [${project.id}] ${project.name} (v${project.version})\n`;
                output += `      Created: ${date}\n`;
                output += `      Description: ${project.description}\n`;
                output += `      Content Preview: ${project.content.substring(0, 100)}${project.content.length > 100 ? '...' : ''}\n\n`;
            });
        }
        
        // Project Version History
        output += `📚 PROJECT VERSION HISTORY:\n`;
        const projectGroups = {};
        debugInfo.projectVersions.forEach(pv => {
            if (!projectGroups[pv.name]) {
                projectGroups[pv.name] = [];
            }
            projectGroups[pv.name].push(pv);
        });
        
        Object.keys(projectGroups).forEach(projectName => {
            output += `  ${projectName}:\n`;
            projectGroups[projectName].forEach(version => {
                const date = new Date(version.created_at).toLocaleString();
                const status = version.is_current ? ' [CURRENT]' : '';
                output += `    v${version.version} - ${date}${status}\n`;
            });
            output += '\n';
        });
        
        // Note-Project Associations
        output += `🔗 NOTE-PROJECT ASSOCIATIONS:\n`;
        if (debugInfo.associations.length === 0) {
            output += '  No associations found.\n';
        } else {
            debugInfo.associations.forEach(assoc => {
                const notePreview = assoc.note_content.length > 40 ? 
                    assoc.note_content.substring(0, 40) + '...' : 
                    assoc.note_content;
                output += `  Note [${assoc.note_id}] → Project [${assoc.project_id}] "${assoc.project_name}"\n`;
                output += `    Note: "${notePreview}"\n\n`;
            });
        }
        
        output += '=== END DEBUG INFO ===\n';
        
        res.setHeader('Content-Type', 'text/plain');
        res.send(output);
    } catch (error) {
        console.error('Error fetching debug info:', error);
        res.status(500).send('Error fetching debug information');
    }
});

// Serve main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Graceful shutdown
process.on('SIGINT', () => {
    db.close();
    process.exit(0);
});

app.listen(port, () => {
    console.log(`Notes app running on http://localhost:${port}`);
});

module.exports = app;