# Notes App - Project Summary

## Overview
A mobile-optimized web application for note-taking that automatically organizes notes into projects using AI. Built with Node.js, SQLite, and vanilla JavaScript.

## Architecture

### Storage Layer (SQLite)
- **notes**: Raw notes as submitted by users
- **projects**: AI-generated projects with versioning support
  - Fields: id, version, name, description, content, created_at, is_current, parent_version_id
- **note_project_associations**: Links notes to projects

### API Layer (Express.js)
- `POST /api/notes` - Add note (with optional project_id for explicit assignment)
- `GET /api/notes` - Retrieve all notes
- `GET /api/projects` - Get current projects
- `PUT /api/projects/:id` - Edit project manually
- `DELETE /api/projects/:id` - Delete project
- `GET /api/debug` - Debug endpoint showing database state

### AI Integration
- Uses OpenAI-compatible API endpoints (configurable)
- Supports thinking models with `<result></result>` tag extraction
- Three main AI operations:
  1. **Project Selection**: Decides whether note belongs to existing project or needs new one
  2. **Project Update**: Updates existing project content with new note
  3. **Project Creation**: Creates initial content for new projects

### Frontend (Vanilla JS)
- Mobile-first responsive design
- Two main views: Add Note and Projects
- Real-time status feedback
- Project management interface

## Key Features

### Note Management
- Simple text area for note input
- AI-powered automatic project categorization
- Explicit project selection (bypass AI detection)
- Keyboard shortcuts (Ctrl/Cmd+Enter to submit)

### Project Management
- **View Projects**: List all current projects with content
- **Add Note to Project**: Direct assignment to specific project
- **Edit Project**: Manual editing of name, description, and content
- **Delete Project**: Remove project and all versions
- **Project Versioning**: Track changes over time

### AI Workflow
1. User submits note
2. System gets existing project summaries (name + description only)
3. AI decides: add to existing project OR create new project
4. AI updates/creates project content incorporating the note
5. Note is associated with the project

### Mobile Optimization
- Touch-friendly buttons (44px minimum)
- Responsive layouts
- Sticky navigation
- Optimized for iPhone/Android browsers

## Configuration

### Environment Variables (.env)
```bash
PORT=3000
LLM_ENDPOINT=http://localhost:1234/v1/chat/completions
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-3.5-turbo
```

### Supported LLM Endpoints
- OpenAI API
- Local LLMs (LM Studio, etc.)
- Any OpenAI-compatible endpoint

## Technical Details

### Database Schema
- **Versioning**: Projects maintain version history
- **Associations**: Many-to-many relationship between notes and projects
- **Current Projects**: `is_current` flag for active project versions

### LLM Integration
- HTTP POST requests (no OpenAI library dependency)
- Comprehensive request/response logging for debugging
- Thinking model support with tag extraction
- Error handling and JSON parsing

### Error Handling
- Network error recovery
- JSON parsing error handling
- User-friendly error messages
- Server-side validation

## Development Notes

### Recent Improvements
- Fixed `[object Object]` display issue in project content
- Added explicit project selection to bypass AI detection
- Implemented manual project editing
- Added comprehensive LLM debugging logs
- Updated prompts for plain text content formatting

### Project Structure
```
notes-app/
├── src/
│   ├── db/
│   │   ├── schema.sql
│   │   └── database.js
│   ├── services/
│   │   └── llm.js
│   └── server.js
├── public/
│   ├── css/styles.css
│   ├── js/app.js
│   └── index.html
├── package.json
├── .env.example
├── .gitignore
└── README.md
```

### Dependencies
- express: Web server
- sqlite3: Database
- cors: Cross-origin requests
- dotenv: Environment configuration

## Usage Commands

```bash
# Install dependencies
npm install

# Start server
npm start

# Development mode
npm run dev

# Configure environment
cp .env.example .env
# Edit .env with your LLM endpoint details
```

## Access Points
- Main app: http://localhost:3000
- Debug endpoint: http://localhost:3000/api/debug

## Future Considerations
- Project search/filtering
- Export functionality
- Note editing
- Bulk operations
- Project templates
- Collaboration features