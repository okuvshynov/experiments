# Notes App

A mobile-optimized web application for taking notes that automatically organizes them into projects using AI.

## Features

- **Add Notes**: Simple interface for adding notes
- **AI-Powered Organization**: Automatically categorizes notes into projects
- **Project Management**: View and manage automatically generated projects
- **Mobile Optimized**: Works great on phones and tablets
- **Versioning**: Keep track of project changes over time

## Architecture

### Storage Layer
- SQLite database with three main tables:
  - `notes`: Raw notes as submitted
  - `projects`: AI-generated projects with versioning
  - `note_project_associations`: Links notes to projects

### API Layer
- `POST /api/notes`: Add a new note (triggers AI processing)
- `GET /api/notes`: Get all notes
- `GET /api/projects`: Get current projects

### AI Processing Workflow
1. User submits a note
2. System gets existing project summaries (name + description only)
3. AI decides whether to add to existing project or create new one
4. AI updates/creates project content
5. Note is associated with the project

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your LLM endpoint and API key
   ```

3. Start the server:
   ```bash
   npm start
   # or for development:
   npm run dev
   ```

4. Open http://localhost:3000 in your browser

## Configuration

The app works with any OpenAI-compatible API endpoint. Configure in `.env`:

- **OpenAI**: Set `LLM_ENDPOINT=https://api.openai.com/v1/chat/completions`
- **Local LLM** (e.g., LM Studio): Set `LLM_ENDPOINT=http://localhost:1234/v1/chat/completions`
- **Other providers**: Use their OpenAI-compatible endpoint

## Mobile Usage

The UI is optimized for mobile devices with:
- Touch-friendly buttons and inputs
- Responsive design
- Sticky navigation
- Easy note entry with keyboard shortcuts (Ctrl/Cmd+Enter to submit)

## Database

SQLite database (`notes.sqlite`) is created automatically on first run. Project versioning allows you to see how projects evolve as you add notes.