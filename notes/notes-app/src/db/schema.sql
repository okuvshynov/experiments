-- Notes table to store raw notes as submitted
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON field for any additional data
);

-- Projects table with versioning support
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    description TEXT NOT NULL, -- Brief description for LLM to understand project scope
    content TEXT NOT NULL, -- Full content including todos, notes, etc.
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_current BOOLEAN DEFAULT 1,
    parent_version_id INTEGER,
    FOREIGN KEY (parent_version_id) REFERENCES projects(id)
);

-- Association table to track which notes contribute to which projects
CREATE TABLE IF NOT EXISTS note_project_associations (
    note_id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (note_id, project_id),
    FOREIGN KEY (note_id) REFERENCES notes(id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_projects_current ON projects(is_current);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);
CREATE INDEX IF NOT EXISTS idx_projects_version ON projects(parent_version_id);