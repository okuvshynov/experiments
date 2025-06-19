const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const path = require('path');

class Database {
    constructor(dbPath = './notes.sqlite') {
        this.db = new sqlite3.Database(dbPath);
        this.init();
    }

    init() {
        const schemaPath = path.join(__dirname, 'schema.sql');
        const schema = fs.readFileSync(schemaPath, 'utf8');
        
        this.db.exec(schema, (err) => {
            if (err) {
                console.error('Error initializing database:', err);
            } else {
                console.log('Database initialized successfully');
            }
        });
    }

    // Notes operations
    async addNote(content, metadata = null) {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare('INSERT INTO notes (content, metadata) VALUES (?, ?)');
            stmt.run([content, JSON.stringify(metadata)], function(err) {
                if (err) reject(err);
                else resolve(this.lastID);
            });
            stmt.finalize();
        });
    }

    async getNotes() {
        return new Promise((resolve, reject) => {
            this.db.all('SELECT * FROM notes ORDER BY created_at DESC', (err, rows) => {
                if (err) reject(err);
                else resolve(rows.map(row => ({
                    ...row,
                    metadata: row.metadata ? JSON.parse(row.metadata) : null
                })));
            });
        });
    }

    // Projects operations
    async getCurrentProjects() {
        return new Promise((resolve, reject) => {
            this.db.all(
                'SELECT id, name, description, content FROM projects WHERE is_current = 1 ORDER BY name',
                (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                }
            );
        });
    }

    async getProjectsSummary() {
        return new Promise((resolve, reject) => {
            this.db.all(
                'SELECT id, name, description FROM projects WHERE is_current = 1 ORDER BY name',
                (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                }
            );
        });
    }

    async createProject(name, description, content) {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare('INSERT INTO projects (name, description, content) VALUES (?, ?, ?)');
            stmt.run([name, description, content], function(err) {
                if (err) reject(err);
                else resolve(this.lastID);
            });
            stmt.finalize();
        });
    }

    async updateProject(projectId, name, description, content) {
        return new Promise((resolve, reject) => {
            this.db.serialize(() => {
                // Mark current version as not current
                this.db.run('UPDATE projects SET is_current = 0 WHERE id = ?', [projectId]);
                
                // Get version number for new version
                this.db.get(
                    'SELECT MAX(version) as max_version FROM projects WHERE parent_version_id = ? OR id = ?',
                    [projectId, projectId],
                    (err, row) => {
                        if (err) {
                            reject(err);
                            return;
                        }
                        
                        const newVersion = (row.max_version || 0) + 1;
                        const stmt = this.db.prepare(
                            'INSERT INTO projects (name, description, content, version, parent_version_id) VALUES (?, ?, ?, ?, ?)'
                        );
                        stmt.run([name, description, content, newVersion, projectId], function(err) {
                            if (err) reject(err);
                            else resolve(this.lastID);
                        });
                        stmt.finalize();
                    }
                );
            });
        });
    }

    async associateNoteWithProject(noteId, projectId) {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare(
                'INSERT OR IGNORE INTO note_project_associations (note_id, project_id) VALUES (?, ?)'
            );
            stmt.run([noteId, projectId], function(err) {
                if (err) reject(err);
                else resolve(this.changes > 0);
            });
            stmt.finalize();
        });
    }

    async getDebugInfo() {
        return new Promise((resolve, reject) => {
            this.db.serialize(() => {
                const info = {
                    notes: [],
                    projects: [],
                    associations: []
                };

                // Get notes count and recent notes
                this.db.all('SELECT COUNT(*) as count FROM notes', (err, countResult) => {
                    if (err) {
                        reject(err);
                        return;
                    }
                    info.notesCount = countResult[0].count;

                    // Get recent notes
                    this.db.all('SELECT id, content, created_at FROM notes ORDER BY created_at DESC LIMIT 10', (err, notes) => {
                        if (err) {
                            reject(err);
                            return;
                        }
                        info.notes = notes;

                        // Get current projects
                        this.db.all('SELECT * FROM projects WHERE is_current = 1 ORDER BY name', (err, projects) => {
                            if (err) {
                                reject(err);
                                return;
                            }
                            info.projects = projects;

                            // Get all project versions for history
                            this.db.all('SELECT id, name, version, is_current, created_at FROM projects ORDER BY name, version', (err, projectVersions) => {
                                if (err) {
                                    reject(err);
                                    return;
                                }
                                info.projectVersions = projectVersions;

                                // Get associations
                                this.db.all(`
                                    SELECT npa.note_id, npa.project_id, n.content as note_content, p.name as project_name
                                    FROM note_project_associations npa
                                    JOIN notes n ON npa.note_id = n.id
                                    JOIN projects p ON npa.project_id = p.id
                                    ORDER BY npa.created_at DESC
                                `, (err, associations) => {
                                    if (err) {
                                        reject(err);
                                        return;
                                    }
                                    info.associations = associations;
                                    resolve(info);
                                });
                            });
                        });
                    });
                });
            });
        });
    }

    close() {
        this.db.close();
    }
}

module.exports = Database;