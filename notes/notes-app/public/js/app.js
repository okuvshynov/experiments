class NotesApp {
    constructor() {
        this.currentSection = 'add-note';
        this.selectedProjectId = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadProjects();
    }

    bindEvents() {
        // Navigation
        document.getElementById('add-note-btn').addEventListener('click', () => {
            this.showSection('add-note');
        });

        document.getElementById('view-projects-btn').addEventListener('click', () => {
            this.showSection('view-projects');
            this.loadProjects();
        });

        // Note submission
        document.getElementById('submit-note').addEventListener('click', () => {
            this.submitNote();
        });

        // Enter key in textarea (Ctrl+Enter or Cmd+Enter)
        document.getElementById('note-input').addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                this.submitNote();
            }
        });
    }

    showSection(sectionName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`${sectionName}-btn`).classList.add('active');

        // Update sections
        document.querySelectorAll('.section').forEach(section => section.classList.remove('active'));
        
        // Map section names to correct IDs
        const sectionId = sectionName === 'view-projects' ? 'projects-section' : `${sectionName}-section`;
        document.getElementById(sectionId).classList.add('active');

        this.currentSection = sectionName;
    }

    async submitNote() {
        const noteInput = document.getElementById('note-input');
        const submitBtn = document.getElementById('submit-note');
        const status = document.getElementById('submit-status');
        
        const content = noteInput.value.trim();
        
        if (!content) {
            this.showStatus('Please enter a note', 'error');
            return;
        }

        // Disable form during submission
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        this.showStatus('Processing note with AI...', 'info');

        try {
            const requestBody = { content };
            if (this.selectedProjectId) {
                requestBody.project_id = this.selectedProjectId;
            }
            
            const response = await fetch('/api/notes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            const result = await response.json();

            if (response.ok) {
                const statusMessage = this.selectedProjectId ? 
                    'Note added to project successfully!' : 
                    `Note added successfully! ${result.action === 'new' ? 'Created new project.' : 'Updated existing project.'}`;
                
                this.showStatus(statusMessage, 'success');
                noteInput.value = '';
                this.clearProjectSelection();
                
                // Auto-switch to projects view after a delay
                setTimeout(() => {
                    this.showSection('view-projects');
                    this.loadProjects();
                }, 2000);
            } else {
                this.showStatus(result.error || 'Failed to add note', 'error');
            }
        } catch (error) {
            console.error('Error submitting note:', error);
            this.showStatus('Network error. Please try again.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Add Note';
        }
    }

    async loadProjects() {
        const projectsList = document.getElementById('projects-list');
        
        try {
            const response = await fetch('/api/projects');
            const projects = await response.json();

            if (projects.length === 0) {
                projectsList.innerHTML = `
                    <div class="empty-state">
                        <h3>No projects yet</h3>
                        <p>Add your first note to get started!</p>
                    </div>
                `;
                return;
            }

            projectsList.innerHTML = projects.map(project => `
                <div class="project-card">
                    <div class="project-header">
                        <h3 class="project-title">${this.escapeHtml(project.name)}</h3>
                        <p class="project-description">${this.escapeHtml(project.description)}</p>
                        <div class="project-actions">
                            <button class="add-note-to-project-btn" data-project-id="${project.id}" data-project-name="${this.escapeHtml(project.name)}">
                                + Add Note
                            </button>
                            <button class="edit-project-btn" data-project-id="${project.id}">
                                ✏️ Edit
                            </button>
                            <button class="delete-project-btn" data-project-id="${project.id}" data-project-name="${this.escapeHtml(project.name)}">
                                🗑️ Delete
                            </button>
                        </div>
                    </div>
                    <div class="project-content">${this.escapeHtml(project.content)}</div>
                </div>
            `).join('');
            
            // Bind events for the new buttons
            this.bindProjectButtons();

        } catch (error) {
            console.error('Error loading projects:', error);
            projectsList.innerHTML = `
                <div class="empty-state">
                    <h3>Error loading projects</h3>
                    <p>Please try refreshing the page.</p>
                </div>
            `;
        }
    }

    bindProjectButtons() {
        // Add note buttons
        document.querySelectorAll('.add-note-to-project-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const projectId = parseInt(e.target.dataset.projectId);
                const projectName = e.target.dataset.projectName;
                this.selectProject(projectId, projectName);
            });
        });

        // Edit buttons
        document.querySelectorAll('.edit-project-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const projectId = parseInt(e.target.dataset.projectId);
                this.editProject(projectId);
            });
        });

        // Delete buttons
        document.querySelectorAll('.delete-project-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const projectId = parseInt(e.target.dataset.projectId);
                const projectName = e.target.dataset.projectName;
                this.deleteProject(projectId, projectName);
            });
        });
    }

    selectProject(projectId, projectName) {
        this.selectedProjectId = projectId;
        this.showSection('add-note');
        this.updateNoteFormForProject(projectName);
    }

    clearProjectSelection() {
        this.selectedProjectId = null;
        this.updateNoteFormForProject(null);
    }

    updateNoteFormForProject(projectName) {
        const noteForm = document.querySelector('.note-form');
        let projectIndicator = document.getElementById('project-indicator');
        
        if (projectName) {
            if (!projectIndicator) {
                projectIndicator = document.createElement('div');
                projectIndicator.id = 'project-indicator';
                projectIndicator.className = 'project-indicator';
                noteForm.insertBefore(projectIndicator, noteForm.firstChild);
            }
            projectIndicator.innerHTML = `
                <span class="project-label">Adding note to: <strong>${projectName}</strong></span>
                <button type="button" class="clear-project-btn" onclick="app.clearProjectSelection()">×</button>
            `;
        } else {
            if (projectIndicator) {
                projectIndicator.remove();
            }
        }
    }

    async deleteProject(projectId, projectName) {
        if (!confirm(`Are you sure you want to delete the project "${projectName}"? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/projects/${projectId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus('Project deleted successfully!', 'success');
                this.loadProjects(); // Refresh the list
            } else {
                this.showStatus(result.error || 'Failed to delete project', 'error');
            }
        } catch (error) {
            console.error('Error deleting project:', error);
            this.showStatus('Network error. Please try again.', 'error');
        }
    }

    async editProject(projectId) {
        try {
            // Get current project data
            const response = await fetch('/api/projects');
            const projects = await response.json();
            const project = projects.find(p => p.id === projectId);

            if (!project) {
                this.showStatus('Project not found', 'error');
                return;
            }

            // Create edit form
            this.showEditForm(project);
        } catch (error) {
            console.error('Error preparing edit:', error);
            this.showStatus('Failed to load project for editing', 'error');
        }
    }

    showEditForm(project) {
        const projectsList = document.getElementById('projects-list');
        
        const editFormHtml = `
            <div class="edit-project-form">
                <h3>Edit Project</h3>
                <div class="form-group">
                    <label for="edit-project-name">Name:</label>
                    <input type="text" id="edit-project-name" value="${this.escapeHtml(project.name)}">
                </div>
                <div class="form-group">
                    <label for="edit-project-description">Description:</label>
                    <textarea id="edit-project-description" rows="3">${this.escapeHtml(project.description)}</textarea>
                </div>
                <div class="form-group">
                    <label for="edit-project-content">Content:</label>
                    <textarea id="edit-project-content" rows="10">${this.escapeHtml(project.content)}</textarea>
                </div>
                <div class="form-actions">
                    <button class="save-project-btn" data-project-id="${project.id}">Save Changes</button>
                    <button class="cancel-edit-btn">Cancel</button>
                </div>
            </div>
        `;

        projectsList.innerHTML = editFormHtml;
        
        // Bind form events
        document.querySelector('.save-project-btn').addEventListener('click', (e) => {
            const projectId = parseInt(e.target.dataset.projectId);
            this.saveProject(projectId);
        });

        document.querySelector('.cancel-edit-btn').addEventListener('click', () => {
            this.loadProjects();
        });
    }

    async saveProject(projectId) {
        const name = document.getElementById('edit-project-name').value.trim();
        const description = document.getElementById('edit-project-description').value.trim();
        const content = document.getElementById('edit-project-content').value;

        if (!name || !description) {
            this.showStatus('Name and description are required', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/projects/${projectId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, description, content })
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus('Project updated successfully!', 'success');
                this.loadProjects(); // Refresh the list
            } else {
                this.showStatus(result.error || 'Failed to update project', 'error');
            }
        } catch (error) {
            console.error('Error saving project:', error);
            this.showStatus('Network error. Please try again.', 'error');
        }
    }

    showStatus(message, type) {
        const status = document.getElementById('submit-status');
        status.textContent = message;
        status.className = `status ${type}`;
        status.style.display = 'block';

        // Auto-hide success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }
    }

    escapeHtml(text) {
        // Handle non-string inputs
        if (typeof text !== 'string') {
            if (text === null || text === undefined) {
                return '';
            }
            // Convert to string if it's an object or other type
            text = typeof text === 'object' ? JSON.stringify(text, null, 2) : String(text);
        }
        
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new NotesApp();
});