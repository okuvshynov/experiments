class NotesApp {
    constructor() {
        this.currentSection = 'add-note';
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
            const response = await fetch('/api/notes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ content })
            });

            const result = await response.json();

            if (response.ok) {
                this.showStatus(
                    `Note added successfully! ${result.action === 'new' ? 'Created new project.' : 'Updated existing project.'}`, 
                    'success'
                );
                noteInput.value = '';
                
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
                    </div>
                    <div class="project-content">${this.escapeHtml(project.content)}</div>
                </div>
            `).join('');

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
document.addEventListener('DOMContentLoaded', () => {
    new NotesApp();
});