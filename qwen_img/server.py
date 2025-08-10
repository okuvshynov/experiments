from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from diffusers import DiffusionPipeline # type: ignore
import torch
import uuid
import os
from datetime import datetime
import threading
import queue
import time
from pathlib import Path
import traceback
import argparse

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'generated_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_NAME = "Qwen/Qwen-Image"

# Default server settings – may be overridden via CLI arguments
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5005
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Job storage (in-memory)
jobs = {}
job_queue = queue.Queue()

# Job status enum
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Job model
class Job:
    def __init__(self, prompt, negative_prompt="", width=320, height=240, steps=50, cfg_scale=4.0, seed=42):
        self.id = str(uuid.uuid4())
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.completed_at = None
        self.image_path = None
        self.error = None
        
    def to_dict(self):
        return {
            'id': self.id,
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'width': self.width,
            'height': self.height,
            'steps': self.steps,
            'cfg_scale': self.cfg_scale,
            'seed': self.seed,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'image_path': self.image_path,
            'error': self.error
        }

# Global pipeline variable
pipeline = None
pipeline_lock = threading.Lock()

def initialize_pipeline():
    global pipeline
    with pipeline_lock:
        if pipeline is None:
            print("Initializing Qwen-Image pipeline...")
            # Determine device and dtype
            if torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float32
            else:
                device = "cpu"
                torch_dtype = torch.float32
            
            print(f"Using device: {device}, dtype: {torch_dtype}")
            
            # Load the pipeline
            pipeline = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)
            pipeline = pipeline.to(device)
            print("Pipeline initialized successfully!")

def process_job(job):
    try:
        # Update job status
        job.status = JobStatus.PROCESSING
        
        # Initialize pipeline if not already done
        initialize_pipeline()
        
        # Generate image
        with pipeline_lock:
            # Determine device

            device = pipeline.device # type: ignore
            
            generator = torch.Generator(device=str(device)).manual_seed(job.seed)
            
            images = pipeline(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.width,
                height=job.height,
                num_inference_steps=job.steps,
                true_cfg_scale=job.cfg_scale,
                generator=generator,
                num_images_per_prompt=1
            ).images # type: ignore
        
        # Save image
        image_filename = f"{job.id}.png"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        images[0].save(image_path)
        
        # Update job
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.image_path = image_filename
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        print(f"Error processing job {job.id}: {e}")
        traceback.print_exc()

def worker():
    """Background worker to process jobs from the queue"""
    while True:
        try:
            job_id = job_queue.get(timeout=1)
            if job_id in jobs:
                job = jobs[job_id]
                print(f"Processing job {job_id}...")
                process_job(job)
                print(f"Job {job_id} completed with status: {job.status}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")
            traceback.print_exc()

# Start background worker thread
worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/jobs')
def jobs_page():
    return render_template('jobs.html')

@app.route('/api/submit_job', methods=['POST'])
def submit_job():
    """API endpoint to submit an image generation job.

    If ``width`` or ``height`` are omitted from the request payload, the
    server‑wide defaults (which can be overridden via command‑line arguments) are
    used.
    """
    data = request.json

    # Validate required fields
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400

    # Use defaults for width/height when not provided by the client
    width = int(data.get('width', DEFAULT_WIDTH))
    height = int(data.get('height', DEFAULT_HEIGHT))

    # Create new job
    job = Job(
        prompt=data['prompt'],
        negative_prompt=data.get('negative_prompt', ''),
        width=width,
        height=height,
        steps=int(data.get('steps', 50)),
        cfg_scale=float(data.get('cfg_scale', 4.0)),
        seed=int(data.get('seed', 42))
    )
    
    # Store job and add to queue
    jobs[job.id] = job
    job_queue.put(job.id)
    
    return jsonify({'job_id': job.id, 'status': job.status})

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify(job.to_dict())

@app.route('/api/jobs')
def get_all_jobs():
    # Return jobs sorted by creation time (newest first)
    sorted_jobs = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
    return jsonify([job.to_dict() for job in sorted_jobs])

@app.route('/download/<job_id>')
def download_image(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    
    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED or not job.image_path:
        return "Image not available", 404
    
    image_path = os.path.join(UPLOAD_FOLDER, job.image_path)
    if not os.path.exists(image_path):
        return "Image file not found", 404
    
    return send_file(image_path, as_attachment=True, download_name=f"{job.id}.png")

@app.route('/view/<job_id>')
def view_image(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    
    job = jobs[job_id]
    if job.status != JobStatus.COMPLETED or not job.image_path:
        return "Image not available", 404
    
    image_path = os.path.join(UPLOAD_FOLDER, job.image_path)
    if not os.path.exists(image_path):
        return "Image file not found", 404
    
    return send_file(image_path, mimetype='image/png')

def parse_args():
    """Parse command‑line arguments for server configuration.

    The arguments allow overriding the host, port, and the default image width
    and height used when a client does not specify them in the request.
    """
    parser = argparse.ArgumentParser(description="Run Qwen‑Image Flask server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the Flask server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the Flask server")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Default image width if not provided in request")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Default image height if not provided in request")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Override global defaults with CLI values
    DEFAULT_HOST = args.host
    DEFAULT_PORT = args.port
    DEFAULT_WIDTH = args.width
    DEFAULT_HEIGHT = args.height
    app.run(debug=True, host=DEFAULT_HOST, port=DEFAULT_PORT)
