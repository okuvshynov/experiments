# Qwen Image Generation Server

A Flask-based web server for generating images using the Qwen-Image diffusion model. The server provides a job queue system, web UI, and API endpoints for submitting and managing image generation jobs.

## Features

- Web-based UI for submitting image generation requests
- Job queue system for processing requests sequentially
- Real-time job status updates
- Image viewing and downloading
- RESTful API endpoints

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

3. Open your browser and navigate to:
- Main interface: http://localhost:5000/
- Jobs page: http://localhost:5000/jobs

## Usage

### Web Interface

1. **Generate Image**: Navigate to the main page, enter your prompt and parameters, then click "Generate Image"
2. **View Jobs**: Go to the jobs page to see all submitted jobs, their status, and download completed images

### API Endpoints

- `POST /api/submit_job` - Submit a new image generation job
  ```json
  {
    "prompt": "Your image description",
    "negative_prompt": "What to avoid",
    "width": 320,
    "height": 240,
    "steps": 50,
    "cfg_scale": 4.0,
    "seed": 42
  }
  ```

- `GET /api/job/<job_id>` - Get status of a specific job
- `GET /api/jobs` - Get all jobs
- `GET /download/<job_id>` - Download generated image
- `GET /view/<job_id>` - View generated image

## Configuration

The server uses the following default settings:
- Model: Qwen/Qwen-Image
- Default image size: 320x240
- Default inference steps: 50
- Default CFG scale: 4.0

Generated images are saved in the `generated_images/` directory.

## Notes

- The first image generation will take longer as the model needs to be loaded
- Jobs are processed sequentially in the order they are submitted
- The server automatically detects and uses CUDA, MPS, or CPU based on availability