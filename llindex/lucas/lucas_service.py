# this is main entry point for the service

from flask import Flask, request, jsonify
import os
import json
import logging
import requests
import sys
import time

from lucas.llm_client import client_factory
from lucas.index_format import format_default
from lucas.tools.toolset import Toolset

app = Flask(__name__)

# TODO: use DB here
jobs = {}

@app.route('/query', methods=['POST'])
def query():
    query = request.json

    codebase_path = query['directory']
    index_file = os.path.join(codebase_path, ".llidx")

    if not os.path.isfile(index_file):
        logging.warning(f"The index file '{index_file}' does not exist. Continue without index.")
        index_formatted = ""
    else:
        with open(index_file, 'r') as f:
            index = json.load(f)

        logging.info('loaded index')
        index_formatted = format_default(index)

    script_dir = os.path.dirname(__file__)

    with open(os.path.join(script_dir, 'prompts', 'query_with_tools.txt')) as f:
        prompt = f.read()

    message = query['message']
    task = f'<task>{message}</task>'
    user_message = prompt + index_formatted + '\n\n' + task

    client = client_factory(query['client'])
    toolset = Toolset(codebase_path)

    reply = client.send(user_message, toolset)

    return jsonify({"reply": reply}), 200

@app.route('/jobs', methods=['POST'])
def create_job():
    job_data = request.json
    job_id = str(len(jobs) + 1)  # Simple ID generation
    jobs[job_id] = job_data
    return jsonify({"id": job_id, "message": "Job created successfully"}), 201

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify(job)
    return jsonify({"error": "Job not found"}), 404

@app.route('/jobs/<job_id>', methods=['PUT'])
def update_job(job_id):
    if job_id in jobs:
        jobs[job_id] = request.json
        return jsonify({"message": "Job updated successfully"})
    return jsonify({"error": "Job not found"}), 404

@app.route('/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    if job_id in jobs:
        del jobs[job_id]
        return jsonify({"message": "Job deleted successfully"})
    return jsonify({"error": "Job not found"}), 404

@app.route('/jobs', methods=['GET'])
def list_jobs():
    return jsonify(jobs)

def process_directory(directory):
    # Implement your directory processing logic here
    # For example, list files, perform operations, etc.
    files = os.listdir(directory)
    return {"processed_files": files}

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    app.run(debug=True)
