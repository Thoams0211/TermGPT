import os
from pathlib import Path
from openai import OpenAI
import time



def upload_file(client, file_path):
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    return file_object.id

def create_batch_job(client, input_file_id):
    batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    return batch.id

def check_job_status(client, batch_id):
    batch = client.batches.retrieve(batch_id=batch_id)
    return batch.status

def get_output_id(client, batch_id):
    batch = client.batches.retrieve(batch_id=batch_id)
    return batch.output_file_id

def get_error_id(client, batch_id):
    batch = client.batches.retrieve(batch_id=batch_id)
    return batch.error_file_id

def download_results(client, output_file_id, output_file_path):
    content = client.files.content(output_file_id)
    content.write_to_file(output_file_path)

def download_errors(client, error_file_id, error_file_path):
    content = client.files.content(error_file_id)
    content.write_to_file(error_file_path)
