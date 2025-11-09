import os
from pathlib import Path
from openai import OpenAI
import time



def upload_file(client, file_path):
    print(f"正在上传包含请求信息的JSONL文件...")
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    print(f"文件上传成功。得到文件ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(client, input_file_id):
    print(f"正在基于文件ID，创建Batch任务...")
    batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Batch任务创建完成。 得到Batch任务ID: {batch.id}\n")
    return batch.id

def check_job_status(client, batch_id):
    print(f"正在检查Batch任务状态...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Batch任务状态: {batch.status}\n")
    return batch.status

def get_output_id(client, batch_id):
    print(f"正在获取Batch任务中执行成功请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"输出文件ID: {batch.output_file_id}\n")
    return batch.output_file_id

def get_error_id(client, batch_id):
    print(f"正在获取Batch任务中执行错误请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"错误文件ID: {batch.error_file_id}\n")
    return batch.error_file_id

def download_results(client, output_file_id, output_file_path):
    print(f"正在打印并下载Batch任务的请求成功结果...")
    content = client.files.content(output_file_id)
    # 打印部分内容以供测试
    print(f"打印请求成功结果的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存结果文件至本地
    content.write_to_file(output_file_path)
    print(f"完整的输出结果已保存至本地输出文件result.jsonl\n")

def download_errors(client, error_file_id, error_file_path):
    print(f"正在打印并下载Batch任务的请求失败信息...")
    content = client.files.content(error_file_id)
    # 打印部分内容以供测试
    print(f"打印请求失败信息的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存错误信息文件至本地
    content.write_to_file(error_file_path)
    print(f"完整的请求失败信息已保存至本地错误文件error.jsonl\n")
