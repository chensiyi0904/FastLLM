#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 13:01
# @Author  : chensy
# @Email  : chensiyi010904@163.com
# @File    : FastLLM.py
# @Software: PyCharm
import configparser
import csv
import queue
import threading
import time
from typing import Optional
from queue import Queue
import requests
from openai import OpenAI
from tqdm import tqdm


def model_exists(json_data, model_name):
    """
    查看ollama的对应的模型是否存在
    :param json_data: 请求api/tags看是否有对应的模型
    :param model_name: 模型名称
    :return:
    """
    models = json_data.get("models", [])
    for model in models:
        if model.get("name") == model_name:
            return True
    return False


def param_type_check(param):
    """
    尝试把参数浮点数化，比较暴力
    :param param: dict的参数
    :return: 修改后的param
    """

    def convert_to_float(in_value):
        """
        尝试将值转换为 float，如果无法转换，则返回原始值。

        :param in_value: 输入值
        :return: 转换后的 float 值或原始值
        """
        try:
            return float(in_value)
        except (ValueError, TypeError):
            return in_value

    for key, value in param.items():
        param[key] = convert_to_float(value)
    return param


class FastLLM:
    def __init__(
            self,
            config_file: Optional[str] = './config.ini',
    ):
        """
        初始化
        :param config_file: 如果没有提供就是config.ini，或是覆盖掉
        """
        self.progress_bar = None
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.task = Queue()
        self._initialize_params()
        self.writer_lock = threading.Lock()

    def _initialize_params(self):
        """
        初始化参数
        :return: None
        """
        # Reading parameters from the configuration file
        if 'ollama' in self.config:
            self.ollama_host = [host.strip() for host in self.config['ollama'].get('host', '').split("@")]
            self.ollama_port = [port.strip() for port in self.config['ollama'].get('port', "").split("@")]

        if 'online_llm' in self.config:
            self.online_llm_base_url = self.config['online_llm'].get('base_url', '').strip()
            self.online_llm_api_key = self.config['online_llm'].get('api_key', '').strip()
            self.thread_count = self.config['online_llm'].getint('thread_count', 0)

        if 'param' in self.config:
            self.model = self.config['param'].get('model', '').strip()
            self.param = param_type_check(dict(self.config.items('param')))

        if 'res_csv' in self.config:
            res_csv_path = (self.config['res_csv'].get('path', '').
                            strip()) if self.config['res_csv'].get('path', '').strip() != "" else "./res.csv"
            header = [port.strip() for port in self.config['res_csv'].get('header', "").split("@")]
            self.res_writer = csv.writer(open(res_csv_path, 'a', encoding='utf-8', newline=''))
            self.res_writer.writerow(header)

    def set_task(self, task: list[str]):
        """
        设置需要完成的任务
        :param task: 任务列表，由str类型的prompt构成的list
        :return: None
        """
        self.progress_bar = tqdm(total=len(task), desc="Processing Tasks", unit="task")
        for i, t in enumerate(task):
            self.task.put((i, t))  # 将任务元组放入队列

    def local_llm_generate(self):
        """
        使用本地大模型进行推理
        :return: None
        """

        def request(local_host, local_port):
            """
            进行本地大模型请求
            :param local_host: 本地大模型ip
            :param local_port: 本地大模型端口号
            :return:
            """
            while not self.task.empty():
                try:
                    current_task = self.task.get(timeout=1)  # 设置超时时间，避免死锁
                except queue.Empty:
                    break  # 如果队列为空，结束线程

                # 请求处理
                task_id, task_prompt = current_task
                ollama_url = f'http://{local_host}:{local_port}/v1'
                try:
                    client = OpenAI(
                        base_url=ollama_url,
                        api_key="required but unused",
                    )
                    ollama_response = client.chat.completions.create(
                        **self.param,
                        messages=[
                            {"role": "user", "content": task_prompt},
                        ]
                    )
                    response_content = str(ollama_response.choices[0].message.content)
                    with self.writer_lock:
                        self.res_writer.writerow(
                            [task_id, time.time(), "success", ollama_url, task_prompt, response_content])
                except Exception as e:
                    self.res_writer.writerow(
                        [task_id, time.time(), "fail", ollama_url, task_prompt])

                    print(f"Error processing task {task_id} on {ollama_url}: {e}")
                finally:
                    self.task.task_done()  # 标记任务完成
                    self.progress_bar.update(1)

        threads = []  # 保存所有线程
        for host, port in zip(self.ollama_host, self.ollama_port):
            models_url = f'http://{host}:{port}/api/tags'
            response = requests.get(models_url)
            try:
                response_data = response.json()
            except ValueError:
                print(f"Failed to parse JSON from {models_url}")
                continue
            if model_exists(response_data, self.model):
                for _ in range(3):  # 为每个 host-port 对创建3个线程
                    thread = threading.Thread(target=request, args=(host, port))
                    threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        self.task.join()
        self.progress_bar.close()
        print("All tasks have been processed.")
