import multiprocessing
import os.path
import random

import requests
from retrying import retry
from tqdm import tqdm
import json
import ssl
import re
from collections import defaultdict
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import glob
from .gpt4class_chatanywhere import PostRobot






class GPTGenerator:

    def __init__(self, 
                 data_set: str,
                 generator: str,
                 args: dict={},
                 user_name: str=None,
                 api_key: str=None,
                 organization: str=None,
                 is_api_keys: bool=False,
                 n_processes: int=50,
                 save_for_eval: bool=True):
        
        self.data_set = data_set
        self.generator_name = generator
        self.args = args
        self.user_name = user_name
        self.api_key = api_key
        self.organization = organization
        self.is_api_keys = is_api_keys
        self.n_processes = n_processes
        self.save_for_eval = save_for_eval

        self.data_file = os.path.join('gpt_generation/data', f'{data_set}.jsonl')
        self.files_dir = os.path.join('gpt_generation/files', data_set, self.generator_name)
        # self._create_folder(self.files_dir)

        self.sample_list = []

        self._init_generator(self.generator_name)

    @staticmethod
    def _create_folder(path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _save_json(data: dict, path: str):
        """Save a dict to json file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, encoding='utf-8', mode='w') as fw:
            fw.write(json.dumps(data, indent=4, ensure_ascii=False))

    @staticmethod
    def _save_jsonl(data: List[dict], path: str):
        """Save a list to jsonl file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
    
    def _init_generator(self, generator: str):
        """Initialize"""
        assert generator in ('gpt-3.5-turbo', 'gpt-4-api','gpt-4-api-chatanywhere')
        if generator == 'gpt-3.5-turbo':
            assert self.user_name is not None and self.user_name != '', "Please set your name"
            from gpt import GPT
            self.generator = GPT(user_name=self.user_name, new_version='0.1.0')

        elif generator == 'gpt-4-web':
            import sys
            sys.path.append('..')
            from chatgpt_wrapper.main_browser_wrapper import wrapper_init
            self.generator = wrapper_init()
            self.generator.singleCall('/model gpt4')
            self.n_processes = 1

        elif generator == 'gpt-4-api':
            assert self.is_api_keys or (self.api_key is not None and self.organization is not None), "Please set your api key and organization"
            if self.is_api_keys:
                self._load_keys()
            self.generator = self._call_gpt4
        
        elif generator == 'gpt-4-api-chatanywhere':
            assert self.api_key is not None, "Please set your api key"
            self.generator = PostRobot(self.api_key)

    def _read_sample(self) -> None:
        """
        Reads a competitor's sample data from a JSONL file and stores it in the final_result dictionary.
        """
        with open(self.data_file, encoding='utf-8', mode='r') as reader:
            self.sample_list = [json.loads(line) for line in reader.read().strip().split('\n')]
   
    # @retry(wait_fixed=10000, stop_max_attempt_number=3)
    def _request_gen_turbo(self, content):
        """Request turbo"""
        from gpt import GPT
        self.generator = GPT(user_name=self.user_name, new_version='0.1.0')
        flag, result = self.generator.call(content, args=self.args)
        if result == "context_length_exceeded":
            return "context_length_exceeded"
        if result == "" or not flag or result == "Error":
            raise ValueError
        return result
    
    def _request_gen_gpt4_web(self, content):
        """Request gpt4 to review"""
        self.generator.singleCall('/new')
        result_items = self.generator.backend.ask(content, title=None, model_customizations={})
        flags, result = result_items[0], result_items[1]
        if not flags or result == "":
            raise ValueError
        return result
    
    def _request_gen_gpt4_api_chatanywhere(self, content):
        flag = False
        try_time = 0
        while not flag and try_time < 7:
            try_time += 1
            try:
                flag, message =  self.generator.generate(content, args=self.args)
                if not flag:
                    print(f'error: {message}')
            except Exception as e:
                print('报错：',e)
        if not flag:
            raise ValueError('ChatGPT请求失败')
        return message
    
    @retry(wait_fixed=10000, stop_max_attempt_number=3)
    def _request_gen_gpt4_api(self, content):
        """Request gpt4 to generate (API)"""
        result = self.generator(content, args=self.args)
        if result == "":
            raise ValueError
        return result

    def _load_keys(self):
        gpt4_keys = open('utils/gpt4_keys').read().strip().split('\n')
        keys = []
        for key in gpt4_keys:
            api_key = 'sk-' + key.split('sk-')[1].split('|')[0].split('-')[0]
            organization = 'org-' + key.split('org-')[1].split('|')[0].split('-')[0]
            keys.append((api_key, organization))
        self.keys = keys

    def _call_gpt4(self, content, args):
        url = "https://api.openai.com/v1/chat/completions"
        if self.is_api_keys:
            api_key, organization = random.choice(self.keys)
        else:
            api_key = self.api_key
            organization = self.organization
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }
        parameters = {
            "model": 'gpt-4',
            "messages": [{'role': 'user', 'content': content}],
            **args,
        }
        response = requests.post(
            url,
            headers=headers,
            json=parameters
        )
        response = json.loads(response.content.decode("utf-8"))
        return response['choices'][0]['message']['content']

    def _request_gen(self, sample):
        """Request one generation"""
        index = sample["id"]
        content = sample["query"]

        output_file = os.path.join(self.files_dir, f"{index}.json")
        if os.path.exists(output_file):
            return -1
        
        if self.generator_name == 'gpt-3.5-turbo':
            result = self._request_gen_turbo(content)
        elif self.generator_name == 'gpt-4-web':
            result = self._request_gen_gpt4_web(content)
        elif self.generator_name == 'gpt-4-api':
            result = self._request_gen_gpt4_api(content)
        elif self.generator_name == 'gpt-4-api-chatanywhere':
            result = self._request_gen_gpt4_api_chatanywhere(content)

        sample["output"] = result
        self._save_json(sample, output_file)
        return index

    def request_gen(self):
        """Request multiple generation"""
        for sample in tqdm(self.sample_list, desc="Processing samples", unit="sample"):
            if not os.path.exists(os.path.join(self.files_dir, str(sample["id"]) + ".json")):
                self._request_gen(sample)

    def request_gen_mp(self):
        """Request multiple generation via multiprocessing"""
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = [
                pool.apply_async(self._request_gen, args=(sample,))
                for sample in self.sample_list
                if not os.path.exists(os.path.join(self.files_dir, str(sample["id"]) + ".json"))
            ]
            for r in tqdm(results, desc="Processing samples", unit="sample"):
                r.wait()

            result_list = [r.get() for r in results]
            pool.close()
            pool.join()
    
    def find_sample_by_id(self, id, data_list):
        """Return the sample that matches id"""
        for data in data_list:
            if data["id"] == id:
                return data
            
    def _merge_output(self):
        """Merge outputs for one-time evaluation"""
        data_list = []
        for file in glob.glob(f'{self.files_dir}/*.json'):
            data_list.append(json.loads(open(file, encoding="utf-8").read()))

        sample_list = [self.find_sample_by_id(sample['id'], data_list) for sample in self.sample_list]
        return sample_list
    
    def save(self, sample_list: List[dict]):
        if self.generator_name == 'gpt-3.5-turbo':
            save_name = 'turbo'
        elif self.generator_name.startswith('gpt-4'):
            save_name = 'gpt4'

        if self.save_for_eval:
            output_file = os.path.join('data/', self.data_set, f'{save_name}.jsonl')
        else:
            output_file = os.path.join('gpt_generation/outputs', f'{self.data_set}_{save_name}.jsonl')
        self._save_jsonl(sample_list, output_file)

    def generate(self):
        self._read_sample()

        if self.n_processes == 1:
            self.request_gen()
        else:
            self.request_gen_mp()
        
        sample_list = self._merge_output()
        self.save(sample_list)



