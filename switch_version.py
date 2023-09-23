import os
import json
import os
import shutil
import glob


if __name__ == '__main__':
    for top_directory in ('outputs', 'gpt4_api_outputs'):
        review_files = glob.glob(os.path.join(top_directory, '*', '*', '*', 'files/output0.jsonl'))
        for file in review_files:
            example_list = [json.loads(line) for line in open(file, encoding='utf-8').read().strip().split('\n')]

            prompt = example_list[0]['content'].split('[System]')[-1].strip()
            if 'a scale of 1 to 10' in prompt:
                evaluation_method = 'scoring'
            elif 'better' in prompt and 'worse' in prompt and 'equal' in prompt:
                evaluation_method = 'classification'

            if 'In the last line' in prompt:
                evaluation_method += '_cot'
            else:
                assert 'first output a single line'

            data_name =  os.path.normpath(file).split(os.sep)[1]
            pair_name = os.path.normpath(file).split(os.sep)[2]
            setting_name = os.path.normpath(file).split(os.sep)[3]
            os.makedirs(os.path.join(top_directory, data_name, pair_name, evaluation_method), exist_ok=True)
            shutil.move(
                os.path.join(top_directory, data_name, pair_name, setting_name), 
                os.path.join(top_directory, data_name, pair_name, evaluation_method, setting_name)
            )

