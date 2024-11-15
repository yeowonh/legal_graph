import tiktoken
import sys, os

import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

total_rows = 0
total_tokens = 0


"""
각 GPT 모델 사용 시 비용을 계산

python3 codes/GPT_cost_calculator.py --file_path ../results/1-2/DCM_1-2_law_main.json --model_name gpt-3.5-turbo

"""


# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--file_path", type=str, required=True, help="txt file path")
g.add_argument("--model_name", type=str, required=True, help="model name to calculate")

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def json_count_tokens(json_path: str, model_name: str):
    contents = []
    content_token_length = []

    global total_rows

    # 파일을 열고 JSON 데이터를 파싱하여 불러오기
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    total_rows += len(data)
    for row in data:
        if type(row['content']) == str:
            contents.append(row['content'])
            content_token_length.append(num_tokens_from_string(row['content'], model_name))

        elif type(row['content']) == list:
            contents += row['content']
            content_token_length.append(sum([num_tokens_from_string(x, model_name) for x in row['content']]))

        else:
            raise ValueError("not str or list")

    sns.displot(content_token_length, kind='kde')
    plt.xlim(0, )
    plt.title(json_path.split("/")[-1])

    print("- Total row count : ", len(content_token_length))
    print("- max tokens : ", max(content_token_length))
    print("- min tokens : ", min(content_token_length))
    print("- avg tokens : ", sum(content_token_length)/len(content_token_length))
    return contents, content_token_length

def txt_count_tokens(txt_path: str, model_name: str):
    # 파일을 열고 txt 데이터를 파싱하여 불러오기
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    content_token_length = num_tokens_from_string(content, model_name)
    print("- Total token count : ", content_token_length)

    return content_token_length

def cost_calculator(input_token_length: int, model_name: str):
    if model_name == "gpt-3.5-turbo":
        print(f"## input price : {3 * (input_token_length / 1000000)} $")
        print(f"## batch input price : {1.5 * (input_token_length / 1000000)} $")
    elif model_name == "gpt-4-turbo":
        print(f"## input price : {10 * (input_token_length / 1000000)} $")
        print(f"## batch input price : {5 * (input_token_length / 1000000)} $")
    elif model_name == "chatgpt-4o-latest":
        print(f"## input price : {5 * (input_token_length / 1000000)} $")
        print(f'## No batch input ##')

    return 

def main(args):
    print(f"## Token count in {args.file_path}##")
    print(f"## Calculate cost by model: {args.model_name}##")

    print(args.file_path[-5:])
    if args.file_path[-5:] == ".json":
        _, token_length = json_count_tokens(args.file_path, args.model_name)
        cost_calculator(sum(token_length), args.model_name)
    else:
        token_length = txt_count_tokens(args.file_path, args.model_name)
        cost_calculator(token_length, args.model_name)

if __name__ == "__main__":
    exit(main(parser.parse_args()))