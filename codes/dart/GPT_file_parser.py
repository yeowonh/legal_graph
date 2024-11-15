"""
GPT_parsing_test.ipynb
의 py 버전 -> 터미널 실행 가능

python3 codes/GPT_file_parser.py\
    --model_name gpt-4o\
    --txt_path /mnt/c/Users/Shic/Downloads/legal_graph/DCM/dart/110_공시서류확인및서명/110_1_제도안내/제도안내.txt\
    --prompt_path /mnt/c/Users/Shic/Downloads/legal_graph/codes/prompt/gpt_parsing_prompt.txt\
    --result_folder_path /mnt/c/Users/Shic/Downloads/legal_graph/DCM/tmp\
    --describe gpt-4o
"""

import tiktoken
import sys, os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from GPT_cost_calculator import num_tokens_from_string, cost_calculator
from parsing_utils import delete_newlines


# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--model_name", type=str, required=True, help="model name to parsing")
g.add_argument("--txt_path", type=str, required=True, help="txt (html) file path")
g.add_argument("--prompt_path", type=str, required=True, help="prompt txt file path")
g.add_argument("--result_folder_path", type=str, required=True, default="../DCM/dart_GPT_markdown/", help="path to save result markdown txt")
g.add_argument("--describe", type=str,  help="additional name will be saved in result file name")


def main(args):
    # Openai 불러오기
    load_dotenv(verbose=True)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    RESULT_PATH = args.result_folder_path

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    # 프롬프트 불러오기
    with open(args.prompt_path, 'r', encoding='utf-8') as file:
        PROMPT = file.read()

    name = os.path.basename(args.txt_path)
    if name[-4:] != ".txt":
        raise ValueError("Input file must be text file. Check txt_path!")

    # 파일을 열고 txt 데이터를 파싱하여 불러오기
    with open(args.txt_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # new line 제거
    content = delete_newlines(content)

    print(f'## {args.txt_path} loaded ##')
    print(f'## {num_tokens_from_string(content, args.model_name)} tokens are in {name} ##')
    cost_calculator(len(content), args.model_name)


    completion = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": content
            }
        ]
    )

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        print(f"새로운 result folder 생성 : {RESULT_PATH}")

    # markdown으로 파싱된 파일 저장
    if args.describe != None:
        with open(os.path.join(RESULT_PATH, args.model_name + "_" + args.describe + "_"  + name), "w") as file:
            print(f"## {RESULT_PATH} 에 {args.model_name + '_' + args.describe + '_'  + name} 저장 ##")
            file.write(completion.choices[0].message.content)
    
    else:
        with open(os.path.join(RESULT_PATH, args.model_name + "_" + name), "w") as file:
            print(f"## {RESULT_PATH} 에 {args.model_name + '_' + name} 저장 ##")
            file.write(completion.choices[0].message.content)


if __name__ == "__main__":
    exit(main(parser.parse_args()))