"""
GPT parsing


- 전체 dart folder path 넣으면 markdown 변환된 jsonl 파일로 결과물 산출되도록
- input : dart folder root path (default = "../DCM/dart/")
- output : jsonl files 
    - dart_html.jsonl // dart html + hwp 원본 parsing 파일
    - dart_markdown.jsonl // dart markdown + hwp 변환 파일

- jsonl 내에서,
    - 웹페이지 html 파일은 title = 120_3_작성사례 으로 저장됨
    - 한글 첨부파일은 title = 120_3_작성사례_3_120.MD&A_재무상태_및_영업실적3 으로 저장됨
python3 codes/dart/GPT_folder_parser.py

python codes/DART/GPT_folder_parser.py --model_name gpt-4o --txt_folder_path DCM/1-2/html --prompt_path codes/dart/prompt/gpt_parsing_prompt.txt  --result_folder_path DCM/1-2/tmp  --describe gpt-4o

"""
import sys, os
import argparse

from parsing_utils import delete_newlines, markdown_parsing, has_subdirectory
import json
from tqdm import tqdm

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 현재 스크립트의 경로를 기준으로 절대 경로로 ROOT_PATH 설정
#ROOT_PATH = '../DCM/dart/'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_PATH = os.path.join(PARENT_DIR, 'DCM', 'dart')

parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--model_name", type=str, default="gpt-4o", help="model name to parsing")
g.add_argument("--txt_folder_path", type=str, default=ROOT_PATH, help="txt (html) folder path")
g.add_argument("--prompt_path", type=str, default=os.path.join(PARENT_DIR, "prompt/", "gpt_parsing_prompt.txt"), help="prompt txt file path")
g.add_argument("--result_folder_path", type=str, default=os.path.join(PARENT_DIR, "DCM/", "dart_GPT_markdown/"), help="path to save result markdown txt")
g.add_argument("--describe", type=str, help="additional name will be saved in result file name")



def main(args):
    RESULT_PATH = args.result_folder_path
    html_list = []
    markdown_list = []

    # 프롬프트 불러오기
    with open(args.prompt_path, 'r', encoding='utf-8') as file:
        PROMPT = file.read()

    if os.path.exists(args.txt_folder_path):
        dir_list = os.listdir(args.txt_folder_path)
        for dir in tqdm(dir_list):
            directory_path = os.path.join(args.txt_folder_path, dir)
            subdir_list = os.listdir(directory_path)
            
            for file_name in subdir_list:
                subdirectory_path = os.path.join(os.path.join(directory_path, file_name))

                # 탭 단위로 나눠진 경우
                try:
                    has_subdirectory(subdirectory_path)
                    # print("has subdirectory")
                    # print(subdirectory_path.split('/')[-1])

                    for tab_file_name in os.listdir(subdirectory_path):
                        # 한글 파일 처리
                        # 3_120.MD&A_재무상태_및_영업실적3.hwp.txt -> 이렇게 저장되어 있다고 가정할 때
                        if tab_file_name.split('.')[-2] == "hwp":
                            # 파일 열고 txt 파싱하여 불러오기
                            with open(os.path.join(subdirectory_path, tab_file_name), 'r', encoding='utf-8') as file:
                                content = file.read()
                            
                            # new line 제거
                            content = delete_newlines(content)

                            # 여기서 dict 형식으로 저장하기
                            # title은 저장된 txt 파일 이름 가져오고, metadata는 상위 디렉토리 이름 가져오기
                            html_list.append({'title' : subdirectory_path.split('/')[-1] + '_' + tab_file_name.split('.')[0], 'content' : content, 'metadata' : directory_path.split('/')[-1]})

                            markdown_list.append({'title' : subdirectory_path.split('/')[-1] + '_' + tab_file_name.split('.')[0],
                                                  'content' : markdown_parsing(content, PROMPT, args.model_name),
                                                  'metadata' : directory_path.split('/')[-1]})


                        elif tab_file_name.endswith('.txt'):
                            # 파일 열고 txt 파싱하여 불러오기
                            with open(os.path.join(subdirectory_path, tab_file_name), 'r', encoding='utf-8') as file:
                                content = file.read()
                            
                            # new line 제거
                            content = delete_newlines(content)


                            # 여기서 dict 형식으로 저장하기
                            # title은 저장된 txt 파일 이름 가져오고, metadata는 상위 디렉토리 이름 가져오기
                            html_list.append({'title' : subdirectory_path.split('/')[-1], 'content' : content, 'metadata' : directory_path.split('/')[-1]})

                            markdown_list.append({'title' : subdirectory_path.split('/')[-1],
                                                  'content' : markdown_parsing(content, PROMPT, args.model_name),
                                                  'metadata' : directory_path.split('/')[-1]})
                        
                        else:
                            raise ValueError(f"## 파일 확장자 확인 필요 : {tab_file_name}")


                # 하위에 바로 txt 파일이 있는 경우
                except:
                    if file_name.split('.')[-2] == "hwp":
                        # 파일을 열고 txt 데이터를 파싱하여 불러오기
                        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        # new line 제거
                        content = delete_newlines(content)


                        # 여기서 dict 형식으로 저장하기
                        # title은 저장된 txt 파일 이름 가져오고, metadata는 상위 디렉토리 이름 가져오기
                        html_list.append({'title' : directory_path.split('/')[-1] + '_' + file_name.split('.')[0], 'content' : content, 'metadata' : directory_path.split('/')[-1]})
                        markdown_list.append({'title' : directory_path.split('/')[-1] + '_' + file_name.split('.')[0], 
                                              'content' : markdown_parsing(content, PROMPT, args.model_name), 
                                              'metadata' : directory_path.split('/')[-1]})


                    elif file_name.endswith('.txt'):
                        # 파일을 열고 txt 데이터를 파싱하여 불러오기
                        with open(os.path.join(directory_path, file_name), 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        # new line 제거
                        content = delete_newlines(content)


                        # 여기서 dict 형식으로 저장하기
                        # title은 저장된 txt 파일 이름 가져오고, metadata는 상위 디렉토리 이름 가져오기
                        html_list.append({'title' : directory_path.split('/')[-1], 'content' : content, 'metadata' : directory_path.split('/')[-1]})
                        markdown_list.append({'title' : directory_path.split('/')[-1], 
                                              'content' : markdown_parsing(content, PROMPT, args.model_name), 
                                              'metadata' : directory_path.split('/')[-1]})
                    
                    else:
                        raise ValueError(f"## 파일 확장자 확인 필요 : {tab_file_name}")


    else:
        raise ValueError(f"There is no directory in {args.txt_folder_path}")


    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        print(f"## 새로운 result folder 생성 : {RESULT_PATH} ##")

    # markdown으로 파싱된 파일 저장
    if args.describe != None: # 추가 description 있을 때
        with open(os.path.join(RESULT_PATH, f"dart_html_{args.describe}.jsonl"), encoding="utf-8", mode = "w") as file:
            for html in html_list:
                file.write(json.dumps(html, ensure_ascii=False) + '\n')
        with open(os.path.join(RESULT_PATH, f"dart_markdown_{args.describe}.jsonl"), encoding="utf-8", mode = "w") as file:
            for markdown in markdown_list:
                file.write(json.dumps(markdown, ensure_ascii=False) + '\n')

    else:
        with open(os.path.join(RESULT_PATH, "dart_html.jsonl"), encoding="utf-8", mode = "w") as file:
            for html in html_list:
                file.write(json.dumps(html, ensure_ascii=False) + '\n')
        with open(os.path.join(RESULT_PATH, "dart_markdown.jsonl"), encoding="utf-8", mode = "w") as file:
            for markdown in markdown_list:
                file.write(json.dumps(markdown, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    exit(main(parser.parse_args()))