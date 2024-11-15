import argparse
import json
import re
import sys, os


# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legal.utils import parse_law, parse_supplementary, parse_contents

parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--txt_path", type=str, required=True, help="txt file path")
g.add_argument("--save_path", type=str, required=True, help="save clause path")

"""
## preprocessing.py
python preprocessing.py --txt_path ../data/DCM/DCM_original/08_은행법/08_law_main.txt --save_path ../data/DCM/DCM_json/08/08_law_main.json
python preprocessing.py --txt_path ../data/DCM/DCM_original/08_은행법/08_enforcement_main.txt --save_path ../data/DCM/DCM_json/08/08_enforcement_main.json
"""

# 편 > 장 > 절 > 관 > 조
def main(args):
    print(f"## Preprocess file path in {args.txt_path} ##")
    with open(args.txt_path, "r", encoding="utf-8") as f:
        contents = f.readlines()

    document_title = contents[0].replace("\ufeff", "")
    date, revise_info = re.split(r" \[", contents[1])
    date = date.replace("[", "").replace("]", "")
    revise_info = revise_info.replace("[", "").replace("]", "")

    source = contents[-2].strip()

    meta_info = {
        "document_title": document_title,
        "date": date,
        "revise_info": revise_info,
        "source": source
    }

    articles_list = []

    if "main" in args.txt_path:
        print("## main 법령 처리 ##")
        content_text_list = ''.join(contents[2:]).split("\n" * 4)

        content_text = content_text_list[1]
        articles_list = parse_law(content_text, meta_info)

        # content_text_list length 2 초과인 경우는 부칙이 붙음
        if len(content_text_list) > 2:
            print("## main 법령 내부 부칙 추가 처리 ##")
            articles_list += parse_law(content_text_list[2], meta_info)

    else: # supplementary
        print("## 부칙 / 별표 처리 ##")
        content_text_list = ''.join(contents[2:]).split("\n" * 4)[1:]
        for content_text in content_text_list:
            articles_list += parse_supplementary(content_text, meta_info)
    
    # 항 나누기
    articles_list = parse_contents(articles_list)

    os.makedirs('/'.join(args.save_path.split('/')[:-1]), exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(articles_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    exit(main(parser.parse_args()))
