import argparse
import json
import re
import sys, os
import copy
from tqdm import tqdm

"""
## preprocessing_clause.py

- 이미 전처리 되어 있는 파일을 리스트 -> 항 단위로 쪼개는 전처리 파일
- python preprocessing_clause.py --txt_path ../results/1-7/DCM_1-7_law_main.json --save_path ../results/1-7/DCM_1-7_law_main_clause.json
- python preprocessing_clause.py --txt_path ../results/1-7/DCM_1-7_enforcement_main.json --save_path ../results/1-7/DCM_1-7_enforcement_main_clause.json
- python preprocessing_clause.py --txt_path ../results/1-7/DCM_1-7_order_main.json --save_path ../results/1-7/DCM_1-7_order_main_clause.json

python preprocessing_clause.py --txt_path ../data/DCM/DCM_json/08/08_law_main.json --save_path  ../data/DCM/DCM_json/08/08_law_main.json 
python preprocessing_clause.py --txt_path ../data/DCM/DCM_json/08/08_enforcement_main.json --save_path  ../data/DCM/DCM_json/08/08_enforcement_main.json 


- results/1-n/파일이름_clause.json으로 저장됨
"""


# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--txt_path", type=str, required=True, help="txt file path")
g.add_argument("--save_path", type=str, required=True, help="save clause path")

# def insert_list_at_position(original_list, list_to_insert, position):
#     # 리스트를 두 부분으로 나누고 사이에 다른 리스트를 끼워넣기
#     new_list = original_list[:position] + list_to_insert + original_list[position:]
#     return new_list


def remove_elements_by_indices(original_list, indices_to_remove):
    # 인덱스 번호 리스트를 내림차순으로 정렬
    indices_to_remove = sorted(indices_to_remove, reverse=True)
    
    # 각 인덱스에 해당하는 요소를 리스트에서 제거
    for index in indices_to_remove:
        if 0 <= index < len(original_list):  # 인덱스가 유효한 경우에만 제거
            del original_list[index]
    
    return original_list



def main(args):
    print(f"## Preprocess file path in {args.txt_path} ##")
    article_index = {
        "①" : "1항", "②" : "2항", "③" : "3항", "④" : "4항", "⑤" : "5항",
        "⑥" : "6항", "⑦" : "7항", "⑧" : "8항", "⑨" : "9항", "⑩" : "10항",
        "⑪" : "11항", "⑫" : "12항", "⑬" : "13항", "⑭" : "14항", "⑮" : "15항",
        "⑯" : "16항", "⑰" : "17항", "⑱" : "18항", "⑲" : "19항", "⑳" : "20항",
        "㉑" : "21항", "㉒" : "22항", "㉓" : "23항", "㉔" : "24항", "㉕" : "25항",
        "㉖" : "26항", "㉗" : "27항", "㉘" : "28항", "㉙" : "29항", "㉚" : "30항",
        "㉛" : "31항", "㉜" : "32항", "㉝" : "33항", "㉞" : "34항", "㉟" : "35항",
        "㊱" : "36항", "㊲" : "37항", "㊳" : "38항", "㊴" : "39항", "㊵" : "40항",
        "㊶" : "41항", "㊷" : "42항", "㊸" : "43항", "㊹" : "44항", "㊺" : "45항",
        "㊻" : "46항", "㊼" : "47항", "㊽" : "48항", "㊾" : "49항", "㊿" : "50항"
    }
    
    def split_article(json_path: str) -> list:
        # 파일을 열고 JSON 데이터를 파싱하여 불러오기
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        data_copy = copy.deepcopy(data)
        remove_idx = []
        new_rows_cnt = 0

        for idx, row in tqdm(enumerate(data)):
            if type(row['content']) == str:
                continue

            elif type(row['content']) == list:
                # article 단위 split하기
                remove_idx.append(idx + new_rows_cnt)

                for content in row['content']:
                    new_row = copy.deepcopy(row)
                    
                    if content[0] in article_index.keys():
                        new_article_index = article_index[content[0]] # 항 번호
                        new_row['content'] = content[1:].strip()
                        new_row['index'] += "제" + new_article_index
                        data_copy.insert(idx + new_rows_cnt + 1, new_row)
                        new_rows_cnt += 1

                    else: # 항 바깥에 조가 있는 경우
                        new_row = copy.deepcopy(row)
                        new_row['content'] = ''.join(row['content'])
                        data_copy.insert(idx + new_rows_cnt + 1, new_row)
                        new_rows_cnt += 1
                        break

            else:
                raise ValueError("not str or list")
        
        data_copy = remove_elements_by_indices(data_copy, remove_idx)

        return data_copy
    

    clauses_list = split_article(args.txt_path)

    print(f'## save clause file in {args.save_path}')
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(clauses_list, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    exit(main(parser.parse_args()))
