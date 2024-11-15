import argparse
import json
import re
import GraphDB.utils as utils
from GraphDB.LegalGraphDB import LegalGraphDB
import os, sys
from typing import Dict
from tqdm import tqdm

"""
hierarchy relation 생성

- clause graph의 REFERS_TO는 이미 구축되어 있는 것이라 가정

python make_hierarchy.py
"""


def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    
    g.add_argument("--json_path", type=str, default="../data/graph/clause/title_embedding/01/01_law_main.json", help="json_file_path")

    args = parser.parse_args()
    clause_type = "Clause_" + args.json_path.split('/')[-1].split('.')[0]
    title_prefix = "Title_" + args.json_path.split('/')[-1].split('.')[0] + "_"

    index_dict = {"편" : title_prefix+"doc", "장" : title_prefix+"chapter", "절" : title_prefix+"section", "관" : title_prefix+"subsection", 
                  "조" : clause_type, "항" : clause_type}


    print("## Settings ##")
    print("## json_path : ", args.json_path)
    print("## clause type : ", clause_type)
    print('## title_prefix : ', title_prefix)

    dbms = LegalGraphDB(auradb=False, config=config)


    # json 불러오기
    with open(args.json_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    # node_type, index, title
    def get_node_info(text:str) -> list:
        parts = text.split(' ', 1) # doc, 제10편, 벌칙 | clause, 제224조제2항, 벌칙
        
        if len(parts) == 2:
            index, title = parts
        else:
            index = parts[0]
            title = ''.join(parts[1:])

        if index[-1] in index_dict.keys():
            return index_dict[index[-1]], index, title

        # 제11조의2 같은 경우 - 모두 항으로 취급
        else:
            print(f"Exception case in {text}")
            return clause_type, index, title


    # print(data[0])
    total_meta = []
    print(len(data))

    ######## 전처리 단계 ############
    # row 단위로 돌면서 확인
    # for row in data:
    #     hierarchy = []

    #     for key, value in row['metadata']['title'].items():
    #         if value != None and key != 'supplementary':
    #             # 제7편 거래소 <개정 2013. 5. 28.> 같은 경우 전처리
    #             if re.search(r'<개정 \d{4}\. \d{1,2}\. \d{1,2}\.?>', value) != None:
    #                 value = re.sub(r'<개정 \d{4}\. \d{1,2}\. \d{1,2}\.?>', '', value).strip()

    #             #  "제5장 온라인소액투자중개업자 등에 대한 특례 <신설 2015. 7. 24.>",
    #             if re.search(r'<신설 \d{4}\. \d{1,2}\. \d{1,2}\.?>', value) != None:
    #                 value = re.sub(r'<신설 \d{4}\. \d{1,2}\. \d{1,2}\.?>', '', value).strip()

    #             hierarchy.append(value)

    #     # 삭제 항이 아닐 경우에만 & 부칙이 아닐 경우에만 추가
    #     if row['subtitle'] != None and row['metadata']['title']['supplementary'] == None:
    #         hierarchy.append(row['index'] + " " + row['subtitle'])
    #         total_meta.append(hierarchy)

    # print(total_meta)

    ########## 수정 코드 #############
    # 순회하면서 path 추가



    ########### 그래프 생성 코드 #########
    # 생성된 total meta 2개씩 순회
    # for hierarchy_list in tqdm(total_meta):
    #     for idx in range(len(hierarchy_list)-1):
    #         # first node
    #         first_index = hierarchy_list[idx]

    #         # second node
    #         second_index = hierarchy_list[idx+1]
    #         first_node_type, first_law_index, first_name = get_node_info(first_index)
    #         second_node_type, second_law_index, second_name = get_node_info(second_index)

    #         # 첫 번재 노드가 존재하는지 확인
    #         first_node_id = dbms.get_node_id(first_node_type, "law_index", first_law_index, "name", first_name)

    #         if not first_node_id:
    #             # 노드 생성 - create_node
    #             first_node_id = dbms.create_node({"node_type" : first_node_type, "law_index" : first_law_index, "name" : first_name})
            
            
    #         # 두 번째 노드가 존재하는지 확인
    #         second_node_id = dbms.get_node_id(second_node_type, "law_index", second_law_index, "name", second_name)
    #         if not second_node_id:
    #             # 노드 생성 - create_node
    #             second_node_id = dbms.create_node({"node_type" : second_node_type, "law_index" : second_law_index, "name" : second_name})
            
    #         # id를 바탕으로 relationship 생성
    #         dbms.create_relationship("hierarchy", first_node_id, second_node_id)

    

if __name__ == "__main__":
    # config.json 파일 경로를 절대 경로로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    config_path = f"{project_root}/codes/configs/config.json"
    # config_path = os.path.join(project_root, 'codes', 'configs', 'config.json')
    print(f"Using config file: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    main(config=config)
