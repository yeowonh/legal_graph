# 프로젝트의 루트 디렉토리를 sys.path에 추가
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
import json
from typing import Dict
import copy
import time

import GraphDB.utils as utils
import codes.GraphDB.process as process
from GraphDB.LegalGraphDB import LegalGraphDB

load_dotenv(verbose=True)

## openai==1.42.0 으로 버전 맞춰야 함 -> 아니면 오류..

import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

"""
run in wsl terminal

<triplet 추출 ~ 그래프 구축>
python3 make_keyword_graph.py --data_type 법 \
    --original_file_path ../results/1-2/DCM_1-2_law_main_clause.json \
    --input_jsonl_path ../results/1-2/jsonl/01_law_main_definition_text.jsonl \
    --triplet_jsonl_path ../results/1-2/jsonl/01_law_main_triplet_output.jsonl \
    --meta_path ../results/1-2/jsonl/01_law_main_definition_metadata.csv

<그래프 구축만>
python3 make_keyword_graph.py --data_type 법 \
    --original_file_path ../results/1-2/DCM_1-2_law_main_clause.json \
    --input_jsonl_path ../results/1-2/jsonl/01_law_main_definition_text.jsonl \
    --triplet_jsonl_path ../results/1-2/jsonl/01_law_main_triplet_output.jsonl \
    --meta_path ../results/1-2/jsonl/01_law_main_definition_metadata.csv \
    --resume True

in window
python make_keyword_graph.py --data_type 법 --original_file_path ../results/1-2/DCM_1-2_law_main_clause.json --input_jsonl_path ../results/1-2/jsonl/01_law_main_definition_text.jsonl --triplet_jsonl_path ../results/1-2/jsonl/01_law_main_triplet_output.jsonl --meta_path ../results/1-2/jsonl/01_law_main_definition_metadata.csv
"""


def main(config: Dict):
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--data_type", type=str, required=True, help="법/시행령/시행규칙")
    g.add_argument("--original_file_path", type=str, required=True, help="original json file path to make keyword")
    g.add_argument("--input_jsonl_path", type=str, required=True, help="input jsonl file path")
    g.add_argument("--triplet_jsonl_path", type=str, required=True, help="triplet jsonl output file path")
    g.add_argument("--meta_path", type=str, required=True, help="metacsv file path to saved")
    g.add_argument("--model", type=str, default=config["model"], help="model to make triplets")
    g.add_argument("--prompt_path", type=str, default=config["keyword_prompt_path"], help="prompt_path to make triplets")
    g.add_argument("--preprocess", type=str, default="True", help="Determine preprocess texts")
    g.add_argument("--resume", type=str, default="False", help="resume from triplet files")

    args = parser.parse_args()

    print("## Settings ##")
    print("## original_file_path : ", args.original_file_path)
    print("## input_jsonl_path : ", args.input_jsonl_path)
    print("## triplet_jsonl_path : ", args.triplet_jsonl_path)
    print("## meta_path : ", args.meta_path)
    print("## prompt_path : ", args.prompt_path)
    print("## model : ", args.model)
    print("## resume : ", args.resume)
    
    if args.resume == "False":
        # extract definition sentences
        same_sentences, include_sentences = utils.extract_definition_sentences(args.original_file_path)
        print(f"## 동의어 관계 : {len(same_sentences)}\nexample : {same_sentences[0]}\n\n포함 관계 : {len(include_sentences)}\nexample : {include_sentences[0]}")
        definition_sentences = copy.deepcopy(same_sentences)

        for sentence in include_sentences:
            if sentence not in definition_sentences:
                definition_sentences.append(sentence)
            else:
                continue
        
        print(f"## 전체 관계 합집합 : {len(definition_sentences)}\n\n동의어 관계 : {len(same_sentences)}\nexample : {same_sentences[0]}\n\n포함 관계 : {len(include_sentences)}\nexample : {include_sentences[0]}")

        # preprocess
        if args.preprocess == "True":
            definition_sentences = process.replace_hanja(definition_sentences)
        
        # chunk_split
        splitted_definition_sentences = []
        for item in definition_sentences:
            text_chunks = utils.chunk_split(item['content'], config['chunk_size'], config['chunk_overlap'])
            
            if len(text_chunks) == 1:
                splitted_definition_sentences.append(item)

            else:
                for chunked_text in text_chunks:
                    copy_item = copy.deepcopy(item)
                    copy_item['content'] = chunked_text
                    splitted_definition_sentences.append(copy_item)
        
        print(f"## 청킹 전 item 개수 : {len(definition_sentences)}, 청킹 후 item 개수 : {len(splitted_definition_sentences)}")

        # make triplet by chunked item
        utils.make_triplet_jsonl(splitted_definition_sentences,
                            data_type=args.data_type,
                            prompt_path=args.prompt_path,
                            input_jsonl_path=args.input_jsonl_path,
                            meta_path=args.meta_path)

        batch_result = utils.make_triplet_batch(input_jsonl_path = args.input_jsonl_path)
        batch_id = batch_result.id

        while True:
            status = utils.get_batch_status(batch_id)
            print('## current status : ', status.status)

            if status.status == "completed":
                output_file_id = status.output_file_id
                print(f'## batch completed!\noutput file id : {output_file_id}')
                break

            else:
                print('## Wait untill completed...')
                time.sleep(60)

        utils.get_batch_answer(output_file_id=output_file_id, result_path=args.triplet_jsonl_path)

    triplets = utils.get_triplet(result_path=args.triplet_jsonl_path, meta_path=args.meta_path)

    print(f"## triplet example : {triplets[0]}")

    include = [x[1] for x in triplets if x[1]['type'] == "INCLUDED_IN"]
    same = [x[1] for x in triplets if x[1]['type'] == "SAME_AS"]

    print(f"## Before postprocess\n\nINCLUDED_IN : {len(include)} | SAME_AS : {len(same)} | TOTAL : {len(include) + len(same)}")

    with open('../codes/GraphDB/remove_idx.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # postprocess
    if args.data_type == '법':
        # JSON 파일 열기
        manual_remove_idx = data['law_main_remove_idx']

    elif args.data_type == '시행령':
        # JSON 파일 열기
        manual_remove_idx = data['enforcement_main_remove_idx']

    elif args.data_type == '시행규칙':
        manual_remove_idx = data['order_main_remove_idx']
    
    elif args.data_type == '규정':
        manual_remove_idx = data['regulation_main_remove_idx']

    else:
        raise ValueError("Undefined data type : Choose 법 / 시행령 / 시행규칙 / 규정")


    
    removed_idx, new_triplets = process.postprocess(triplets, manual_remove_idx)
    
    print(f"## After postprocess\n\nremoved_triplets : {len(removed_idx)} | TOTAL: {len(new_triplets)}")

    # print('#### removed triplets ####')
    # for idx in removed_idx:
    #     print(f"## removed triplet : {triplets[idx][0]['keyword_name']} - {triplets[idx][1]['type']} - {triplets[idx][2]['keyword_name']}")

    # db load
    dbms = LegalGraphDB(auradb=False, config=config, json_path="../data/graph/clause/title_embedding/01/01_law_main.json")

    # keyword graph 구축
    print("## make keyword graph ##")
    dbms.create_keyword_graph(new_triplets)


if __name__ == "__main__":
    with open(f'../codes/configs/config.json', 'r') as f:
        config = json.load(f)
    
    main(config=config)