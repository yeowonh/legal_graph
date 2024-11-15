"""
JSON의 각 조항에 임베딩 벡터 추가 및 저장 
사용자는 입력 JSON 파일, 임베딩 모델이 포함된 설정 파일(config), 임베딩이 추가된 JSON 파일이 저장될 경로 지정

사용 방법:
    python add_embedding_to_json.py --json_path <입력_json_파일> --save_path <저장될_json_파일> --property <임베딩이_추가될_속성의_태그>
    
인자 설명:
    --json_path:     법률 조항이 포함된 입력 JSON 파일의 경로.
    --save_path:     임베딩이 추가된 JSON 파일이 저장될 경로.
    --property :         임베딩이 추가될 노드의 속성 태그. E.g., 'subtitle', 'content'

예시:
    python add_embedding_to_json.py --json_path ../data/DCM/DCM_json/01_order_main.json \
                                --save_path ../data/graph/clause/embedding/01_order_main.json
"""


import sys, os
import openai
from dotenv import load_dotenv
import argparse
import json
import logging
import GraphDB.utils as utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 환경 변수 로드
load_dotenv(verbose=True)

# 로깅 설정 (INFO 레벨)
logging.basicConfig(level=logging.INFO)

# 사용자 입력 파싱
def parse_args():
    parser = argparse.ArgumentParser(description="Process and add embeddings to legal clause JSON files.")
    
    parser.add_argument('--json_path', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the output JSON file with embeddings")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 임베딩 추가 및 저장
    utils.add_embedding_to_json(
        json_path=args.json_path,
        save_path=args.save_path
    )
    print(f"Embedding added to {args.json_path} and saved to {args.save_path}")

if __name__ == "__main__":
    main()
