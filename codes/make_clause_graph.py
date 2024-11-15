"""
json -> neo4j Graph 생성 
노드 생성, 엣지 생성 및 전체 그래프 생성을 지원,
임베딩 된 노드 생성 원할 시 embedding 옵션 추가 가능

Usage:
    python make_clause_graph.py --process_type <graph|node|edge> \
                                --input_nodes_folder_path <input_json_folder> \
                                --edges_json_path <edges_json_file> \
                                --embedding 
    
Arguments:
    --process_type:             Choose the type of process to run: 'graph', 'node', or 'edge'
    --input_nodes_folder_path:  Path to folder containing node JSON files
    --edges_json_path:          Path to JSON file for creating edges (required for 'graph' and 'edge' types)
    --embedding:                Enable embedding during node creation (optional)

Example:
    # Create a graph with embeddings (WSL example)
    python3 make_clause_graph.py --process_type graph \
        --input_nodes_folder_path ../data/graph/clause/embedding/ \
        --edges_json_path ../data/graph/clause/refers_to_triplets.json \
        --embedding

    # Create nodes only (WSL example)
    python3 make_clause_graph.py --process_type node \
        --input_nodes_folder_path ../data/DCM/DCM_json/ \
        --embedding

    # Create edges only (Windows CMD example)
    python make_clause_graph.py --process_type edge ^ 
        --input_nodes_folder_path ../data/graph/clause/embedding/ ^ 
        --edges_json_path ../data/graph/clause/refers_to_triplets.json

    # Create nodes without embedding (Windows CMD example)
    python make_clause_graph.py --process_type node ^ 
        --input_nodes_folder_path ..\data\DCM\DCM_json
"""
""""
[run in wsl terminal]
<graph>
python3 make_clause_graph.py --process_type graph \
    --input_nodes_folder_path ../data/graph/clause/embedding/  \
    --edges_json_path ../data/graph/clause/refers_to_triplets.json \
    --embedding  

<node>
python3 make_clause_graph.py --process_type node \
    --input_nodes_folder_path ../data/DCM/DCM_json/ \
    --embedding  

<edge> 
python3 make_clause_graph.py --process_type edge \
    --input_nodes_folder_path ../data/graph/clause/embedding/ \
    --edges_json_path ../data/graph/clause/refers_to_triplets.json

    
[run in windows cmd terminal]
<graph>
python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/03/ --edges_json_path ../data/graph/clause/edge_triplet/03/03_refers_to_triplets.json --embedding  

<node>
python make_clause_graph.py --process_type node ^ 
 --input_nodes_folder_path ..\data\DCM\DCM_json ^ 
 --embedding  

<edge>
python make_clause_graph.py --process_type edge --input_nodes_folder_path ../data/graph/clause/embedding/03/ --edges_json_path ../data/graph/clause/edge_triplet/03/03_refers_to_triplets.json
python make_clause_graph.py --process_type edge --input_nodes_folder_path ../data/graph/clause/embedding/ --edges_json_path ../data/graph/clause/edge_triplet/01/01_refers_to_triplets.json
"""


import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
print(f"Added {project_root} to sys.path")
from dotenv import load_dotenv
import argparse
import json
from typing import Dict, List
import logging
import GraphDB.utils as utils
from GraphDB.LegalGraphDB import LegalGraphDB

load_dotenv(verbose=True)

# 로깅 설정 (INFO 레벨)
logging.basicConfig(level=logging.INFO)


# 공통 함수: 노드 파일 처리 함수
def process_node_files(input_folder: str, dbms: LegalGraphDB, database: str, embedding: bool):
    if not os.path.exists(input_folder):
        raise ValueError(f"Provided input folder path {input_folder} does not exist.")
    # dbms.delete_all_nodes()
    print("######################## Make nodes ########################")
    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            file_path = os.path.join(input_folder, file)
            logging.info(f"Processing node file: {file}")
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                node_type = file.split(".")[0]
                for idx, item in enumerate(data):
                    dbms.create_clause_node(
                        node_type=f"Clause_{node_type}",
                        node_property=item,
                        embedding=embedding  #True => 임베딩 포함 노드 , False => 임베딩 미포함 노드
                    )
                logging.info(f"Create Node Num : {len(data)}")
            except Exception as e:
                logging.error(f"Failed to process node file {file}: {e}")
   
# 공통 함수: 엣지 파일 처리 함수
def process_edge_files(input_nodes_folder_path: str, edges_json_path: str, dbms: LegalGraphDB, database: str):
    # REFERS_TO 재구축
    #dbms.delete_all_relationship()
    # dbms.delete_specific_relationship("refers_to")
    print("######################## Make edges ########################")
    if not os.path.exists(edges_json_path):
        
        refers_to_triplets = []
        for file in os.listdir(input_nodes_folder_path):
            print("file: ", file)
            if file.endswith(".json"):
                
                #input_nodes_folder_path의 상위 폴더에 matched_pattern에 저장 
                edges_json_path = f"{edges_json_path}/{file}"
                # edges_json_path = os.path.join(edges_json_path, file)
                
                refers_to_triplets.append(utils.create_refers_to_triplets_list(edges_json_path))

        # 엣지 저장 및 neo4j로 전송
        try:
            with open(edges_json_path, 'w', encoding='utf-8') as f:
                json.dump(refers_to_triplets, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved triplets to {edges_json_path}")
        except Exception as e:
            logging.error(f"Failed to save triplets: {e}")
            pass 

    else : 
        try:
            with open(edges_json_path, 'r', encoding='utf-8') as f:
                refers_to_triplets = json.load(f)
            for triplet_list in refers_to_triplets:
                for triplet in triplet_list:
                    dbms.create_clause_relationship(triplet)
            logging.info(f"Created all edges from {edges_json_path} to NEO4j database:  {database}")
        except Exception as e:
            logging.error(f"Failed to create relationships in NEO4j: {e}")
            

def main(config: Dict):
    
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--process_type", type=str, required=True, choices=['graph', 'node', 'edge'], help="graph/node/edge")
    g.add_argument("--input_nodes_folder_path", type=str, help="Path to folder containing node JSON files")
    g.add_argument("--edges_json_path", type=str, help="Path to JSON file for creating edges")
    g.add_argument("--embedding", action='store_true', help="Enable embedding during node creation")

    args = parser.parse_args()

    logging.info(f"Process Type: {args.process_type}")
    database = config.get("database")
    dbms = LegalGraphDB(auradb=False, config=config, json_path="../data/graph/clause/title_embedding/01/01_law_main.json")
    
     # 그래프 생성
    if args.process_type == "graph":
        
        print("######################## Make graph ########################")
        #graph 전체 삭제 
        # dbms.delete_all_nodes()
        if not args.input_nodes_folder_path or not args.edges_json_path:
            raise ValueError("Both --input_nodes_folder_path and --edges_json_path are required for graph creation.")
        
        logging.info("Starting full graph creation process...")
        process_node_files(args.input_nodes_folder_path, dbms, database, args.embedding)
        process_edge_files(args.input_nodes_folder_path, args.edges_json_path, dbms, database)

    # 노드 생성
    elif args.process_type == "node":
        if not args.input_nodes_folder_path:
            raise ValueError("--input_nodes_folder_path is required for node creation.")
        
        logging.info("Starting node creation process...")
        process_node_files(args.input_nodes_folder_path, dbms, database, args.embedding)

    # 엣지 생성
    elif args.process_type == "edge":
        if not args.edges_json_path or not args.input_nodes_folder_path:
            raise ValueError("--edges_json_path and --input_nodes_folder_path are required for edge creation.")
        
        logging.info("Starting edge creation process...")
        process_edge_files(args.input_nodes_folder_path, args.edges_json_path, dbms, database)

if __name__ == "__main__":
    # config.json 파일 경로를 절대 경로로 설정
    config_path = f"{project_root}/codes/configs/config.json"
    # config_path = os.path.join(project_root, 'codes', 'configs', 'config.json')
    print(f"Using config file: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    main(config=config)
