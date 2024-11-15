## 01_자본시장과금융투자업에관한법률
# python add_embedding.py --json_path ../data/graph/clause/title_embedding/01/01_law_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/01/01_law_main.json
# python add_embedding.py --json_path ../data/graph/clause/title_embedding/01/01_enforcement_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/01/01_enforcement_main.json
# python add_embedding.py --json_path ../data/graph/clause/title_embedding/01/01_order_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/01/01_order_main.json

python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/01/ --edges_json_path ../data/graph/clause/edge_triplet/01/01_refers_to_triplets.json --embedding

## 02_금융지주회사감독규정
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_law_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_law_main.json
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_enforcement_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_enforcement_main.json
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_order_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_order_main.json

# python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/02/ --edges_json_path ../data/graph/clause/edge_triplet/02/02_refers_to_triplets_postprocess.json --embedding

## 03_증권의발행및공시등에관한규정
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_law_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_law_main.json
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_enforcement_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_enforcement_main.json
# python add_embedding.py --json_path ../data/DCM/DCM_json/02/02_order_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/02/02_order_main.json

# python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/03/ --edges_json_path ../data/graph/clause/edge_triplet/03/03_refers_to_triplets_postprocess.json --embedding

## 04_kofia
#python add_embedding.py --json_path ../data/DCM/DCM_json/04/04_regulation_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/04/04_regulation_main.json

# python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/04/ --edges_json_path ../data/graph/clause/edge_triplet/04/04_refers_to_triplets_postprocess.json --embedding

## 08_은행법
# python add_embedding.py --json_path ../data/DCM/DCM_json/08/08_law_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/08/08_law_main.json
# python add_embedding.py --json_path ../data/DCM/DCM_json/08/08_enforcement_main.json --config_path ./configs/config.json --save_path ../data/graph/clause/embedding/08/08_enforcement_main.json

# python make_clause_graph.py --process_type graph --input_nodes_folder_path ../data/graph/clause/embedding/08/ --edges_json_path ../data/graph/clause/edge_triplet/08/08_refers_to_triplets_postprocess.json --embedding
