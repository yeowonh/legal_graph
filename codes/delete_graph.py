import argparse
import json
from GraphDB.LegalGraphDB import LegalGraphDB


"""
python delete_graph.py --db_name legal-graph
"""

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--config_path", type=str, default="../codes/configs/config.json", help="config_path")
    g.add_argument("--db_name", type=str, required=True, help="database name")
    args = parser.parse_args()

    print("## Settings ##")
    print("## db_name : ", args.db_name)
    print("## config_path : ", args.config_path)

    # config_file 가져오기
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        print(f"Loaded config data from {args.config_path}")

    dbms = LegalGraphDB(auradb=False, config=config, json_path="../data/graph/clause/title_embedding/01/01_law_main.json")
    # 이전 graph 삭제
    dbms.delete_all_relationship()
    dbms.delete_all_nodes()

if __name__ == "__main__":
    exit(main())