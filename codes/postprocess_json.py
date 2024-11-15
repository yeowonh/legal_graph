"""
JSON 후처리 - '<(개정|신설) \d{4}\. \d{1,2}\. \d{1,2}\.?>' 패턴 제거 및 저장

사용 방법:
    python postprocess_json.py --file_path <후처리할_JSON_파일_경로>

인자 설명:
    --file_path: 후처리할 JSON 파일의 경로

예시:
    python postprocess_json.py --file_path ../data/graph/clause/embedding/01/01_law_main.json
"""

import re
import json
import argparse

def postprocess_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = re.compile(r'<(개정|신설) \d{4}\. \d{1,2}\. \d{1,2}\.?>')

    for item in data:
        title = item.get("metadata", {}).get("title", {})
        for key in title.keys():
            if title[key]:
                title[key] = patterns.sub("", title[key])
                title[key] = title[key].strip()

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Postprocessed JSON file saved at {file_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess JSON files to remove specified patterns.")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the JSON file to be postprocessed")
    return parser.parse_args()

def main():
    args = parse_args()
    postprocess_json(args.file_path)

if __name__ == "__main__":
    main()
