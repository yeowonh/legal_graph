#!/bin/bash

# JSON 파일 경로 리스트
file_paths=(
    "../data/graph/clause/embedding/01/01_law_main.json"
    "../data/graph/clause/embedding/01/01_enforcement_main.json"
    "../data/graph/clause/embedding/01/01_order_main.json"
    # "../data/graph/clause/title_embedding/02/02_law_main.json"
    # "../data/graph/clause/title_embedding/02/02_enforcement_main.json"
    # "../data/graph/clause/title_embedding/02/02_order_main.json"
    # "../data/graph/clause/title_embedding/03/03_regulation_main.json"
    # "../data/graph/clause/title_embedding/08/08_enforcement_main.json"
    # "../data/graph/clause/title_embedding/08/08_law_main.json"
)

# 각 파일에 대해 postprocess_json.py 스크립트 실행
for file_path in "${file_paths[@]}"
do
    echo "Processing $file_path..."
    python postprocess_json.py --file_path "$file_path"
done

echo "All files processed."
