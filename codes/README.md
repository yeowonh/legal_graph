## Codes

### make_keyword_graph

- triplet 추출 ~ 그래프 구축
```bash
python3 make_keyword_graph.py --data_type 법 \
    --original_file_path ../results/1-2/DCM_1-2_law_main_clause.json \
    --jsonl_path ../results/1-2/jsonl/01_law_main_definition_text.jsonl \
    --meta_path ../results/1-2/jsonl/01_law_main_definition_metadata.csv
```


- 그래프 구축만
``` bash
python3 make_keyword_graph.py --data_type 법 \
    --original_file_path ../results/1-2/DCM_1-2_law_main_clause.json \
    --jsonl_path ../results/1-2/jsonl/01_law_main_definition_text.jsonl \
    --meta_path ../results/1-2/jsonl/01_law_main_definition_metadata.csv
    --resume True
```
