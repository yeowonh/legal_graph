import argparse
import json
import re

"""

python codes/postprocess_refers_to_triplet.py --postprocess_path ./data/graph/clause/edge_triplet/04/04_refers_to_triplets.json --save_path ./data/graph/clause/edge_triplet/04/04_refers_to_triplets_postprocess.json
0n_refers_to_triplets.json 후처리 진행
"""

def main():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--postprocess_path", type=str, help="postprocess file path")
    g.add_argument("--save_path", type=str, help="postprocess file path")
    args = parser.parse_args()

    print("## Settings ##")
    print(f"## postprocessing in {args.postprocess_path} ##")

    # data list load
    # data/data_list_meta.json
    with open("../data/data_list_meta.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
        data_list = data_list['data_list']

    # print(data_list)

    total_matches = []
    # Pattern to match law names inside 「」
    law_pattern = r'「(.*?)」'
    # Pattern to match content between closing 」 and article number
    after_law_pattern = r'\s*(?:\([^)]*\)\s*)*'

    # Pattern to match article numbers after law name
    article_pattern = r'(제[\d가-힣]+조(?:의[\d가-힣]+)?(?:제[\d가-힣]+항)?)'
    data_list_count = {}
    for key in data_list.keys():
        data_list_count[key] = 0

    with open(args.postprocess_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    print('## import data on {args.postprocess_path} | length : {len()}')

    for row in data[0]:
        text = row['contents']

        for law_match in re.finditer(law_pattern, text):
            # Extract law name
            law_name = law_match.group(1)
            # Remove parentheses and content inside from law name
            law_name_clean = re.sub(r'\(.*?\)', '', law_name)
            # Add back the 「」 around law name
            law_name_with_brackets = f'「{law_name_clean}」'
            
            # Position after closing 」 in the text
            pos_after_law = law_match.end()
            
            # Skip any content inside parentheses and spaces
            after_law_match = re.match(after_law_pattern, text[pos_after_law:])
            if after_law_match:
                pos_after_skipped = pos_after_law + after_law_match.end()
            else:
                pos_after_skipped = pos_after_law
            
            # Search for article number starting from pos_after_skipped
            article_match = re.match(article_pattern, text[pos_after_skipped:])
            if article_match:
                # Extract article number
                article_number = article_match.group(1)
                
                # 제21조의2에 와 같은 경우 '에' 조사 제거
                article_number = article_number.replace("에", "")

                # Create tuple and add to list
                total_matches.append((law_name_with_brackets, article_number))
                # print('## law name :', law_name_with_brackets, 'article number :', article_number)

                if article_number == row['target_index']:
                    if law_name_with_brackets in data_list.keys():
                        # print(f"## target doc changed {row['target_document']} -> {data_list[law_name_with_brackets]}")
                        row['target_document'] = data_list[law_name_with_brackets]
                        data_list_count[law_name_with_brackets] += 1


                    else:
                        # print(f"## target doc {law_name_with_brackets} not in our documents!\n\n")
                        row['target_document'] = "not_in_my_document"

    print('## 변경된 데이터 숫자 ##')
    for key, value in data_list_count.items():
        print(f"## {key} data | {value} 개 변경")
    
    
    #refers_to_triplets를 json 형태로 저장
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(data[0], f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    exit(main())