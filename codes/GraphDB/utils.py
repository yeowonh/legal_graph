import json, os, sys
import re
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import copy
from tqdm import tqdm
import ast
from dotenv import load_dotenv
from llama_index.core import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from GraphDB.LegalGraphDB import LegalGraphDB

from datetime import datetime
import contextlib
import logging

import pprint

"""
- load_client_and_config(): OpenAI 클라이언트와 설정 파일 로드 및 초기화
- get_gpt_answer(query: str, system_prompt: str) -> str: 시스템 프롬프트에 따른 GPT 답변 반환 (입력: `query`(str), `system_prompt`(str), 출력: str)
- extract_user_keyword(query: str) -> list: 사용자 쿼리에서 키워드 추출하여 리스트로 반환 (입력: `query`(str), 출력: list)
- text_embed(text: str): 텍스트 임베딩 생성 (입력: `text`(str))
- extract_keyword(text: str): 텍스트에서 키워드 추출 (입력: `text`(str))
- add_embedding_to_json(json_path: str, embed_model: str, save_path: str, config: dict): JSON 읽어서 임베딩 추가 후 JSON 저장 (입력: `json_path`(str), `embed_model`(str), `save_path`(str), `config`(dict))
- merge_to_json(path: str, save_path="../../results", filtering="_clause.json"): 여러 JSON 파일을 하나의 JSON 파일로 병합 (입력: `path`(str), `save_path`(str), `filtering`(str))
- flatten_json(nested_json, parent_key='', sep='_'): JSON 객체 평탄화 (입력: `nested_json`(dict), `parent_key`(str), `sep`(str))
- json_chunk_split(json_path: str, save_path: str): JSON 파일을 청크로 분할하여 저장 (입력: `json_path`(str), `save_path`(str))
- chunk_split(text: str, chunk_size: int, chunk_overlap: int): 주어진 청크 크기와 중복에 따라 텍스트 분할 (입력: `text`(str), `chunk_size`(int), `chunk_overlap`(int))
- search_json(json_path: str, word1: str, word2: str): 특정 키워드를 포함하는 법 조항 찾기 및 반환 (입력: `json_path`(str), `word1`(str), `word2`(str))
- extract_definition_sentences(json_path: str): 특정 패턴에 해당하는 정의 문장 추출 (메타데이터 포함) (입력: `json_path`(str))
- split_clause(text) -> list[str]: 조항을 분할하여 리스트로 반환 (입력: `text`(str), 출력: list[str])
- make_triplet_jsonl(clause_list: list, data_type: str, prompt_path: str, input_jsonl_path: str, meta_path: str, model_name="gpt-4o") -> pd.DataFrame: GPT 배치 파서를 위한 JSONL 생성 및 메타데이터 저장 (입력: `clause_list`(list), `data_type`(str), `prompt_path`(str), `input_jsonl_path`(str), `meta_path`(str), `model_name`(str), 출력: `pd.DataFrame`)
- make_triplet_batch(input_jsonl_path): GPT-4o API를 사용한 삼중항 추출 (배치 단위) (입력: `input_jsonl_path`(str))
- cancel_triplet_batch(id_list, custom_id=None): GPT 배치 요청 취소 (입력: `id_list`, `custom_id`)
- get_batch_status(batch_id): 배치 ID로 상태 확인 (입력: `batch_id`)
- get_batch_answer(output_file_id, result_path): 출력 파일 ID로 배치 답변 받아서 저장 (입력: `output_file_id`, `result_path`)
- get_triplet(result_path: str, meta_path: str) -> list: GPT 배치 결과를 처리하여 삼중항 리스트 생성 (입력: `result_path`(str), `meta_path`(str), 출력: list)
- load_json_as_documents(input_file): JSON 파일을 Document 리스트로 로드 (입력: `input_file`)
- split_documents(documents, chunk_size, chunk_overlap): 문서 분할 (입력: `documents`, `chunk_size`, `chunk_overlap`)
- extract_document_information(input_path): 입력 경로에서 문서 정보 추출 (입력: `input_path`(str))
- find_clause_pattern(input_path, documents): 문서에서 조항 패턴 찾기 (입력: `input_path`(str), `documents`)
- index_to_article(text): 인덱스에서 조, 항, 목 추출 (입력: `text`(str))
- article_to_index(article, clause=None, subarticle=None): 조, 항, 목을 인덱스로 변환 (입력: `article`, `clause`, `subarticle`)
- process_multiple_pattern(first, second): 다중 패턴 처리 (입력: `first`, `second`)
- process_range_pattern(start_index, end_index, edges, current_index): 범위 패턴 처리 (입력: `start_index`, `end_index`, `edges`, `current_index`)
- extract_refers_to_edges(matched_dict): 탐지한 패턴에서 엣지 추출 (입력: `matched_dict`)
- create_refers_to_triplets_list(input_path): 입력 경로에서 삼중항 리스트 생성 (입력: `input_path`(str))
"""

load_dotenv(verbose=True)

# 공통 client, config 캐시
client = None
config = None
dbms = None
# 로깅 설정
logging.basicConfig(filename='make_clause_graph.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# client와 config를 한 번만 로드하여 재사용
# dbms도 마찬가지
def initialize():
    global client, config, dbms

    if client is None:
        # client 초기화 (client 생성 코드는 상황에 맞게 작성)
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # client 초기화 함수는 구현해야 함
        print('## client loaded ##')

    if config is None:
        path = os.getcwd()
        root_path = os.path.dirname(path)

        # os path issue
        # config_path = os.path.join(root_path, 'codes', 'configs', 'config.json')
        config_path = f"configs/config.json"

        with open(config_path, 'r') as f:
            config = json.load(f)

        print('## config file loaded ##')
        
        
    if dbms is None:
        dbms = LegalGraphDB(auradb=False, config=config, json_path="../data/graph/clause/title_embedding/01/01_law_main.json")
        print('## config file loaded ##')

    


# system prompt에 따른 GPT 답변 받아오기
def get_gpt_answer(query: str, system_prompt: str) -> str:
    initialize()
    
    global client, config

    completion = client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=config['max_tokens']
    )
    
    print(f"###### Query #####")
    print(query)

    print("###### System Prompt #####")
    print(system_prompt)
    
    answer = completion.choices[0].message.content
    print(f"###### Answer #####")
    print(answer)
    return answer


# 사용자 쿼리에서 키워드 추출하여 리스트로 반환
def extract_user_keyword(query: str) -> list:
    initialize()
    global config 
    with open(config["query_keyword_prompt_path"], 'r', encoding='utf-8') as file:
        USER_KEYWORD_EXTRACT_PROMPT = file.read()
    
    keywords = get_gpt_answer(query, USER_KEYWORD_EXTRACT_PROMPT)
    keywords = [x.strip() for x in keywords.split("|")]
    return keywords

def extract_GPT_answer(query:str):
    initialize()
    global config 
    with open(config["query_answer_prompt_path"], 'r', encoding='utf-8') as file:
        GET_QUERY_ANSWER_PROMPT = file.read()
    
    answer = get_gpt_answer(query, GET_QUERY_ANSWER_PROMPT)
    
    return answer

def extract_query_classification(query:str):
    initialize()
    global config 
    with open(config["query_classification_prompt_path"], 'r', encoding='utf-8') as file:
        GET_QUERY_CLASSIFICATION_PROMPT = file.read()
    
    answer = get_gpt_answer(query, GET_QUERY_CLASSIFICATION_PROMPT)
    #answer의 첫 숫자를 추출해서 저장 
    query_class = re.findall(r'\d+', answer)[0]
    return query_class


def text_embed(text: str):
    global config, client
    text = text.replace("\n", " ")
    initialize()
    
    if not isinstance(text, str):
        raise ValueError("The input text must be a string.")
    
    try:
        # 임베딩 생성 시도
        return client.embeddings.create(input=[text], model=config['embedding_model']).data[0].embedding
    except Exception as e:
        print(f"Error during embedding creation: {e}")
        raise
    
# 코사인 유사도 계산 함수
def calculate_cosine_similarity(text1: str, text2: str):
    embedding1 = text_embed(text1)
    embedding2 = text_embed(text2)

    # 코사인 유사도 계산
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


# 텍스트에서 키워드 추출
def extract_keyword(text:str):
    global config
    initialize()
    keyword_model = config['model']
    prompt_path = config['query_keyword_prompt_path']

    keywords = [] 
    with open(prompt_path, 'r', encoding='utf-8') as file:
        PROMPT = file.read()
        
    completion = client.chat.completions.create(
            model=keyword_model,
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": text
                }
            ])
    
    for keyword in (completion.choices[0].message.content).split("|") : # "증권신고서 | 제출기한 연장 | 법적 책임"
        keywords.append(keyword.strip())

    keywords[0] = keywords[0].split("\n")[-1] 
    # keywords 리스트 돌면서 spacebar 기준으로 split해서 전체 저장 
    # keywords 리스트에서 각 항목을 spacebar로 split하고 전체 단어를 저장
    all_words = [word for keyword in keywords for word in keyword.split()]

    return keywords, all_words


# json 읽어서 embedding 추가 후 json 저장
def add_embedding_to_json(json_path: str, save_path: str):
    initialize()  # 전역 변수 초기화

    # json 파일 경로에 chunk 폴더 추가
    chunk_path = f"{os.path.dirname(json_path)}/chunk/{os.path.basename(json_path)}"

    # json -> chunk 변환
    data = json_chunk_split(json_path, chunk_path)

    # embedding 추가
    print("##### embedding start #####")
    print(f" - json_path : {chunk_path}")

    # subtitle이 None이 아닌 clause만 남기도록 필터링
    clauses = [clause for clause in data if clause.get('subtitle') is not None]
    
    
    for idx, clause in tqdm(enumerate(clauses), total=len(clauses)):
        #print(f" - clause index : {clause['index']}")

        # metadata에서 title을 가져와서 None이 아닌 것만 조합하여 reference 생성
        reference = ' '.join(
            clause['metadata']['title'][key]
            for key in clause['metadata']['title'] if clause['metadata']['title'][key] is not None
        )
        
        # 편, 장, 절, 관 제거 후 subtitle 추가
        reference = re.sub(r'제\d+편|제\d+장|제\d+절|제\d+관', '', reference).strip()
        
        # subtitle 추가 (이미 None이 아닌 경우만 남겨졌으므로 확인 불필요)
        reference += f" {clause['subtitle']}"
        
        reference = re.sub(r'\s+', ' ', reference).strip()

        # 텍스트 임베딩 생성 및 추가
        text = f"<{reference}>{clause['content']}"
        print(f" - text : {reference}")
        clause['embedding'] = text_embed(text)
    
    clauses = delete_deletion_node(clauses)
    # 저장 경로 디렉터리 생성
    save_dir = os.path.dirname(save_path)
    print(f"## make save dir in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON 파일로 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, ensure_ascii=False, indent=4)
        
    print(f" - save_path : {save_path}")
    print(f"##### Embedding done  #####")
    return clauses




# 여러 json file을 하나의 json file로 merge
def merge_to_json(path: str, save_path="../../results", filtering="_clause.json"):
    data = []
    # Search for files within the directory (including subdirectories)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filtering):
                print(f"Merging file: {file} | file_length: {len(file)}")
                # os.path issue
                file_path = os.path.join(root, file)
                file_path = f"{root}/{file}"
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        json_data = json.load(f)
                        data.append(json_data)  # Add data to the list
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")
    
    # Save the merged data as JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"## Data successfully saved to {save_path}")
    return data

# JSON 객체를 평탄화하는 함수
def flatten_json(nested_json, parent_key='', sep='_'):
    items = []
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# JSON 파일을 chunk로 나누어 저장 (리스트 형태로 저장)
"""def json_chunk_split(json_path: str, chunk_size: int, chunk_overlap: int, save_path: str):
    print("##### Start chunking #####")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Chunk 전 \n - len : {len(data)}\n - json_path : {json_path}\n - chunk_size : {chunk_size}\n - chunk_overlap : {chunk_overlap}")

    new_data = []
    for clause in tqdm(data):
        splitted_doc = 0
        chunks = chunk_split(clause['content'], chunk_size, chunk_overlap)
        if len(chunks) > 1:
            splitted_doc += 1
        for chunk in chunks:
            clause_copy = clause.copy()  
            clause_copy['content'] = chunk  
            new_data.append(clause_copy)  
        
    print(f"Chunk 후 \n - len : {len(new_data)}\n - save_path : {save_path}")

    # 리스트 형태로 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
        print("##### Chunking done #####")
    return new_data"""

# JSON 파일을 chunk로 나누어 저장 (리스트 형태로 저장)
def json_chunk_split(json_path: str, save_path: str):
    global config, client, dbms  # 전역 변수를 사용하기 위해 global 선언
    initialize()  # 전역 변수 초기화

    # 설정값 가져오기
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']

    print("##### Start chunking #####")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Chunk 전 \n - len : {len(data)}\n - json_path : {json_path}\n - chunk_size : {chunk_size}\n - chunk_overlap : {chunk_overlap}")

    new_data = []
    for clause in tqdm(data):
        splitted_doc = 0
        chunks = chunk_split(clause['content'], chunk_size, chunk_overlap)
        if len(chunks) > 1:
            splitted_doc += 1
        for chunk in chunks:
            clause_copy = clause.copy()  
            clause_copy['content'] = chunk  
            new_data.append(clause_copy)  

    print(f"Chunk 후 \n - len : {len(new_data)}\n - save_path : {save_path}")

    print(f"## make save dir in {'/'.join(save_path.split('/')[:-1])}")
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)

    # 리스트 형태로 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
        print("##### Chunking done #####")
        
    return new_data


# 주어진 chunk size, overlap size에 따른 recursive split
def chunk_split(text : str, chunk_size: int, chunk_overlap:int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter.split_text(text) 


# 특정 키워드를 포함하는 법 조항 찾기 -> 프린트, 반환
def search_json(json_path: str, word1: str, word2:str):
    def print_clause(clauses: list):
        for clause in clauses:
            print(f'{clause["index"]} : {clause["content"]}')
    clauses_and = []
    clauses_1 = []
    clauses_2 = []
    clauses_total = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for clause in data:
        if word1 in clause["content"] and word2 in clause["content"]:
            clauses_and.append(clause)
        if word1 in clause["content"]:
            clauses_1.append(clause)
            clauses_total.append(clause)
        if word2 in clause["content"]:
            clauses_2.append(clause)
            clauses_total.append(clause)
        
    
    print(f'## {word1} & {word2} ##')
    if len(clauses_and) > 0:
        print_clause(clauses_and)
    else:
        print("공통조항 없음")
    
    print(f'\n\n## {word1} 언급 조항 ##')
    if len(clauses_1) > 0:
        print_clause(clauses_1)
    else:
        print(f"{word1} 언급 조항 없음")
    
    print(f'/n/n## {word2} 언급 조항 ##')
    if len(clauses_2) > 0:
        print_clause(clauses_2)
    else:
        print(f"{word2} 언급 조항 없음")

    return clauses_and, clauses_1, clauses_2, clauses_total



# 특정 패턴에 해당하는 clause 내 정의 문장만 추출 (metadata 포함)
def extract_definition_sentences(json_path: str):
    SAME_PATTERN = r'이하.* 한다'   # 이하 ~~ 한다
    INCLUDE_PATTERN = r'.*포함(한다|된다|하여)' # 포함한다/된다/하여

    same_sentences = []
    include_sentences = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for clause in tqdm(data):
        splitted_sentences = split_clause(clause['content'])

        for text in splitted_sentences:
            # 정의 문장일 경우
            if re.search(SAME_PATTERN, text):
                clause_metadata = copy.deepcopy(clause)
                clause_metadata['content'] = text
                same_sentences.append(clause_metadata)

            if re.search(INCLUDE_PATTERN, text):
                clause_metadata = copy.deepcopy(clause)
                clause_metadata['content'] = text
                include_sentences.append(clause_metadata)
            
            else:
                continue
        
    return same_sentences, include_sentences



# clause 찢어서 리스트로 반환
def split_clause(text) -> list[str]:
    result = []
    buffer = ''
    stack = []  # To keep track of parentheses
    i = 0

    while i < len(text):
        c = text[i]
        if c == '(':
            stack.append('(')
            buffer += c
            i += 1
        elif c == ')':
            if stack:
                stack.pop()
            buffer += c
            i += 1
        elif c == '/n':
            # Do not split if newline is immediately before a letter
            if i + 1 < len(text) and text[i + 1].isalpha():
                buffer += c
            else:
                if buffer.strip():
                    result.append(buffer.strip())
                buffer = ''
            i += 1
        elif c == '.':
            # Check if inside parentheses or after a number
            if stack or (i > 0 and text[i - 1].isdigit()):
                buffer += c
            else:
                buffer += c
                if buffer.strip():
                    result.append(buffer.strip())
                buffer = ''
            i += 1
        else:
            buffer += c
            i += 1
    if buffer.strip():
        result.append(buffer.strip())

    return result


# GPT 배치 파서를 위한 JSONL 생성 및 메타데이터 저장
def make_triplet_jsonl(clause_list: list, data_type: str,  prompt_path: str, input_jsonl_path: str, meta_path: str, model_name="gpt-4o") -> pd.DataFrame:
    id_list = []
    metadata_list = []
    
    # 프롬프트 불러오기
    with open(prompt_path, 'r', encoding='utf-8') as file:
        PROMPT = file.read()

    with open(input_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, clause in enumerate(clause_list):
            json_line = json.dumps({"custom_id" : f"request-{idx}",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {"model": model_name, 
                                                "messages": [{"role": "system", "content": PROMPT},
                                                            {"role": "user", "content": clause['content']}],
                                                "max_tokens": 1024}
                                    }, ensure_ascii=False)
            f.write(json_line + '\n')

            id_list.append(f"request-{idx}")
            
            metadata_list.append({"law_index": f"{data_type}_{clause['index']}", # 법_제x조제x항
                                  "document_title": clause["metadata"]["document_title"],
                                  "created_date": clause["metadata"]["date"],
                                  "revise_info": clause["metadata"]["revise_info"],
                                  "source": clause["metadata"]["source"],
                                  "reference" : f"{data_type}_{' '.join(clause['metadata']['title'][x] for x in clause['metadata']['title'].keys() if clause['metadata']['title'][x] != None)}_{clause['index']}_{clause['subtitle']}" # 법_제4장 과징금 제2절 예탁관련제도 제2관 투자익명조합_제x조제x항_금융투자상품
                                  })

    
    metadata = pd.DataFrame(columns=["request_id", "metadata"])
    metadata["request_id"] = id_list
    metadata["metadata"] = metadata_list

    metadata.to_csv(meta_path, index=False)

    print(f"## jsonl file saved in {input_jsonl_path}")
    print(f"## meta file saved in {meta_path}")
    
    return


# gpt 4o api 사용한 triplet extractor (배치 단위)
def make_triplet_batch(input_jsonl_path):
    initialize()
    batch_input_file = client.files.create(
        file=open(input_jsonl_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    metadata = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "triplet extraction"
        }
    )

    return metadata


# triplet extractor 배치 취소
def cancel_triplet_batch(id_list, custom_id=None):
    initialize()

    # custom id 입력 있을 경우 -> 특정 request id만 cancel
    if custom_id != None:
        client.batches.cancel(custom_id)
    
    else:
        for request_id in id_list:
            client.batches.cancel(request_id)

    return

# batch id 받아서 status 확인
def get_batch_status(batch_id):
    initialize()
    return client.batches.retrieve(batch_id)

def get_batch_answer(output_file_id, result_path):
    initialize()
    answer = client.files.content(output_file_id).content
    with open(result_path, 'wb') as file:
        file.write(answer)
    
    print(f'## file saved in {result_path}')
    return


# batch request 받기 "(S | edge | O)"
def get_triplet(result_path:str, meta_path: str) -> list:
    """
    gpt batch request를 받아, '(S | edge | O)' format
    -> metadata와 매치하여 노드 전체 property + edge 관계를 담은 triplet list 반환
    
    output : (S_property | edge_type | O_property)
    """
    triplets = []
    exception = 0
    
    # jsonl result path
    # UTF-8 인코딩으로 파일 열기
    with open(result_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file:
            # 각 줄을 JSON 객체로 파싱
            json_obj = json.loads(line)
            data.append(json_obj)

    print(f"## {len(data)} 개의 데이터 로드됨. 데이터 예시 : {data[0]}")

    # metadata csv
    metadata_csv = pd.read_csv(meta_path)

    # 한글을 포함하는 정규표현식
    hangul_pattern = re.compile(r'[가-힣]')

    # jsonl 기준으로 순회
    # metadata와 매칭
    for request in tqdm(data):
        answer = request["response"]["body"]["choices"][0]["message"]["content"]

        # 동일한 request_id 가진 metadata 불러와서 매칭하기
        metadata = metadata_csv.loc[metadata_csv["request_id"] == request["custom_id"], "metadata"]
        metadata = ast.literal_eval(metadata.values[0])

        triplet_list = answer.split('\n')

        for triplet in triplet_list:
            triplet = triplet.strip()
            
            # triplet 공백인 경우
            if len(triplet) == 0:
                print("output format error: node was blank\n\noutput :{triplet}")
                exception += 1
                continue

            try:
                s, edge_type, o = triplet.split(' | ')
                edge_type = edge_type.strip()

                if len(edge_type) > 0:
                    # edge type, reference
                    edge = {'type' : edge_type}
                    snode = copy.deepcopy(metadata)
                    snode["keyword_name"] = ''.join([x for x in s.strip() if x != '(' and x != ')'])

                    onode = copy.deepcopy(metadata)
                    onode["keyword_name"] = ''.join([x for x in o.strip() if x != '(' and x != ')'])

                    edge['reference'] = metadata['law_index'] + '_' + metadata['reference'].split('_')[-1]
                    print('## edge : ', edge)

                    triplets.append((snode, edge, onode))


            except Exception as e:
                # 여기서 계속 output format error: not enough values to unpack (expected 3, got 1)
                print(f"output format error: {e}\n\noutput :{triplet}")
                exception += 1

    for triplet in triplets:
        if hangul_pattern.search(triplet[1]['type']):
            print(f"output format error: 한글 엣지 감지! edge_type: {triplet[1]['type']}")
            exception += 1

    # 한글이 포함되지 않은 경우만 리스트에 삽입
    triplets = [triplet for triplet in triplets if not hangul_pattern.search(triplet[1]['type'])]

    # exception 몇 개 존재하는지?
    print(f"## 예외 처리 데이터 : {exception}")

    return triplets

# JSON 파일을 Document 리스트로 로드
def load_json_as_documents(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    documents = []
    for entry in json_data:
        doc = Document(
            text=entry["content"],
            metadata={
                "index": entry["index"],
                "name": entry["subtitle"],
                "document_title": entry["metadata"]["document_title"],
                "created_date": entry["metadata"]["date"],
                "revise_info": entry["metadata"]["revise_info"],
                "source": entry["metadata"]["source"],
                "title_doc": entry["metadata"]["title"]["doc"],
                "title_chapter": entry["metadata"]["title"]["chapter"],
                "title_section": entry["metadata"]["title"]["section"],
                "title_subsection": entry["metadata"]["title"]["subsection"],
            }
        )
        documents.append(doc)
    
    return documents

# text 분할
def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    split_docs = []
    
    for doc in documents:
        # 텍스트를 분할하고 각 조각을 Document로 다시 생성
        chunks = text_splitter.split_text(doc.text)
        for chunk in chunks:
            split_doc = Document(
                text=chunk,
                metadata=doc.metadata,
                metadata_seperator=doc.metadata_seperator,
                metadata_template=doc.metadata_template,
                text_template=doc.text_template
            )
            split_docs.append(split_doc)
    
    return split_docs


# 문서 정보 추출
def extract_document_information(input_path):
    file_name = input_path.split("/")[-1]

    document_number = file_name.split("_")[0]
    law_type = file_name.split("_")[1]
    document_type = file_name.split("_")[2].split(".")[0]
    return document_number, law_type, document_type


# documents돌면서 article, clause pattern 찾기 
# def find_clause_pattern(input_path, documents):
#     no_matches = 0 
#     document_number, law_type, document_type = extract_document_information(input_path)  # 메타데이터 추출
#     current_document = f"{document_number}_{law_type}_{document_type}"

#     # 결과를 저장할 딕셔너리
#     result_dict = {
#         'meta': {
#             'document_number': document_number,
#             'law_type': law_type,
#             'document_type': document_type,
#             'current_document': current_document
#         },
#         'patterns': {}
#     }

#     for doc in documents: 
#         #print("\n", "="*30, doc.metadata['index'], "="*30)
#         text = doc.text
#         index = doc.metadata['index']
#         #print("contents:", text)

#         # 기본 패턴 | 연속 패턴 | 다중 패턴을 처리하는 정규식
#         # pattern = (
#         #     r"([가-힣\w\(\)\"「」]*)?\s*(제\d+조(?:의\d+)?(?:제\d+항)?)"  # 제~조제~항 패턴
#         #     r"|([가-힣\w\(\)\"「」]*)?\s*(제\d+항)"  # 단독 제~항 패턴
#         #     r"((?:,\s*제\d+조(?:의\d+)?(?:제\d+항)?| 및\s*제\d+조(?:의\d+)?(?:제\d+항)?|,\s*제\d+항| 및\s*제\d+항)*)"
#         # )
        
#         # # 연속 패턴 추가 (제~조부터, 제~항부터, 제~조까지 등)
#         # pattern += r"|(제\d+조(?:의\d+)?(?:제\d+항)?(?:부터|까지)|제\d+항(?:부터|까지))"

#         ### 제5-23조의2제3항 같은 경우 추가
#         pattern = (
#             r"([가-힣\w\(\)\"「」]*)?\s*(제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?)"  # 제~조제~항 패턴 수정
#             r"|([가-힣\w\(\)\"「」]*)?\s*(제\d+항)"  # 단독 제~항 패턴
#             r"((?:,\s*제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?| 및\s*제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?|,\s*제\d+항| 및\s*제\d+항)*)"
#         )

#         # 연속 패턴 추가 (제~조부터, 제~항부터, 제~조까지 등)
#         pattern += r"|(제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?(?:부터|까지)|제\d+항(?:부터|까지))"

#         # 텍스트에서 모든 매치된 패턴과 그 앞 단어 찾기
#         matches = re.findall(pattern, text)
        
#         if not matches:
#             no_matches+=1
#             continue

#         # 문서의 index를 키로 하고 텍스트와 패턴 리스트를 함께 저장
#         result_dict['patterns'][index] = {
#             'text': text,  # 텍스트를 저장
#             'matches': []  # 매칭된 패턴 리스트 저장
#         }

#         for match in matches:
#             # '제~조제~항' 패턴과 '제~항' 단독 패턴을 구분하여 처리
#             if match[1]:  # '제~조제~항' 패턴
#                 preceeding_word = match[0].strip() if match[0] else "\\n"
#                 matched_pattern = match[1].strip()
#             elif match[3]:  # 단독 '제~항' 패턴
#                 preceeding_word = match[2].strip() if match[2] else "\\n"
#                 matched_pattern = match[3].strip()
#             else:
#                 continue  # 패턴에 매칭되지 않으면 넘어감

#             # 앞 단어에서 쉼표를 보존하고, 공백만 제거하도록 수정
#             preceeding_word = preceeding_word if preceeding_word != "\\n" else ","

#             # 패턴 리스트에 추가
#             final = {
#                 'preceeding_word': preceeding_word,
#                 'matched_pattern': matched_pattern
#             }
            
#             result_dict['patterns'][index]['matches'].append(final)
#     print(f"## no matches : {no_matches}")
#     return result_dict


#refactoring 버전 
def find_clause_pattern(input_path, documents):
    # 메타데이터 추출
    document_number, law_type, document_type = extract_document_information(input_path)
    current_document = f"{document_number}_{law_type}_{document_type}"

    # 결과를 저장할 딕셔너리 초기화
    result_dict = {
        'meta': {
            'document_number': document_number,
            'law_type': law_type,
            'document_type': document_type,
            'current_document': current_document
        },
        'patterns': {}
    }

    # 패턴 정의 (정규 표현식)
    pattern = (
        r"([가-힣\w\(\)\"「」]*)?\s*(제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?)"  # 제~조제~항 패턴
        r"|([가-힣\w\(\)\"「」]*)?\s*(제\d+항)"  # 단독 제~항 패턴
        r"((?:,\s*제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?| 및\s*제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?|,\s*제\d+항| 및\s*제\d+항)*)"
        r"|(제\d+(?:-\d+)*조(?:의\d+)?(?:제\d+항)?(?:부터|까지)|제\d+항(?:부터|까지))"  # 연속 패턴 추가
    )

    # 문서들을 순회하며 패턴 매칭
    no_matches = 0
    for doc in documents:
        text = doc.text
        index = doc.metadata['index']

        # 텍스트에서 모든 매칭된 패턴과 그 앞 단어 찾기
        matches = re.findall(pattern, text)
        if not matches:
            no_matches += 1
            continue

        # 패턴이 매칭된 경우 결과에 추가
        result_dict['patterns'][index] = {
            'text': text,
            'matches': []
        }

        # 매칭된 각 패턴에 대해 처리
        for match in matches:
            if match[1]:  # '제~조제~항' 패턴
                preceeding_word = match[0].strip() if match[0] else "\n"
                matched_pattern = match[1].strip()

                # 추가적인 패턴 분리 (예: '제1항ㆍ제5항 및 제9항')
                sub_matches = re.split(r'[ㆍ및,]', matched_pattern)
                for sub_match in sub_matches:
                    sub_match = sub_match.strip()
                    if sub_match:
                        result_dict['patterns'][index]['matches'].append({
                            'preceeding_word': preceeding_word,
                            'matched_pattern': sub_match
                        })
                        # 이후 패턴의 앞 단어는 'ㆍ', '및', ',' 등을 포함하지 않도록 수정
                        preceeding_word = "ㆍ" if 'ㆍ' in matched_pattern else ("및" if '및' in matched_pattern else preceeding_word)

            elif match[3]:  # 단독 '제~항' 패턴
                preceeding_word = match[2].strip() if match[2] else "\n"
                matched_pattern = match[3].strip()

                # 추가적인 패턴 분리 (예: '제2항ㆍ제3항')
                sub_matches = re.split(r'[ㆍ및,]', matched_pattern)
                for sub_match in sub_matches:
                    sub_match = sub_match.strip()
                    if sub_match:
                        result_dict['patterns'][index]['matches'].append({
                            'preceeding_word': preceeding_word,
                            'matched_pattern': sub_match
                        })
                        # 이후 패턴의 앞 단어는 'ㆍ', '및', ',' 등을 포함하지 않도록 수정
                        preceeding_word = "ㆍ" if 'ㆍ' in matched_pattern else ("및" if '및' in matched_pattern else preceeding_word)


            # 앞 단어에서 쉼표 보존 및 공백 제거
            preceeding_word = preceeding_word if preceeding_word != "\n" else ","

    print(f"## no matches : {no_matches}")
    return result_dict




# index에서 article, clause, subarticle 추출
# input : str (text: 제~조제~항) -> output : tuple (article, subarticle, clause)
### 수정 필요
def index_to_article(text):
    # '제301조'와 같은 패턴에서 article, article_regulation, subarticle, clause 추출
    # 제2-3조와 같은 패턴에서 article과 article_regulation 추출
    article_match = re.search(r"제(\d+)(?:-(\d+))?조", text)  # 제1조 또는 제2-3조
    subarticle_match = re.search(r"의(\d+)", text)  # 제1조의1
    clause_match = re.search(r"제(\d+)항", text)  # 제1항

    if article_match:
        article = int(article_match.group(1)) if article_match.group(1) else None
        article_regulation = int(article_match.group(2)) if article_match.group(2) else None
    else:
        article = None
        article_regulation = None

    subarticle = int(subarticle_match.group(1)) if subarticle_match else None
    clause = int(clause_match.group(1)) if clause_match else None

    return article, article_regulation, subarticle, clause


# article, clause, subarticle -> index로 변환
# input : article, clause, subarticle -> output : str (text: 제~조제~항)
# regulation 추가
def article_to_index(article, clause=None, subarticle=None, regulation=None):
    if regulation:
        # 제2-2조의2
        if subarticle:
            if clause: 
                return f"제{article}-{regulation}조의{subarticle}제{clause}항"
            return f"제{article}-{regulation}조의{subarticle}"
        # 제2-2조
        else : 
            if clause:
                return f"제{article}-{regulation}조제{clause}항"
            return f"제{article}-{regulation}조"
        
    else:
        # 제2조의2
        if subarticle:
            if clause: 
                return f"제{article}조의{subarticle}제{clause}항"
            return f"제{article}조의{subarticle}"
        # 제2조
        else : 
            if clause:
                return f"제{article}조제{clause}항"
            return f"제{article}조"
        
        

# 다중항 패턴 : Multiple Pattern 
def process_multiple_pattern(first, second): #다중 패턴 
    # ex. 제30조제1항 및 제2항 
    # ex. 제3-1조의2제1항 및 제3항
    # first에 있는 조 정보(ex.제30조) 추출해서 second 항정보( ex. 2항) 연결 -> 제30조제2항
    first_article, first_regulation, first_subarticle, _ = index_to_article(first)
    _, _, _, second_clause = index_to_article(second)

    # clause
    second_index = article_to_index(first_article, second_clause, first_subarticle, first_regulation)

    return second_index


# 범위 패턴 : Range Pattern  * refactoring 필요,, 
def process_range_pattern(start_index, end_index, edges,current_index):
    #print("range pattern", start_index, end_index)
    
    # Extract article, regulation, subarticle, clause for both start and end indices
    start_article, start_regulation, start_subarticle, start_clause = index_to_article(start_index)
    end_article, end_regulation, end_subarticle, end_clause = index_to_article(end_index)
    previous_article, previous_regulation, previous_subarticle, previous_clause =index_to_article(edges[-1]['target_index'])
    current_article, current_regulation, current_subarticle, current_clause = index_to_article(current_index)

    target_list = []

    # 제~조부터 제~조까지
    if (start_article is not None) and (start_clause is None) and (end_article is not None) and (end_clause is None):
        #print(f"Processing range: 제{start_article}조부터 제{end_article}조까지")
        for article in range(start_article, end_article + 1):
            # target_list.append(f"제{article}조")
            target_list.append(article_to_index(article, None, None, start_regulation))

    # 제~항부터 제~항까지 -> article 없음
    elif (start_article is None) and (start_clause is not None) and (end_article is None) and (end_clause is not None):
        # print(f"Processing range: 제{start_clause}항부터 제{end_clause}항까지")
        # if previous_article: #이전 조항 정보가 항상 있어서 이전 조항의 정보를 가져와서 에러 생김 ex. enforcement 제58조제5항 -> 이거 왜 필요하지? 일단 주석처리 
        #     target_article = previous_article
        #     target_subarticle = previous_subarticle
        #     target_regulation = previous_regulation # regulation 추가

        target_article = current_article
        target_subarticle = current_subarticle
        target_regulation = current_regulation # regulation 추가

        for clause in range(start_clause, end_clause + 1):
           target_list.append(article_to_index(target_article, clause, target_subarticle, target_regulation))
    
    # 제~조제~항부터 제~항까지
    elif (start_article is not None) and (start_clause is not None) and (end_article is None) and (end_clause is not None):
        #print(f"Processing range: 제{start_article}조의{start_subarticle}제{start_clause}항부터 제{end_clause}항까지")
        for clause in range(start_clause, end_clause + 1):
            # target_list.append(f"제{start_article}조제{clause}항" if not start_subarticle else f"제{start_article}조의{start_subarticle}제{clause}항")
            target_list.append(article_to_index(start_article, clause, start_subarticle, start_regulation))
    
    # 제180조의2부터 제180조의5까지 (조의 범위) - clause가 없음
    elif (start_article is not None) and (start_subarticle is not None) and (end_article == start_article) and (end_subarticle is not None):
        #print(f"Processing range: 제{start_article}조의{start_subarticle}부터 제{end_subarticle}까지")
        for subarticle in range(start_subarticle, end_subarticle + 1):
            # target_list.append(f"제{start_article}조의{subarticle}")
            target_list.append(article_to_index(start_article, None, subarticle, start_regulation))
    
    else:
        logging.info("## Unhandled range pattern with variables - in process_range_pattern()")
        logging.info(f"start_article: {start_article}, start_regulation: {start_regulation}, start_subarticle: {start_subarticle}, start_clause: {start_clause}")
        logging.info(f"end_article: {end_article}, end_regulation: {end_regulation}, end_subarticle: {end_subarticle}, end_clause: {end_clause}")
        logging.info(f"previous_article: {previous_article}, previous_regulation: {previous_regulation}, previous_subarticle: {previous_subarticle}, previous_clause: {previous_clause}")
        logging.info(f"current_article: {current_article}, current_regulation: {current_regulation}, current_subarticle: {current_subarticle}, current_clause: {current_clause}")
        pass
        
    return target_list


# 찾은 패턴에서 시작 문서, 인덱스, 타겟 문서, 인덱스 추출 
# input : dict (matched_dict) -> output : list (edges)
# start_document, start_index , end_document, end_index  

def extract_refers_to_edges(matched_dict):
    #matched_dict에서 patterns의 key 값이 index가 되고, value는 리스트로 저장되어 있음 
    #리스트의 원소는 딕셔너리로 저장되어 있음
    #딕셔너리의 key는 preceeding_word와 matched_pattern
    #matched_pattern의 value를 확인해서 target_document와 target_index, target_clause 찾기 
    document_number = matched_dict['meta']['document_number']
    law_type = matched_dict['meta']['law_type']
    document_type = matched_dict['meta']['document_type']
    current_document =  matched_dict['meta']['current_document']
    edges =  []
    

    for index, information in matched_dict['patterns'].items():
        current_index = index 
        current_text = information['text']
        #print("\n", "="*30, current_index, "="*30)
        #print("contents:", current_text)
        
        current_article, current_regulation, current_subarticle, current_clause = index_to_article(current_index)
            
        #기본은 현재 문서, 현재 조항 기준 
        target_document = current_document
        target_article = current_article
        target_regulation = current_regulation
        target_subarticle = current_subarticle
        target_clause = current_clause
        

        # 탐지한 패턴 확인하면서 target_document, target_article, target_clause 추출 
        start_index = current_index 
        for pattern in information['matches']:
            ######### Range Pattern ####### 
            if any(keyword in pattern['preceeding_word'] for keyword in ["부터","까지"]): 
                #print("이전 조항 정보:", target_article, target_subarticle, target_clause)

                if ("제" in pattern["preceeding_word"]):    
                    #EX. {"preceeding_word": "제1조부터", "matched_pattern": "제3조"}
                    #EX. {"preceeding_word": "제1항부터", "matched_pattern": "제4항"}
                    start_index = pattern['preceeding_word']
                    end_index = pattern['matched_pattern']

                else : # 이전 조항 연결                         EX. {"preceeding_word": "부터", "matched_pattern": "제3조"}
                    start_index = edges[-1]['target_index']
                    end_index = pattern['matched_pattern']

                target_list = process_range_pattern(start_index, end_index, edges, current_index)
                
                for target_index in target_list: #range pattern에서 나온 target_list를 돌면서 edge 생성
                    target_article, target_regulation, target_subarticle, target_clause = index_to_article(target_index)
                    #target_article이 none이면 current_article로 대체
                    target_article = target_article if target_article else current_article
                    #target_clause가 none이면 ""로 대체 
                    target_index = article_to_index(target_article, target_clause, target_subarticle, target_regulation)

                    edge = {
                        'current_document': current_document,
                        'current_index': current_index,
                        'target_document': target_document,
                        'target_index': target_index,
                        'contents': current_text
                    }
                    #print(edge)
                    edges.append(edge)
                continue
                
            
            # matched_pattern =   pattern['matched_pattern']
            target_article, target_regulation, target_subarticle, target_clause = index_to_article(pattern['matched_pattern'])
            
            ###### 다른 문서, 다른 조 ########
            if "「" in pattern['preceeding_word'] or "」" in pattern['preceeding_word']: # 외부 법 참조 경우
                #print("다른 문서, 다른 조")
                """match = re.search(r'(.*?)」', pattern['preceeding_word'])
                if match:
                    target_document = match.group(1)"""
                
                # target document 찾기
                print(f"## 다른 문서, 다른 조의 preceeding_word : {pattern['preceeding_word']} ##")
                # 후처리에서 「」 단락 찾아오기


                target_document = "not_in_my_document"

                #print("target_document:", target_document)
                target_index = article_to_index(target_article, target_clause, target_subarticle, target_regulation) 
                    
                
            elif ("조" not in pattern['matched_pattern'] ) and ("항" in pattern['matched_pattern']):
                if (pattern['preceeding_word']=="조" ): #같은 조 라고 가져왔을 때 이전의 조 정보 가져와야 함 
                    
                    #print("같은 문서, 앞에랑 같은 조")
                    target_document = edges[-1]['target_document']
                    previous_article, previous_regulation, previous_subarticle, _ = index_to_article(edges[-1]['target_index'])
                    current_article, current_regulation, current_subarticle, current_clause = index_to_article(pattern['matched_pattern'])
                    target_index = article_to_index(previous_article, current_clause, previous_subarticle, previous_regulation)
                
                elif any(keyword in pattern['preceeding_word'] for keyword in [",", "및", "또는"]):
                    # pattern: {'preceeding_word': '및', 'matched_pattern': '제2항'}
                    #다중 패턴 -> 이전 조항 연결 
                    target_document = edges[-1]['target_document'] #이전 문서로 변경 
                    target_index = process_multiple_pattern(edges[-1]['target_index'], pattern['matched_pattern']) 

                else : 
                    # 예시 
                    # pattern: {'preceeding_word': '감독이사는', 'matched_pattern': '제2항'} pattern: {'preceeding_word': '말한다)은', 'matched_pattern': '제1항'}

                    #print("같은 문서, 같은 조")
                    target_document = current_document
                    current_article, current_regulation, current_subarticle, current_clause = index_to_article(current_index)
                    target_article, target_regulation, target_subarticle, target_clause = index_to_article(pattern['matched_pattern'])
                    target_index = article_to_index(current_article, target_clause, current_subarticle, current_regulation)

            
            ###### 같은 문서, 다른 조 ########
            #EX. {'preceeding_word': '', 'matched_pattern': '제78조제1항'}
            elif ("조" in pattern['matched_pattern'] ) and ("항" in pattern['matched_pattern']):
                #print("같은 문서, 다른 조")
                target_document = current_document
                target_article = target_article
                target_clause = target_clause
                target_regulation = target_regulation

                target_index = article_to_index(target_article, target_clause, target_subarticle, target_regulation)

                
            else : #항 정보 없이 조만 포함하는 경우
                target_index = article_to_index(target_article, target_clause, target_subarticle, target_regulation)
        


        
            if pattern['preceeding_word'][-1]=="법":
                target_document = f"{document_number}_law_{document_type}"
            elif pattern['preceeding_word'][-1]=="영":
                target_document = f"{document_number}_enforcement_{document_type}"
            elif pattern['preceeding_word'][-1]=="부":  
                target_document = f"{document_number}_order_{document_type}"
            elif pattern['preceeding_word'][-2:]=="규칙": 
                target_document = f"{document_number}_order_{document_type}"
            elif pattern['preceeding_word'][-2:]=="동법":
                target_document = f"{document_number}_law_{document_type}"

            else : #현재 문서 그대로 찍으면 됨 
                logging.info(f"## Unhandled preceeding word {pattern['preceeding_word']} - in extract_refers_to_edges()")
            

            edge = {
                'current_document': current_document,
                'current_index': current_index, 
                'target_document': target_document,
                'target_index': target_index,
                'contents': current_text
            }

            #print(edge,"\n")
            edges.append(edge)

    return edges




# input : str (input_path) -> output : list (triplets)
# triplet list 생성
### 외부 데이터 매칭

def create_refers_to_triplets_list(input_path):
    document_number, law_type, document_type = extract_document_information(input_path)
    print("====input_path:", input_path,"======")
    print("## document_number:", document_number, "law_type:", law_type, "document_type:", document_type)
    
    documents = load_json_as_documents(input_path)
    print(f"## Number of documents loaded: {len(documents)}")
    
    matched_dict = find_clause_pattern(input_path, documents)
    print(f"## Number of matched patterns: {len(matched_dict['patterns'])}")

    #matched_pattern.json 형태 저장 
    with open(f"../data/graph/clause/matched_pattern/{document_number}_{law_type}_{document_type}_matched_pattern.json", 'w', encoding='utf-8') as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=4)
    triplets = extract_refers_to_edges(matched_dict)
    print("## Finish creating triplets ")
    print("## Number of triplets created: ", len(triplets))
    print("## Sample triplets: ", triplets[:1])
    
    return triplets


def reranking(candidate_list, answer_embedding):
    print("######## Reranking ########")
    # Compute cosine similarity for each candidate and store it in the dictionary
    for item in candidate_list:
        item['answer_similarity'] = cosine_similarity([answer_embedding], [item['embedding']])[0][0]
        item['mean_similarity'] = (item['answer_similarity'] + item['similarity'] )/2
        
    # Sort candidates by weighted similarity in descending order
    top_k = sorted(candidate_list, key=lambda x: -x['mean_similarity'])

    # Print out the reranked list with similarity scores
    for idx, item in enumerate(top_k):
        print(f"Top {idx + 1}: {item['labels']} {item['index']} {item['name']}, mean_similarity: {item['mean_similarity']:.2f}, query_similarity: {item['similarity']:.2f}, answer_similarity: {item['answer_similarity']:.2f}, ")
        print(f"query_similarity : {item['similarity']:.2f} Text: {item['text']}\n\n")

    return top_k




# 답변을 주제에 따라 찢어서 분류
def split_answer_text(answer: str):
    """
    Splits the given answer text into a structured format.
    주어진 답변 텍스트를 구조화된 형식으로 분리합니다.
    Args:
        answer (str): The answer text to be split. 분리할 답변 텍스트.
    Returns:
        list: A list of dictionaries, each containing the following keys:
            - number (int): 번호
            - subject (str): 제목
            - description (str): 설명
            - law (dict): 관련 법 정보
                - document_title (str): 법률 이름
                - subtitle (str): 조항 정보
    Example:
        >>> answer_text = "1. **Title**: - **설명**: Description - **관련 법 조항**: - **법률 이름**: Law Name - **조항 정보**: Clause Info"
        >>> split_answer_text(answer_text)
        [{'number': 1, 'subject': 'Title', 'description': 'Description', 'law': {'document_title': 'Law Name', 'subtitle': 'Clause Info'}}]
    """
    
    split_answer = []
    item_pattern = re.compile(
        r'(\d+)\.\s\*\*(.+?)\*\*:\s*'                # 번호와 제목
        r'-\s\*\*설명\*\*:\s*(.+?)\s*'                # 설명
        r'-\s\*\*관련 법 조항\*\*:\s*'                # 관련 법 조항
        r'-\s\*\*법률 이름\*\*:\s*(.+?)\s*'           # 법률 이름
        r'-\s\*\*조항 정보\*\*:\s*(.+?)(?=\n\d+\.\s|\Z)', # 조항 정보 (다음 번호 또는 문서 끝까지)
        re.DOTALL
    )

    for match in item_pattern.finditer(answer):
        number = int(match.group(1).strip())
        subject = match.group(2).strip()
        description = match.group(3).strip()
        document_title = match.group(4).strip()
        subtitle = match.group(5).strip()

        item_dict = {
            'number': number,
            'subject': subject,
            'description': description,
            'law': {
                'document_title': document_title,
                'subtitle': subtitle
            },
            'subtitle_embedding':text_embed(subtitle),
            'subject_embedding':text_embed(subject + description)
        }
        split_answer.append(item_dict)
    return split_answer

def multiAgent(query, split_answer):
    """"
    - subtitle vs node의 subtitle
    - subject+ description vs node의 content
    dbms에서 node의 subtitle_embedding vs split_answer.law.subtitle을 embedding 한 결과 비교 
    dbms에서 node의 embedding vs split_answer.subject+split_answer.description을 embedding 한 결과 비교 
    """
    for idx , item in enumerate(split_answer): 
        answer = item[i]
        traverse_graph_with_hops(query, answer, hop)
        initial_nodes = dbms.get_neighbors_to_query(split_answer.subject_embedding, split_answer.subtitle_embedding)


# input : str (query) -> output : list (candidate_list)
def traverse_graph_with_hops(query:str, hop:int):
    """
    query : 질문 str
    system_prompt : get_query_answer.txt
    hop : int (탐색할 hop 수)
    """
    # logger = PrintLogger()
    global dbms, config, client 
    initialize()
    
    system_prompt = config["query_answer_prompt_path"]
    answer = extract_GPT_answer(query)
    query_embedding = text_embed(query)
    answer_embedding = text_embed(answer)
    
    keywords , all_words = extract_keyword(query+"\n"+answer)
    print("Keywords:", len(keywords))
    print(keywords)
    print("All words:", len(all_words))
    print(all_words)
    all_neighbors = {}  # node_id를 키로 하는 딕셔너리로 중복 방지
    
    print("########### hop: 0 ##########")
    initial_nodes = dbms.get_neighbors_to_query(query_embedding)
    visited_nodes = set()

    for node in initial_nodes: #초기 top k 
        node_id = node.get('node_id')# node 객체의 properties에서 law_index 추출
        index = node.get('index') ## None 수정 요망 
        labels = node.get('labels')
        similarity = node.get('similarity')
    
        all_neighbors[node_id] = node  # 초기 노드 저장
        print(f"labels: {labels}, index : {index}, similarity : {similarity}, node_id : {node_id}")
        visited_nodes.add(node_id)
    print(f"all_neighbors for hop 0: {len(all_neighbors)}")

    
    current_hop_nodes = initial_nodes
    for current_hop in range(1, hop): # hop 만큼 반복
        print(f"\n\n########### hop: {current_hop} ##########")
        next_hop_nodes = []

        for node in current_hop_nodes:  # 현재 hop 노드 순회 
            node_id = node.get('node_id')
            index = node.get('index')
            similarity = node.get('similarity')
            node_id = node.get('node_id')
            labels = node.get('labels')
            name = node.get('name')
            

            print(f"\n{labels} {index} {name}, similarity : {similarity}, node_id : {node_id}")
            print(f"text : {node.get('text')}")
            
                
            if node_id : 
                neighbors = dbms.get_refers_to_neighbors(node_id)  # 이웃 노드 list 
                num_neighbors = len(neighbors)
                if node_id not in all_neighbors:
                    node['num_neighbors'] = num_neighbors
                    all_neighbors[node_id] = node  # 중복 방지

                for neighbor in neighbors: # 현재 node의 이웃 노드 순회 
                    if isinstance(neighbor, set): # set -> dict  
                        print(f"Unexpected set: {neighbor}")  
                        neighbor = dict(neighbor)
                    
                    neighbor_id = neighbor['node_id']  
                    neighbor_index = neighbor['index']
                    neighbor_labels = neighbor['labels']
                    neighbor_name =  neighbor['name']

                    if neighbor_id not in visited_nodes:
                        neighbor['hop'] = current_hop

                        # Calculate similarity if not already set
                        if 'similarity' not in neighbor:
                            neighbor_embedding = np.array(neighbor['embedding'])
                            neighbor['similarity'] = cosine_similarity([query_embedding], [neighbor_embedding])[0][0]
                            
                        if neighbor_id not in next_hop_nodes:
                            next_hop_nodes.append(neighbor)
                        print(f" O Append :  {neighbor_labels} {neighbor_index} {neighbor_name}")
                    else : 
                        print(f" X Not Append(Already Visited) : {neighbor_labels} {neighbor_index} {neighbor_name}")
                
                print(f"to next_hop_nodes {num_neighbors}")
                visited_nodes.add(node_id)
        
        print(f"next_hop_nodes length for hop {current_hop}: {len(next_hop_nodes)}")
        print(f"visited nodes length: {len(visited_nodes)}")

        print("## reranking the next_hop_nodes based on similarity and selecting top k")
        """current_hop_nodes = sorted(
            [neighbor for neighbor in next_hop_nodes if neighbor['node_id'] not in visited_nodes],
            key=lambda x: -x['similarity']
        )[:config['top_k']]"""
        filtering = [
                neighbor for neighbor in next_hop_nodes
                if neighbor['node_id'] not in visited_nodes
                and any(keyword in neighbor[field] for field in ['text', 'document_title', 'name'] for keyword in keywords)
            ]
        print("## filtering based on keywords - length : ", len(filtering))
        current_hop_nodes = sorted(filtering,key=lambda x: -x['similarity'])[:config['top_k']]

        print("## next hop nodes")
        print( [(i.get('labels')+" "+ i.get('index')) for i in current_hop_nodes])

        # 다음 노드에서 확인하지 않는 노드는 visited 에서 제외 (중복 방지)
        #visited_nodes = visited_nodes.intersection(next_hop_nodes)

    all_neighbors = sorted(all_neighbors.values(), key=lambda x: (x['hop'], -x['similarity']))
    print("\n\n#### final results #### ")

    for idx, neighbor in enumerate(all_neighbors):
        print(f"Top : {idx} ====== {neighbor.get('labels')}  {neighbor.get('index')} ======== ")
        print(f"similarity : {neighbor.get('similarity'):.2f}, hop : {neighbor.get('hop')}")
        print(f"text: {neighbor.get('text')}, node_id : {neighbor.get('node_id')}\n\n\n")
    print("\n\n\n")

    top_k = reranking(all_neighbors, answer_embedding)
    print("## Finish reranking")


    return 


# 폴더 내 모든 .json 파일에 대해 refers_to_triplets 생성
def get_refers_to_triplet_json(json_folder_path: str):
    # 데이터별 폴더 구분을 위함
    # 01 (자본시장법), 02(증권 규정) ...
    print('## folder path : ', json_folder_path)

    folder_num = os.path.basename(os.path.normpath(json_folder_path))
    print('## folder num : ', folder_num)

    # refers_to_triplets 리스트 초기화
    refers_to_triplets = []

    if len(os.listdir(json_folder_path)) == 0:
        raise ValueError("## No file exists!")

    for file_name in os.listdir(json_folder_path):
        if file_name.endswith(".json"):
            # os path issue
            # input_path = os.path.join(json_folder_path, file_name)
            input_path = f"{json_folder_path}/{file_name}"
            refers_to_triplets.append(create_refers_to_triplets_list(input_path))

    print("## Finish creating all triplets")

    #refers_to_triplets를 json 형태로 저장 
    os.makedirs(f"../data/graph/clause/edge_triplet/{folder_num}/", exist_ok=True)
    
    output_path = f"../data/graph/clause/edge_triplet/{folder_num}/{folder_num}_refers_to_triplets.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refers_to_triplets, f, ensure_ascii=False, indent=4)

    print(f"## Finish saving all triplets in {output_path}")

    return


# 삭제 노드 제거 
def delete_deletion_node(data):
    print("############### Delete Deletion Node ###############")
    print(f"### Before : {len(data)}")

    filtered_data = [
    clause for clause in data
    if not re.fullmatch(
        r'(삭제\s*<\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.>)|'
        r'(삭제\s*<\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.>\s*\n\[본조신설\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\])',
        clause['content'].strip(),
        re.DOTALL
    )
    ]

    print(f"### After : {len(filtered_data)}")
    return filtered_data
    
