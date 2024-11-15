import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from itertools import combinations

import GraphDB.utils as utils
from GraphDB.LegalGraphDB import LegalGraphDB


class ChatModel:
    def __init__(self):
        with open(f'configs/config.json', 'r') as f:
            self.config = json.load(f)

        self.dbms = LegalGraphDB(auradb=False, config=self.config, json_path="../graph/clause/title_embedding/01/01_law_main.json")
        print('## Loading DB...##')

    def get_documents_answer(self, query: str, top_k: int):
        relevant_documents = ""; total_documents = ""

        print(f'## Mode : {self.config["rag"]}')
        print(f"## We will retrieve top-{top_k} relevant documents")
        print('## user query : ', query)

        query_emb = utils.text_embed(query)
        
        query_keywords = utils.extract_keyword(query)
        print(f"## Query keywords : {query_keywords}")

        keyword_comb = list(combinations(query_keywords[0], 2))
        keyword_comb_text = [' '.join(pair) for pair in keyword_comb]
        print(f"## Query keywords combinations : {len(keyword_comb_text)}, {keyword_comb_text}")

        if self.config['rag'] == "Vector":
            results= self.dbms.get_top_k_clause_nodes(query_emb, k = top_k)
            print(f"## Retrieved {len(results)} Documents")
            for result in results:
                relevant_documents += f'##<{result["node_labels"]} - {result["index"]}>, {result["reference"]}, {result["name"]}\n{result["text"]}\n' + "="*50
                total_document = f'##<{result["node_labels"]} - {result["index"]}>, {result["name"]}\n{"="*50}\n'
                # print(idx, total_documents)
                total_documents += total_document

        elif self.config['rag'] == "Graph":
            final_start_nodes = [] 
            for i in range(len(keyword_comb_text)):
                # print("========================="*2)
                node_id_list = [] 
                # print(keyword_comb_text[i])
                keyword_emb = utils.text_embed(keyword_comb_text[i])
                results= self.dbms.get_top_k_clause_nodes(keyword_emb, k = 1)

                for result in results:
                    node_id = int(result['node_id'].split(':')[-1])
                    node_id_list.append(self.dbms.get_top_k_in_same_index(query_emb, node_id, k=1))
                sorted_data = sorted(node_id_list, key=lambda x: x[0]['similarity'], reverse=True)
                # print(sorted_data[0])
                final_start_nodes.append(sorted_data[0][0]['node_id'])

            # distinct 구분 
            final_start_nodes = list(set(final_start_nodes))
            results = self.dbms.find_path_node(final_start_nodes)

            print("## Relevant Documents\n\n")

            for idx, result in enumerate(results):
                if idx < top_k:
                    relevant_document = f'##<{result["labels"]} - {result["index"]}>, {result["reference"]}, {result["name"]}\n{result["text"]}\n{"="*50}\n'
                    print(idx, relevant_document)
                    relevant_documents += relevant_document

                total_document = f'##<{result["labels"]} - {result["index"]}>, {result["name"]}\n{"="*50}\n'
                # print(idx, total_documents)
                total_documents += total_document

        else:
            raise ValueError(f"RAG Mode : Graph | Vector, But current setting is {self.config['rag']} - check your config file")

        SYSTEM_PROMPT = f"""당신은 자본시장법 전문가로서 사용자의 질문에 답해야 합니다.
        **User Query**와 관련 있는 법률 조항 내 **Relevant Documents**가 주어집니다. **Relevant Documents**를 참고하여 적절하고 정확한 답변을 생성해주세요.  
        <Relevant Documents>
        {relevant_documents}
        </Relevant Documents>
        """
        answer = utils.get_gpt_answer(query, SYSTEM_PROMPT)
        
        
        total_documents = f"## Total Retrived : {len(results)}\n\n" + total_documents
        return answer, total_documents



