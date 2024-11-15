from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm
"""
자본시장법 GraphDB 구축을 위한 DB class

[클래스 메서드]
<공통>
- __init__(self) : neo4j (auraDB) 초기화 & 연결 테스트
- close(self) : 드라이버 연결 종료

- add_element_to_node(self, node_id, property_name, new_element): 노드에 속성(property_name)에 요소(new_element)를 추가하고 중복을 제거

- delete_relationship(self, node_A_id, node_B_id, edge_type): 두 노드(node_A_id, node_B_id) 사이의 특정 관계(edge_type)를 삭제
- delete_node(self, node_id): 노드(node_id)를 삭제
- delete_all_relationship(self, database=None): 모든 관계를 삭제
- delete_all_nodes(self, database=None): 모든 노드와 관계를 삭제

- get_node_embedding(self, node_element_id: str): 노드(node_element_id: str)의 임베딩 벡터를 반환


<Keyword Graph>
- get_keyword_node_id(self, keyword_name: str) -> str: 키워드 이름(keyword_name: str)을 기준으로 키워드 노드 ID(str)를 반환
- get_keyword_relation_id(self, node_id: str) -> list: 키워드 노드 ID(node_id: str)에 대한 관계 ID의 리스트를 반환
- get_keyword_subgraphs(self, keyword_list: list): 키워드 리스트(keyword_list)에 대한 서브그래프를 반환
- get_all_triplet(self, query: str, answer: list): 모든 triplet을 반환

- create_keyword_node(self, node_property: dict) -> str: 키워드 노드 속성(node_property: dict)을 기반으로 노드를 생성하고 노드 ID(str)를 반환
- create_keyword_relationship(self, triplet: list): 키워드 관계(triplet: list)를 생성
- create_keyword_graph(self, triplets: list): 키워드 그래프(triplets: list)를 생성

- delete_keyword_subgraph(self, triplets: list): 키워드 서브그래프(triplets: list)를 삭제


<Clause Graph>
- get_clause_node_id(self, node_type: str, law_index: str) -> str: 노드 타입(node_type: str)과 법률 인덱스(law_index: str)를 기반으로 조항 노드 ID(str)를 반환
- get_all_clause_nodes(self): 모든 조항 노드의 임베딩 데이터와 노드 ID를 가져옵니다.
- get_refers_to_neighbors(self, node_id: str): 'refers_to' 관계로 연결된 이웃 노드를 가져옵니다.
- get_neighbors_to_node(self, node_element_id: str): 유사한 노드를 정렬하여 반환
- get_neighbors_to_query(self, query_embedding): 쿼리 임베딩과 유사한 노드를 반환

- create_clause_node(self, node_type: str, node_property: dict, embedding=False) -> str: 조항 노드(node_type: str, node_property: dict)를 생성하고 노드 ID(str)를 반환
- create_clause_relationship(self, triplet: dict): 조항 관계(triplet: dict)를 생성

- create_clause_graph(self, triplet: list, node_A_type: str, node_B_type: str): 조항 그래프를 생성
- create_clause_graph_from_json(self, data): JSON 데이터(data)를 기반으로 조항 그래프를 생성


- traverse_clause_graph_with_hop(self, node_id: str, hop=1): 지정된 홉(hop) 수로 조항 그래프를 탐색 -> utils

"""


class LegalGraphDB:
    # neo4j (auraDB) 초기화 & 연결 테스트
    def __init__(self, config: dict, json_path: str, auradb=False):
        self.config = config

        # 양방향 타입 / 단방향 엣지 타입 정의
        self.bidirectional_relationship = ["SAME_AS"]
        self.unidirectional_relationship = ["INCLUDED_IN", "refers_to"]


        # title type 정의
        self.clause_type = "Clause_" + json_path.split('/')[-1].split('.')[0]
        self.title_prefix = "Title_" + json_path.split('/')[-1].split('.')[0] + "_"

        self.index_dict = {"편" : self.title_prefix+"doc", "장" : self.title_prefix+"chapter", 
              "절" : self.title_prefix+"section", "관" : self.title_prefix+"subsection", 
              "조" : self.clause_type, "항" : self.clause_type}

        print("## Settings ##")
        print("## clause type : ", self.clause_type)
        print('## title_prefix : ', self.title_prefix)

        # 프로젝트 루트 디렉토리에서 .env 파일 로드
        load_dotenv()

        if auradb == False:
            # Neo4j 서버에 연결
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

        
        else:
            # auradb 서버에 연결
            uri = os.getenv("AURADB_URI")
            user = os.getenv("AURADB_USERNAME")
            password = os.getenv("AURADB_PASSWORD")

        # URI와 사용자 정보 설정
        AUTH = (user, password)

        self.driver = GraphDatabase.driver(uri, auth=AUTH)
        self.driver.verify_connectivity()
        self.database = config["database"]
        print("## 서버와의 연결이 성공적으로 확인되었습니다.")


    # 드라이버 연결 종료
    def close(self):
        self.driver.close()
        print("## Neo4j driver 종료")


    
    # 노드에 요소(new_element)를 추가하고 중복을 제거
    def add_element_to_node(self, node_id, property_name, new_element):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (n)
                WHERE ID(n) = $node_id
                WITH n, coalesce(n.""" + property_name + """, []) + [$new_element] AS items
                UNWIND items AS item
                WITH n, COLLECT(DISTINCT item) AS unique_items
                SET n.""" + property_name + """ = unique_items
                """,
                node_id=node_id,
                new_element=new_element
            )

    # 두 노드(node_A_id, node_B_id) 사이의 특정 관계(edge_type)를 삭제
    def delete_relationship(self, node_A_id, node_B_id, edge_type):
        with self.driver.session(database=self.database) as session:
            session.run(
                # "MATCH ()-[r]->() DELETE r"
                f"""
                MATCH (a), (b)
                WHERE id(a) = $from_id AND id(b) = $to_id
                MATCH (a)-[r:{edge_type}]->(b) DELETE r""",
                from_id=node_A_id,
                to_id=node_B_id
            )

        print(f"## {node_A_id} - {node_B_id} 사이의 {edge_type} Edge 삭제 완료!")


    # 노드(node_id)를 삭제
    def delete_node(self, node_id):
        with self.driver.session(database=self.database) as session:
            # 양방향 엣지 삭제
            session.run(
                f"""MATCH (n)-[r]->(m)
                WHERE id(n) = $node_id
                DELETE r""",
                node_id = node_id)
            
            session.run(
                f"""MATCH (m)-[r]->(n)
                WHERE id(n) = $node_id
                DELETE r""",
                node_id = node_id)

            # 노드 삭제 
            session.run(f"""MATCH (n)
                        WHERE id(n) = $node_id
                        DELETE n""",
                        node_id = node_id)
    
    # edge 모두 삭제
    def delete_all_relationship(self):
        with self.driver.session(database=self.database) as session:
            pre_delete_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"## 삭제 이전 전체 relationship 개수: {pre_delete_count}")

            # 모든 노드에 대해 content가 다른 노드의 index를 포함하고 있는지 확인하여 엣지를 생성
            session.run(
                "MATCH ()-[r]->() DELETE r"
            )
            # 삭제 이후 relationship 개수 출력
            post_delete_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"## 삭제 이후 전체 relationship 개수: {post_delete_count}")

        print("## Edge 삭제 완료!")


    # 특정 edge 삭제
    def delete_specific_relationship(self, edge_type: str):
        with self.driver.session(database=self.database) as session:
            pre_delete_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"## 삭제 이전 전체 relationship 개수: {pre_delete_count}")

            session.run(
                """
                MATCH ()-[r]->()
                WHERE type(r) = $edge_type
                DELETE r
                """,
                edge_type=edge_type
            )

             # 삭제 이후 relationship 개수 출력
            post_delete_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            print(f"## 삭제 이후 전체 relationship 개수: {post_delete_count}")
            
        print(f"## {edge_type} Edge 삭제 완료!")
        
    # edge & node 모두 삭제
    def delete_all_nodes(self):
        with self.driver.session(database=self.database) as session:
            # 삭제 이전 relationship 및 node 개수 출력
            pre_relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            pre_node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"## 삭제 이전 전체 relationship 개수: {pre_relationship_count}")
            print(f"## 삭제 이전 전체 node 개수: {pre_node_count}")

            # 모든 관계 삭제
            session.run("MATCH ()-[r]->() DELETE r")
            # 모든 노드 삭제
            session.run("MATCH (n) DELETE n")

            # 삭제 이후 relationship 및 node 개수 출력
            post_relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            post_node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            
            print(f"## 삭제 이후 전체 relationship 개수: {post_relationship_count}")
            print(f"## 삭제 이후 전체 node 개수: {post_node_count}")

        print("## 모든 노드 삭제 완료!")
    

    # 특정 node 삭제
    def delete_specific_node(self, node_type: str):
        with self.driver.session(database=self.database) as session:
            # 삭제 이전 relationship 및 node 개수 출력
            pre_relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            pre_node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"## 삭제 이전 전체 relationship 개수: {pre_relationship_count}")
            print(f"## 삭제 이전 전체 node 개수: {pre_node_count}")

            query = f"MATCH (n:{node_type}) DETACH DELETE n"
            session.run(query)

            # 삭제 이후 relationship 및 node 개수 출력
            post_relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            post_node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            print(f"## 삭제 이후 전체 relationship 개수: {post_relationship_count}")
            print(f"## 삭제 이후 전체 node 개수: {post_node_count}")
        
        print(f"## {node_type} Node 삭제 완료!")



    # 노드(node_element_id)의 임베딩 벡터를 반환
    def get_node_embedding(self, node_element_id):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n)
                WHERE elementId(n) = $node_element_id
                RETURN n.embedding AS embedding
                """,
                node_element_id=node_element_id
            )
            record = result.single()
            return record["embedding"] if record else None
    
    def get_child_node_id(self, parent_node_id: str, node_type: str, properties: dict) -> str:
        with self.driver.session(database=self.database) as session:
            # 속성을 쿼리에 맞게 변환
            props = ' AND '.join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"""
            MATCH (p)-[:hierarchy]->(n:{node_type})
            WHERE ID(p) = $parent_node_id AND {props}
            RETURN n LIMIT 1
            """

            params = properties.copy()
            params['parent_node_id'] = parent_node_id

            result = session.run(query, **params)
            record = result.single()
            if record:
                node_id = record['n'].id
                return node_id
            else:
                return None

    def get_node_id(self, node_type: str, properties: dict) -> str:
        with self.driver.session(database=self.database) as session:
            # 속성을 쿼리에 맞게 변환
            props = ' AND '.join([f"n.{k} = ${k}" for k in properties.keys()])
            query = f"MATCH (n:{node_type}) WHERE {props} RETURN n LIMIT 1"

            result = session.run(query, **properties)
            record = result.single()
            if record:
                node_id = record['n'].id
                return node_id
            else:
                return None


            
    # keyword_name: str을 기준으로 키워드 노드 ID(를 반환
    def get_keyword_node_id(self, keyword_name: str):
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""MATCH (n:keyword {{name: $keyword_name}})
                RETURN n
                LIMIT 1""",
                keyword_name=keyword_name
            )
            record = result.single()
            if record:
                node_id = record['n'].id  # record['n']에서 id를 가져옴
                return node_id
            else:
                return None

    # keyword relation_id list 반환
    def get_keyword_relation_id(self, node_id: str) -> list:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""MATCH (n)-[r]-(m)
                    WHERE ID(n) = $node_id
                    RETURN ID(r) AS relationship_id""",
                node_id=node_id
            )
            relationship_ids = [record["relationship_id"] for record in result]
        
        return relationship_ids
    
    # 들어온 keyword list와 연결 관계가 있는 모든 node를 가져오고 element list를 반환
    def get_keyword_subgraphs(self, keyword_list)-> list:
        triplets = []
        triplet_ids = set()  # To ensure uniqueness
    
        with self.driver.session(database=self.database) as session:
            for keyword_name in keyword_list:
                    query = """
                        MATCH (n:keyword {name: $name})-[r]-(m)
                        RETURN n, r, m
                        """
                    subgraph_result = session.run(query, name=keyword_name)

                    for record in subgraph_result:
                        n = record["n"]
                        r = record["r"]
                        m = record["m"]
                        # Use IDs to ensure uniqueness since Node and Relationship objects are not hashable
                        triplet_id = (n.element_id, r.element_id, m.element_id)
                        if triplet_id not in triplet_ids:
                            triplet_ids.add(triplet_id)
                            triplets.append((n, r, m))
        
        return list(set(triplets))
    
    # query와 연결된 element list를 토대로 원하는 format의 triplet text로 반환
    def get_all_triplet(self, query, answer) -> list:
        answer_triplets = self.get_keyword_subgraphs(answer)
        important_triplets = []
        # Output the results
        print('## User query : ', query)
        print('Keyword : ', answer)
        # print()
        # print("Triplets:")
        for n, r, m in answer_triplets:
            # print(f"({n['name']} -[{r.type}]-> {m['name']}) IN {r['reference']}")
            important_triplets.append(f"({n['name']} -[{r.type}]-> {m['name']}) IN {r['reference']}")
        
        return list(set(important_triplets))
    
    # property 기반 keyword node 생성 -> node id 반환
    def create_node(self, node_property:dict):
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(
                    f"""
                    MERGE (n:{node_property['node_type']} {{
                        name: $name,
                        law_index: $law_index
                    }})
                    RETURN n LIMIT 1
                    """,
                    name = node_property['name'],
                    law_index=node_property['law_index']
                )
                record = result.single()

                if record:
                    node_id = record['n'].id  # record['n']에서 id를 가져옴
                    return node_id
                
                else:
                    raise ValueError("Node not created!")
                
            except Exception as e:
                raise ValueError(f"Error creating keyword node: {e}")
           


    # property 기반 keyword node 생성 -> node id 반환
    def create_keyword_node(self, node_property:dict):
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(
                    f"""
                    CREATE (n:keyword {{
                        name: $keyword_name,
                        law_index: $law_index,
                        document_title: $document_title,
                        created_date: $created_date,
                        revise_info: $revise_info,
                        source: $source,
                        reference: $reference
                    }})
                    WITH n
                    MATCH (n:keyword {{name: $keyword_name}}) RETURN n LIMIT 1
                    """,
                    keyword_name = node_property['keyword_name'],
                    law_index=node_property['law_index'],
                    document_title=node_property['document_title'],
                    created_date=node_property['created_date'],
                    revise_info=node_property['revise_info'],
                    source=node_property['source'],
                    reference=node_property['reference'],
                )
                record = result.single()

                if record:
                    node_id = record['n'].id  # record['n']에서 id를 가져옴
                    return node_id
                
                else:
                    raise ValueError("Node not created!")
                
            except Exception as e:
                raise ValueError(f"Error creating keyword node: {e}")
            

    # keyword relationship 생성
    def create_keyword_relationship(self, triplet: list): 
            from_keyword, from_id = triplet[0]['keyword_name'], triplet[0]['id']
            edge_type = triplet[1]['type']
            to_keyword, to_id = triplet[2]['keyword_name'], triplet[2]['id']

            # 알파벳과 언더스코어만 허용
            if not re.match(r'^[A-Za-z_]+$', edge_type):
                print('## edge_type : ', edge_type)
                # 허용되지 않는 문자는 제거
                edge_type = re.sub(r'[^A-Za-z_]', '', edge_type)
            
            with self.driver.session(database=self.database) as session:
                if edge_type in self.bidirectional_relationship:
                    query = f"""
                        MATCH (a), (b)
                        WHERE ID(a) = $from_id AND ID(b) = $to_id
                        MERGE (a)-[r1:{edge_type}]->(b)
                        MERGE (b)-[r2:{edge_type}]->(a)
                        SET r1.reference = $edge_reference, r2.reference = $edge_reference
                        """
                elif edge_type in self.unidirectional_relationship:
                    query = f"""
                        MATCH (a), (b)
                        WHERE ID(a) = $from_id AND ID(b) = $to_id
                        MERGE (a)-[r:{edge_type}]->(b)
                        SET r.reference = $edge_reference
                        """

                else:
                    print(f"## Undefined edge type! : {edge_type}")
                    print(f'## removed triplet : {from_keyword} - {edge_type} - {to_keyword}')
                    return
                
                
                session.run(
                    query,
                    from_id=from_id,
                    to_id=to_id,
                    edge_reference=triplet[1]['reference']
                )


# node id 바탕으로 relationship 생성
    def create_relationship(self, edge_type, first_node_id, second_node_id):
        with self.driver.session(database=self.database) as session:
            query = f"""
                MATCH (a), (b)
                WHERE ID(a) = $from_id AND ID(b) = $to_id
                MERGE (a)-[r:{edge_type}]->(b)
                """
            
            session.run(
                query,
                from_id=first_node_id,
                to_id=second_node_id
            )




    # keyword graph 생성
    def create_keyword_graph(self, triplets: list):
        for triplet in triplets:
            node_A_property = triplet[0]; edge_property = triplet[1]; node_B_property = triplet[2]
            
            node_A_id = self.get_keyword_node_id(node_A_property['keyword_name'])
            node_B_id = self.get_keyword_node_id(node_B_property['keyword_name'])

            new_triplet = [{'keyword_name' : node_A_property['keyword_name']}, edge_property, {'keyword_name' : node_B_property['keyword_name']}]

            # 없는 경우 노드 생성
            if node_A_id == None:
                node_A_id = self.create_keyword_node(node_A_property)
                new_triplet[0]['id'] = node_A_id
            
            # 있는 경우 law_index, reference 추가
            else:
                self.add_element_to_node(node_A_id, "law_index", node_A_property['law_index'])
                self.add_element_to_node(node_A_id, "reference", node_A_property['reference'])
                new_triplet[0]['id'] = node_A_id


            # 없는 경우 노드 생성
            if node_B_id == None:
                node_B_id = self.create_keyword_node(node_B_property)
                new_triplet[2]['id'] = node_B_id

            # 있는 경우 law_index, reference 추가
            else:
                self.add_element_to_node(node_B_id, "law_index", node_B_property['law_index'])
                self.add_element_to_node(node_B_id, "reference", node_B_property['reference'])
                new_triplet[2]['id'] = node_B_id

            # 같은 경우 생성 X (loop edge 생성 방지)
            if node_A_id != node_B_id:
                # 여기서까지는 제대로 전달됨
                self.create_keyword_relationship(new_triplet)
    
    
    # keyword subgraph를 triplets 입력 받아 삭제
    def delete_keyword_subgraph(self, triplets: list):
        for triplet in triplets:
            node_A_property = triplet[0]; node_B_property = triplet[2]
            
            node_A_id = self.get_keyword_node_id(node_A_property['keyword_name'])
            node_B_id = self.get_keyword_node_id(node_B_property['keyword_name'])

            # 관계 삭제
            self.delete_relationship(node_A_id, node_B_id, "INCLUDED_IN")
            self.delete_relationship(node_A_id, node_B_id, "SAME_AS")

            # 연결된 관계 없을 경우, 노드도 함깨 삭제
            if len(self.get_node_relationships(node_A_id)) == 0:
                self.delete_node(node_A_id)
            
            if len(self.get_node_relationships(node_B_id)) == 0:
                self.delete_node(node_B_id)
            


    # clause node id 반환 - node_type, law_index를 기준으로 동일 node 판단
    def get_clause_node_id(self, node_type:str, law_index: str) -> str:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{node_type} {{law_index: $law_index}}) RETURN n",
                law_index=law_index
            )

            record = result.single()
            if record:
                node_id = id(record)
                return node_id
            else:
                return None


    # embedding 데이터와 노드 ID, emb vector 가져오기
    def get_all_clause_nodes(self):
        # Neo4j에서 임베딩 벡터 가져오기
         with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.type STARTS WITH 'Clause_'
                RETURN id(n) AS nodeId, n.embedding AS embedding
                """
            )

            return [(record["index"], record["embedding"]) for record in result]

    # query와 연결된 이웃 노드들 가져오기 
    def get_refers_to_neighbors(self, node_id):
        print("## get refers_to neighbors for node_id: ", node_id)
        neighbors = []

        query = """
                MATCH (n)-[:refers_to]->(m)
                WHERE elementId(n) = $node_id
                RETURN m
                """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    node = record['m']
                    neighbor = {
                        'node_id':  node.element_id ,
                        'index': node.get('law_index'),
                        'labels': list(node.labels)[0] if node.labels else "Unknown",
                        'text': node.get('text'),
                        'embedding': node.get('embedding'),
                        'document_title' : node.get('document_title'), 
                        'name' : node.get('name')
                    }
                    neighbors.append(neighbor)
        except Exception as e:
            print(f"No neighbors {e}")
            return neighbors

        return neighbors

        
    # node_id와 이웃 노드 탐색해서 유사한 노드 list 정렬 반환 
    def get_neighbors_to_node(self, node_element_id):
        """주어진 노드의 refers_to로 연결된 노드 중 유사도가 높은 순으로 정렬 반환"""
        retrieved_nodes = []
        top_k = self.config.get("top_k", 5)

        query_embedding = self.get_node_embedding(node_element_id)
        if query_embedding is None:
            print(f"No embedding found for node: {node_element_id}")
            return []

        query = """
        MATCH (n)-[:refers_to]-(m)
        WHERE elementId(n) = $node_element_id
        WITH m, gds.similarity.cosine(m.embedding, $query_embedding) AS similarity
        ORDER BY similarity DESC
        RETURN m, similarity
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query, 
                node_element_id=node_element_id, 
                query_embedding=query_embedding, 
            )
            for record in result:
                node = record['m']
                similarity = record['similarity']
                labels = list(node.labels)[0] if node.labels else "Unknown"
                retrieved_nodes.append({
                    'node_id': node.element_id,
                    'index': node.get('law_index'),
                    'document_title': node.get('document_title'),
                    'name': node.get('name'),
                    'labels': labels,
                    'similarity': similarity,
                    'node': node
                })

        return retrieved_nodes

    def find_path_node(self, start_nodes):
        retrieved_nodes = []

        # 주어진 노드들로부터의 경로와 경로에 포함된 모든 노드 탐색
        query = """
        MATCH (n)
        WHERE id(n) IN $list
        WITH collect(n) AS nodes
        UNWIND nodes AS startNode
        UNWIND nodes AS endNode
        WITH startNode, endNode
        WHERE id(startNode) <> id(endNode)
        MATCH p = shortestPath((startNode)-[*]-(endNode))
        UNWIND nodes(p) AS individualNode
        WITH DISTINCT individualNode
        WHERE any(label IN labels(individualNode) WHERE label =~ 'Clause.*')
        RETURN individualNode

        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                list = start_nodes
            )
            
            for idx, record in enumerate(result):
                node = record['individualNode']
                node_id = node.element_id
                node_reference = node.get('reference')
                node_name = node.get('name')
                node_text = node.get('text')
                node_index = node.get('law_index')
                labels = list(node.labels)[0] if len(node.labels) > 0 else "Unknown"
                retrieved_nodes.append({
                    'node_id': node_id,
                    'index': node_index,
                    'labels': labels,
                    'text':node_text,
                    'reference': node_reference,
                    'name': node_name,
                })
                # print(f"##<{node_index}>, {node_reference}, {node_name}")
                # print(node_text)
                # print("="*50)
        return retrieved_nodes

    # query와 유사한 이웃 노드 가져오기 -> subject_embedding과 subtitle_embedding 의 평균 유사도로 정렬 
    def get_neighbors_to_query_subject_subtitle_mean(self, subject_embedding, subtitle_embedding):
        retrieved_nodes = [] 
        top_k = self.config.get("top_k", 5)

        query = f"""
        WITH $subject_embedding AS subject_embedding, $subtitle_embedding AS subtitle_embedding
        MATCH (n)
        WHERE ANY(label IN labels(n) WHERE label STARTS WITH 'Clause')
        WITH n,
            gds.similarity.cosine(n.embedding, subject_embedding) AS subject_similarity,
            gds.similarity.cosine(n.subtitle_embedding, subtitle_embedding) AS subtitle_similarity
        WITH n, (subject_similarity + subtitle_similarity ) / 2 AS similarity
        ORDER BY similarity DESC
        LIMIT {top_k}
        RETURN n, similarity
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, subject_embedding=subject_embedding, subtitle_embedding=subtitle_embedding)  
            for record in result:
                node = record['n']
                node_id = node.element_id
                similarity = record['similarity']
                labels = list(node.labels)[0] if len(node.labels) > 0 else "Unknown"
                retrieved_nodes.append({
                    'node_id': node_id,
                    'index': node.get('law_index'),
                    'labels': labels,
                    'text': node.get('text'),
                    'similarity': similarity,
                    'embedding': node.get('embedding'),
                    'document_title': node.get('document_title'),
                    'name': node.get('name'),
                    'node': node, 
                    'hop': 0
                })
        return retrieved_nodes

    def get_top_k_clause_nodes(self, query_emb, k=3):
        with self.driver.session(database=self.database) as session:
            # Cypher 쿼리 실행 및 반환된 노드 수 출력
            response = session.run(
                """
                MATCH (n)
                WHERE ANY(label IN labels(n) WHERE label STARTS WITH 'Clause')
                WITH n,
                    gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $R
                RETURN n, labels(n) AS node_labels, similarity
                """,
                query_embedding=query_emb,
                R=100
            )
            
            # 반환된 노드 수 확인
            response_list = list(response)  # 쿼리 결과를 리스트로 변환
            #print(f"필터링 전 노드 개수 : {len(response_list)}")
            seen_clauses = set()  # 조와 항 중복 확인을 위한 세트
            results = []

            # Step 2: 조 정보 중복 확인 및 필터링
            for record in response_list:
                node = record["n"]
                law_index = node.get("law_index")
                
                # law_index에서 "제~조" 부분만 추출 (예: "제1조" 등)
                clause_match = re.search(r"(제\d+조)", law_index)
                clause = clause_match.group(1) if clause_match else law_index  # 조 정보 추출 실패 시 전체 사용
                
                if clause in seen_clauses:
                    continue  # 동일한 조 정보가 이미 있는 경우 건너뛰기
                else:
                    # 결과 리스트에 추가할 노드 정보
                    results.append(
                        {
                            "node_id": node.element_id,  
                            "index": law_index,
                            "reference": node.get("reference"),
                            "name": node.get("name"),
                            "similarity": round(record["similarity"], 2),
                            "text": node.get("text"),
                            "document_title": node.get("document_title"),
                            "node_labels": record["node_labels"]
                        }
                    )
                    
                    seen_clauses.add(clause)  # 새로운 조 정보를 추가하여 중복 방지

            # 최종 필터링 후 결과 수 확인
            #print(f"필터링 후 최종 노드 개수: {len(results)}")
            # for result in results[:k]:
            #     print(f'##<{result["index"]}>, {result["reference"]}, {result["name"]}')
            #     print(result["text"])
            #     print("="*50)
            return results[:k]

    # input : query_emb, 탐색하고자하는 노드의 id 
    # task : 
    # query_emb와 노드의 id의 정보를 갖고옴 -> document_title이 같으면서 law_index의 "제~조"가 같은 노드 갖고 옴 
    # "제~조"노드가 N개 있으면 N개와 query_emb와의 유사도 비교 
    # 정렬해서 top_k개 반환 
    def get_top_k_in_same_index(self, query_emb, node_id, k=2):
        with self.driver.session(database=self.database) as session:
            
            # Get the node information for the given node_id
            node_info = session.run(
                """
                MATCH (n)
                WHERE ID(n) = $node_id
                RETURN n.document_title AS document_title, n.law_index AS law_index
                """,
                node_id=node_id
            ).single()

            if not node_info:
                print(f"No node found with ID: {node_id}")
                return []

            document_title = node_info["document_title"]
            law_index = node_info["law_index"]

            # Extract the "제~조" part from law_index
            clause_match = re.search(r"(제\d+조)", law_index)
            clause = clause_match.group(1) if clause_match else law_index

            # Find nodes with the same document_title and clause, and filter by label starting with "Clause"
            similar_nodes = session.run(
                """
                MATCH (n)
                WHERE n.document_title = $document_title 
                AND n.law_index CONTAINS $clause
                AND any(label IN labels(n) WHERE label STARTS WITH 'Clause')
                RETURN ID(n) AS node_id, n.law_index AS index, n.name AS name, 
                    gds.similarity.cosine(n.embedding, $query_embedding) AS similarity, 
                    n.text AS text, n.document_title AS document_title, 
                    labels(n) AS node_labels
                ORDER BY similarity DESC
                LIMIT $top_k
                """,
                document_title=document_title,
                clause=clause,
                query_embedding=query_emb,
                top_k=k
            ).data()
            
            # Format the results
            results = [
                {
                    "node_id": record["node_id"],
                    "index": record["index"],
                    "name": record["name"],
                    "similarity": round(record["similarity"], 2),
                    "text": record["text"],
                    "document_title": record["document_title"],
                    "node_labels": record["node_labels"]
                }
                for record in similar_nodes
            ]

            return results


            

    # 조항 노드(node_type: str, node_property: dict)를 생성하고 노드 ID(str)를 반환
    def create_clause_node(self, node_type, node_property: dict,  embedding=False) -> str:
        with self.driver.session(database=self.database) as session:
            try:
                if embedding:
                    result = session.run(
                        f"""
                        CREATE (n:{node_type} {{
                            law_index: $index,
                            name: $subtitle,
                            document_title: $document_title,
                            created_date: $date,
                            revise_info: $revise_info,
                            source: $source,
                            text: $content,
                            embedding: $embedding,
                            reference: $reference
                        }})
                        RETURN id(n)
                        """,
                        index=node_property['index'],
                        subtitle=node_property['subtitle'],
                        document_title=node_property['metadata']['document_title'],
                        date=node_property['metadata']['date'],
                        revise_info=node_property['metadata']['revise_info'],
                        source=node_property['metadata']['source'],
                        content=node_property['content'],
                        embedding= node_property['embedding'],
                        reference=' '.join(node_property['metadata']['title'][x] for x in node_property['metadata']['title'].keys() if node_property['metadata']['title'][x] is not None)
                    )
                else:
                    result = session.run(
                        f"""
                        CREATE (n:{node_type} {{
                            law_index: $index,
                            name: $subtitle,
                            document_title: $document_title,
                            created_date: $date,
                            revise_info: $revise_info,
                            source: $source,
                            text: $content,
                            reference: $reference
                        }})
                        RETURN ID(n)
                        """,
                        index=node_property['index'],
                        subtitle=node_property['subtitle'],
                        document_title=node_property['metadata']['document_title'],
                        date=node_property['metadata']['date'],
                        revise_info=node_property['metadata']['revise_info'],
                        source=node_property['metadata']['source'],
                        content=node_property['content'],
                        #embedding= node_property['embedding'],
                        reference=' '.join(node_property['metadata']['title'][x] for x in node_property['metadata']['title'].keys() if node_property['metadata']['title'][x] is not None)
                    )

                    record = result.single()
                    if record:
                        node_id = record["ID(n)"]
                        return node_id
                    else:
                        raise ValueError("## Node not created! No record found after CREATE statement.")
            except Exception as e:
                return None



    # Relation 생성 메서드
    def create_clause_relationship(self, triplet: dict):
        # 필요한 값들이 None이 아닌지 확인
        start_document = f"Clause_{triplet.get('current_document')}"
        start_index = triplet.get('current_index')
        target_document = f"Clause_{triplet.get('target_document')}"
        target_index = triplet.get('target_index')

        # 'not_in_my_document'와 같은 유효하지 않은 라벨일 경우 스킵
        if 'not_in_my_document' in [triplet.get('current_document'), triplet.get('target_document')]:
            return

        try:
            with self.driver.session(database=self.database) as session:
                # 정확히 일치하는 타겟 노드가 존재하는지 확인
                target_exists = session.run(
                    f"""
                    MATCH (b:{target_document} {{law_index: $target_index}})
                    RETURN COUNT(b) > 0 AS exists
                    """,
                    target_index=target_index
                ).single()["exists"]

                if not target_exists:
                    # 정확한 타겟 노드가 없을 때, target_index를 포함하는 노드를 모두 찾음
                    potential_targets = session.run(
                        f"""
                        MATCH (b:{target_document})
                        WHERE b.law_index CONTAINS $target_index
                        RETURN b.law_index AS found_index
                        """,
                        target_index=target_index
                    ).data()

                    if not potential_targets:
                        # 타겟 노드도 없고, 부분 일치하는 노드도 없으면 스킵
                        return

                    # 여러 노드에 대해 관계 생성
                    for target in potential_targets:
                        target_index = target["found_index"]
                        session.run(
                            f"""
                            MATCH (a:{start_document} {{law_index: $start_index}})
                            MATCH (b:{target_document} {{law_index: $target_index}})
                            WITH a, b
                            MERGE (a)-[:refers_to]->(b)
                            """,
                            start_index=start_index,
                            target_index=target_index
                        )
                else:
                    # 타겟 노드가 존재할 때 관계를 생성
                    session.run(
                        f"""
                        MATCH (a:{start_document} {{law_index: $start_index}})
                        MATCH (b:{target_document} {{law_index: $target_index}})
                        WITH a, b
                        MERGE (a)-[:refers_to]->(b)
                        """,
                        start_index=start_index,
                        target_index=target_index
                    )
        except Exception as e:
            # 예외가 발생해도 무시
            print(f"Error creating relationship: {e}")
            pass
   


    # clause graph 생성
    def create_clause_graph(self, triplet: list, node_A_type, node_B_type):
        node_A_property = triplet[0]
        edge_type = triplet[1]
        node_B_property = triplet[2]
        
        node_A_id = self.get_clause_node_id(node_A_property['index'])
        node_B_id = self.get_clause_node_id(node_B_property['index'])

        if node_A_id == None:
            triplet[0] = self.create_clause_node(node_A_property)
        else:
            triplet[0] = node_A_id

        if node_B_id == None:
            triplet[2] = self.create_clause_node(node_B_property)
        else:
            triplet[2] = node_B_id

        new_triplet = [(node_A_id, node_A_type), edge_type, (node_B_id, node_B_type)]

        # triplet = [(node_id, node1 type), edge type, ( node_id, node2 type,)]
        self.create_keyword_relationship(new_triplet)


    def create_clause_graph_from_json(self, data: list[dict]):
        for triplet in data:
            self.create_clause_node(triplet)
  


    # node_type, index, title
    def get_node_info(self, text:str) -> list:
        parts = text.split(' ', 1) # doc, 제10편, 벌칙 | clause, 제224조제2항, 벌칙
        
        if len(parts) == 2:
            index, title = parts
        else:
            index = parts[0]
            title = ''.join(parts[1:])

        if index[-1] in self.index_dict.keys():
            return self.index_dict[index[-1]], index, title

        # 제11조의2 같은 경우 - 모두 항으로 취급
        # Clause_01_law_main, 제1조1항, 벌칙
        else:
            if text == self.title_prefix:
                return "root", None, text
            else:
                print(f"Exception case in {text}")
                return self.clause_type, index, title
        
    ##### 1트 #####
    # def create_hierarchy_graph(self, paths):
    #     for path in paths:
    #         with self.driver.session(database=self.database) as session:
    #             current_node = None
    #             for idx, element in enumerate(path):
    #                 node_type, law_index, name = self.get_node_info(element)

    #                 if node_type == "조" or node_type == "항":
                        
    #                 else:
    #                     if current_node is None:
    #                         # First element, root node
    #                         result = session.run(
    #                             f"""
    #                             MERGE (n:{node_type} {{name: $name}})
    #                             RETURN n
    #                             """, name=name)
    #                         current_node = result.single()[0]
    #                     else:
    #                         # Merge the child node under current node
    #                         result = session.run(
    #                             f"""
    #                             MATCH (parent)
    #                             WHERE id(parent) = $parent_id
    #                             MERGE (parent)-[:hierarchy]->(child:{node_type} {{
    #                                 name: $name,
    #                                 law_index: $law_index
    #                             }})
    #                             RETURN child
    #                             """,
    #                             parent_id=current_node.id,
    #                             name=name,
    #                             law_index=law_index
    #                         )
    #                         current_node = result.single()[0]


    ##### 2트 #####
    """
    ["제7편 거래소", "제1장 총칙"], ["제5편 집합투자지구", "제1장 총칙"] 이 들어올 경우, 상위 노드인 "제7편 거래소", "제5편 집합투자지구"가 다르기 때문에 각각 다른 "제1장 총칙"노드가 생성되어야 함에도 불구하고, 같은 노드를 가리키는 문제가 발생합니다.
    """
    # def create_hierarchy_graph(self, paths):
    #     for path in paths:
    #         with self.driver.session(database=self.database) as session:
    #             current_node = None
    #             for idx, element in enumerate(path):
    #                 node_type, law_index, name = self.get_node_info(element)

    #                 if node_type == "조" or node_type == "항":
    #                     # Build the reference string excluding the first element
    #                     reference = ' '.join(path[1:-1])

    #                     # Find the existing node in Neo4j
    #                     result = session.run(
    #                         """
    #                         MATCH (n)
    #                         WHERE n.law_index = $law_index AND n.name = $name AND n.reference = $reference
    #                         RETURN n
    #                         """,
    #                         law_index=law_index,
    #                         name=name,
    #                         reference=reference
    #                     )

    #                     node = result.single()
    #                     if node:
    #                         existing_node = node[0]
    #                         # Create hierarchy relationship with previous node
    #                         session.run(
    #                             """
    #                             MATCH (parent) WHERE id(parent) = $parent_id
    #                             MATCH (child) WHERE id(child) = $child_id
    #                             MERGE (parent)-[:hierarchy]->(child)
    #                             """,
    #                             parent_id=current_node.id,
    #                             child_id=existing_node.id
    #                         )
    #                         current_node = existing_node
    #                     else:
    #                         print(f"No existing node found for law_index {law_index}, name {name}, reference {reference}")
    #                 else:
    #                     # Merge the node to avoid duplicates
    #                     if current_node is None:
    #                         # First element, root node
    #                         result = session.run(
    #                             f"""
    #                             MERGE (n:{node_type} {{name: $name}})
    #                             RETURN n
    #                             """,
    #                             name=name
    #                         )
    #                         current_node = result.single()[0]
    #                     else:
    #                         # Merge the child node under current node
    #                         if law_index:
    #                             result = session.run(
    #                                 f"""
    #                                 MATCH (parent) WHERE id(parent) = $parent_id
    #                                 MERGE (child:{node_type} {{
    #                                     name: $name,
    #                                     law_index: $law_index
    #                                 }})
    #                                 MERGE (parent)-[:hierarchy]->(child)
    #                                 RETURN child
    #                                 """,
    #                                 parent_id=current_node.id,
    #                                 name=name,
    #                                 law_index=law_index
    #                             )
    #                         else:
    #                             result = session.run(
    #                                 f"""
    #                                 MATCH (parent) WHERE id(parent) = $parent_id
    #                                 MERGE (child:{node_type} {{
    #                                     name: $name
    #                                 }})
    #                                 MERGE (parent)-[:hierarchy]->(child)
    #                                 RETURN child
    #                                 """,
    #                                 parent_id=current_node.id,
    #                                 name=name
    #                             )
    #                         current_node = result.single()[0]
    

    # #### 3트 ####
    """
    full name 중복 저장되어 이상한 노드 생김
    e.g. law_index=제249조의22제7항
    name=기업재무안정 사모집합투자기구 등에 대한 특례
    full_name=	Title_01_law_main_ 집합투자기구 사모집합투자기구 등에 대한 특례 기관전용 사모집합투자기구 등 투자익명조합 기업재무안정 사모집합투자기구 등에 대한 특례
    """
    # def create_hierarchy_graph(self, paths):
    #     for path in paths:
    #         with self.driver.session(database=self.database) as session:
    #             current_node = None
    #             full_name = ''
    #             for idx, element in enumerate(path):
    #                 node_type, law_index, name = self.get_node_info(element)

    #                 if node_type == "조" or node_type == "항":
    #                     # Build the reference string excluding the first element
    #                     reference = ' '.join(path[1:-1])

    #                     # Find the existing node in Neo4j
    #                     result = session.run(
    #                         f"""
    #                         MATCH (n:{self.clause_type})
    #                         WHERE {{n.law_index = $law_index AND n.name = $name AND n.reference = $reference}}
    #                         RETURN n
    #                         """,
    #                         law_index=law_index,
    #                         name=name,
    #                         reference=reference
    #                     )

    #                     node = result.single()
    #                     if node:
    #                         existing_node = node[0]
    #                         # Create hierarchy relationship with previous node
    #                         session.run(
    #                             """
    #                             MATCH (parent) WHERE id(parent) = $parent_id
    #                             MATCH (child) WHERE id(child) = $child_id
    #                             MERGE (parent)-[:hierarchy]->(child)
    #                             """,
    #                             parent_id=current_node.id,
    #                             child_id=existing_node.id
    #                         )
    #                         current_node = existing_node
    #                     else:
    #                         print(f"No existing node found for law_index {law_index}, name {name}, reference {reference}")
    #                 else:
    #                     # Build the full_name for uniqueness
    #                     if current_node is None:
    #                         # First element, root node
    #                         full_name = name
    #                     else:
    #                         full_name = current_node['full_name'] + ' ' + name

    #                     # Merge the node to avoid duplicates under the same path
    #                     if current_node is None:
    #                         # First element, root node
    #                         result = session.run(
    #                             f"""
    #                             MERGE (n:{node_type} {{
    #                                 name: $name,
    #                                 full_name: $full_name
    #                             }})
    #                             RETURN n
    #                             """,
    #                             name=name,
    #                             full_name=full_name
    #                         )
    #                         current_node = result.single()[0]
    #                     else:
    #                         # Merge the child node under current node with full_name
    #                         if law_index:
    #                             result = session.run(
    #                                 f"""
    #                                 MATCH (parent) WHERE id(parent) = $parent_id
    #                                 MERGE (child:{node_type} {{
    #                                     name: $name,
    #                                     law_index: $law_index,
    #                                     full_name: $full_name
    #                                 }})
    #                                 MERGE (parent)-[:hierarchy]->(child)
    #                                 RETURN child
    #                                 """,
    #                                 parent_id=current_node.id,
    #                                 name=name,
    #                                 law_index=law_index,
    #                                 full_name=full_name
    #                             )
    #                         else:
    #                             result = session.run(
    #                                 f"""
    #                                 MATCH (parent) WHERE id(parent) = $parent_id
    #                                 MERGE (child:{node_type} {{
    #                                     name: $name,
    #                                     full_name: $full_name
    #                                 }})
    #                                 MERGE (parent)-[:hierarchy]->(child)
    #                                 RETURN child
    #                                 """,
    #                                 parent_id=current_node.id,
    #                                 name=name,
    #                                 full_name=full_name
    #                             )
    #                         current_node = result.single()[0]

    ##### 4트 #####
    """
    <개정 2021. 4. 20.> 이게 reference에 껴있어서 그랬음.. -> postprocess 완료
    """
    def create_hierarchy_graph(self, paths):
        for path in paths:
            with self.driver.session(database=self.database) as session:
                current_node = None
                for idx, element in enumerate(path): #['Title_01_law_main_', '제10편 벌칙', '제4장 과징금', '제2절 예탁관련제도', '제2관 투자익명조합', '제449조제3항 과태료']
                    node_type, law_index, name = self.get_node_info(element)
                    
                    # Build the full_name from the path
                    full_name = ' '.join(path[:idx])

                    print(f"## node_type : {node_type}, law_index : {law_index}, name : {name}, full_name : {full_name}")

                    if node_type == self.clause_type:
                        print('## clause node detected - reference : ', full_name)

                        # Find the existing node in Neo4j
                        result = session.run(
                            f"""
                            MATCH (n:{node_type})
                            WHERE n.law_index = $law_index
                            AND n.name = $name
                            AND '{self.title_prefix} ' + n.reference = $reference
                            RETURN n
                            """,
                            law_index=law_index,
                            name=name,
                            reference=full_name
                        )

                        # node = result.single()
                        nodes = [record['n'] for record in result]
                        if nodes:
                            parent_node = current_node  # Fix the parent_node as current_node
                            for existing_node in nodes:
                                if parent_node:
                                    # Create hierarchy relationship with previous node
                                    session.run(
                                        """
                                        MATCH (parent) WHERE id(parent) = $parent_id
                                        MATCH (child) WHERE id(child) = $child_id
                                        MERGE (parent)-[:hierarchy]->(child)
                                        """,
                                        parent_id=parent_node.id,
                                        child_id=existing_node.id
                                    )
                                else:
                                    # Handle the case where there is no parent_node
                                    print(f"## There was no parent node in {existing_node.id}")
                                    pass
                            # Optionally update current_node for the next iteration
                            current_node = existing_node

                        else:
                            print(f"No existing node found for law_index {law_index}, name {name}, reference {full_name}")

                    # 편장절관은 single로
                    else:
                        # Merge the node to avoid duplicates under the same path
                        if current_node is None:
                            # First element, root node
                            result = session.run(
                                f"""
                                MERGE (n:{node_type} {{
                                    name: $name,
                                    full_name: $full_name
                                }})
                                RETURN n
                                """,
                                name=name,
                                full_name=full_name
                            )
                            current_node = result.single()[0]
                        else:
                            # Merge the child node under current node with full_name
                            if law_index:
                                result = session.run(
                                    f"""
                                    MATCH (parent) WHERE id(parent) = $parent_id
                                    MERGE (child:{node_type} {{
                                        name: $name,
                                        law_index: $law_index,
                                        full_name: $full_name
                                    }})
                                    MERGE (parent)-[:hierarchy]->(child)
                                    RETURN child
                                    """,
                                    parent_id=current_node.id,
                                    name=name,
                                    law_index=law_index,
                                    full_name=full_name
                                )
                            else:
                                result = session.run(
                                    f"""
                                    MATCH (parent) WHERE id(parent) = $parent_id
                                    MERGE (child:{node_type} {{
                                        name: $name,
                                        full_name: $full_name
                                    }})
                                    MERGE (parent)-[:hierarchy]->(child)
                                    RETURN child
                                    """,
                                    parent_id=current_node.id,
                                    name=name,
                                    full_name=full_name
                                )
                            current_node = result.single()[0]
