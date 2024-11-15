from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import re

class LegalGraphRAGDB:
    # neo4j (auraDB) 초기화 & 연결 테스트
    def __init__(self, config: dict, auradb=True):
        self.config = config

        # 양방향 타입 / 단방향 엣지 타입 정의
        self.bidirectional_relationship = ["SAME_AS"]
        self.unidirectional_relationship = ["INCLUDED_IN", "refers_to"]

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

    
    
    # 구조적 노드를 생성하는 함수
    def create_structural_node(self, node_type, node_property: dict, clause_property: dict = None) -> str:
        with self.driver.session(database=self.database) as session:
            try:
                # 중복 노드 방지
                existing_node = session.run(
                    f"""
                    MATCH (n:{node_type} {{law_index: $index, name: $name}})
                    RETURN ID(n) AS node_id
                    """,
                    name=node_property['name'],
                    index=node_property['index']
                ).single()

                if existing_node:
                    return existing_node['node_id']

                # 조항 정보가 있는 경우 추가
                additional_properties = ""
                parameters = {
                    "name": node_property['name'],
                    "index": node_property['index']
                }
                if clause_property:
                    additional_properties = ", text: $text, document_title: $document_title, created_date: $date, revise_info: $revise_info, source: $source, reference: $reference"
                    parameters.update({
                        "text": clause_property['content'],
                        "document_title": clause_property['metadata']['document_title'],
                        "date": clause_property['metadata']['date'],
                        "revise_info": clause_property['metadata']['revise_info'],
                        "source": clause_property['metadata']['source'],
                        "reference": ' '.join(clause_property['metadata']['title'][x] for x in clause_property['metadata']['title'].keys() if clause_property['metadata']['title'][x] is not None)
                    })

                # 노드 생성 쿼리 실행
                result = session.run(
                    f"""
                    CREATE (n:{node_type} {{
                        name: $name,
                        law_index: $index
                        {additional_properties}
                    }})
                    RETURN ID(n)
                    """,
                    **parameters
                )

                record = result.single()
                if record:
                    node_id = record["ID(n)"]
                    return node_id
                else:
                    raise ValueError("## Node not created! No record found after CREATE statement.")
            except Exception as e:
                print(e)
                print("## Node 생성 중 오류 발생")
                return None

    # 구조적 노드를 생성하는 함수
    def create_structural_node(self, node_type, node_property: dict, clause_property: dict = None) -> str:
        with self.driver.session(database=self.database) as session:
            try:
                # 중복 노드 방지
                existing_node = session.run(
                    f"""
                    MATCH (n:{node_type} {{law_index: $index, name: $name}})
                    RETURN ID(n) AS node_id
                    """,
                    name=node_property['name'],
                    index=node_property['index']
                ).single()

                if existing_node:
                    return existing_node['node_id']

                # 조항 정보가 있는 경우 추가
                additional_properties = ""
                parameters = {
                    "name": node_property['name'],
                    "index": node_property['index']
                }
                if clause_property:
                    additional_properties = ", text: $text, document_title: $document_title, created_date: $date, revise_info: $revise_info, source: $source, reference: $reference"
                    parameters.update({
                        "text": clause_property['content'],
                        "document_title": clause_property['metadata']['document_title'],
                        "date": clause_property['metadata']['date'],
                        "revise_info": clause_property['metadata']['revise_info'],
                        "source": clause_property['metadata']['source'],
                        "reference": ' '.join(clause_property['metadata']['title'][x] for x in clause_property['metadata']['title'].keys() if clause_property['metadata']['title'][x] is not None)
                    })

                # 노드 생성 쿼리 실행
                result = session.run(
                    f"""
                    CREATE (n:{node_type} {{
                        name: $name,
                        law_index: $index
                        {additional_properties}
                    }})
                    RETURN ID(n)
                    """,
                    **parameters
                )

                record = result.single()
                if record:
                    node_id = record["ID(n)"]
                    return node_id
                else:
                    raise ValueError("## Node not created! No record found after CREATE statement.")
            except Exception as e:
                print(e)
                print("## Node 생성 중 오류 발생")
                return None


    
    # query_embedding과 유사한 노드 반환
    def get_top_k_clause_nodes(self, query_emb, k=5):
        with self.driver.session(database=self.database) as session:
            response = session.run(
                """
                MATCH (n:Article)
                WITH n, 
                    gds.alpha.similarity.cosine(n.subtitle_embedding, $query_embedding) AS subtitle_similarity,
                    gds.alpha.similarity.cosine(n.content_embedding, $query_embedding) AS content_similarity
                ORDER BY subtitle_similarity DESC, content_similarity DESC
                RETURN n, subtitle_similarity, content_similarity
                """,
                query_embedding=query_emb
            )
            
            seen_embeddings = set()  # 중복 임베딩 확인 
            results = []  
            
            for record in response:
                subtitle_embedding = tuple(record["n"]["subtitle_embedding"])  # 노드의 title_embedding을 튜플로 변환하여 중복 확인
                if subtitle_embedding not in seen_embeddings: 
                    results.append(
                        {
                            "node_id": record["n"].elementId,
                            "index": record["n"].get("law_index"),
                            "title_similarity": record["title_similarity"],
                            "content_similarity": record["content_similarity"],
                            "document_title": record["n"].get("document_title"),
                            "name": record["n"].get("name"),
                            "node": record["n"]  # 노드의 모든 정보를 포함
                        }
                    )
                    seen_embeddings.add(subtitle_embedding)
            
            return results[:k]
        
    # 관계를 생성하는 함수
    def create_structural_relationship(self, from_node, to_node, relationship_type="hierarchy"):
        with self.driver.session(database=self.database) as session:
            try:
                # 상위 노드가 정확히 일치하는지 확인하고 연결할 노드가 중복되지 않도록 함
                result = session.run(
                    f"""
                    MATCH (a {{law_index: $from_index, name: $from_name, node_type: $from_node_type}})
                    MATCH (b {{law_index: $to_index, name: $to_name, node_type: $to_node_type}})
                    RETURN COUNT(b) > 0 AS exists
                    """,
                    from_index=from_node['index'],
                    from_name=from_node['name'],
                    from_node_type=from_node['node_type'],
                    to_index=to_node['index'],
                    to_name=to_node['name'],
                    to_node_type=to_node['node_type']
                ).single()["exists"]

                if result:
                    # 중복 관계 방지
                    session.run(
                        f"""
                        MATCH (a {{law_index: $from_index, name: $from_name, node_type: $from_node_type}}), (b {{law_index: $to_index, name: $to_name, node_type: $to_node_type}})
                        MERGE (a)-[:{relationship_type}]->(b)
                        """,
                        from_index=from_node['index'],
                        from_name=from_node['name'],
                        from_node_type=from_node['node_type'],
                        to_index=to_node['index'],
                        to_name=to_node['name'],
                        to_node_type=to_node['node_type']
                    )
            except Exception as e:
                print(f"Error creating relationship: {e}")
                print("## 관계 생성 중 오류 발생")

    # 조항 노드(node_type: str, node_property: dict)를 생성하고 노드 ID(str)를 반환
    def create_clause_node(self, node_type, node_property: dict, embedding=False) -> str:
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
                print(e)
                print("## Node 생성 중 오류 발생")
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
