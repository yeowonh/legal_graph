from graphrag.neo4j import GraphDatabase

# Neo4j 데이터베이스에 연결하는 클래스
class Neo4jDatabase:

    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    # 데이터를 적재하는 함수
    def create_data(self, node_1, node_2, edge):
        with self.driver.session(database=self.database) as session:
            results = session.write_transaction(self._create_and_return_data, node_1, node_2, edge)
            for record in results:
                print(f"Created relationship: {record['r']} between {record['n1']} and {record['n2']}")

    @staticmethod
    def _create_and_return_data(tx, node_1, node_2, edge):
        # 기업명과 기타 개념을 구분하여 노드 타입 설정

        node_1_type = "Company" if "(company)" in node_1 else "Concept"
        node_2_type = "Company" if "(company)" in node_2 else "Concept"

        node_1 = node_1.replace("(company)", "").strip()
        node_2 = node_2.replace("(company)", "").strip()
        
        query = (
            f"MERGE (n1:{node_1_type} {{name: $node_1}}) "
            f"MERGE (n2:{node_2_type} {{name: $node_2}}) "
            "MERGE (n1)-[r:RELATED_TO {description: $edge}]->(n2) "
            "RETURN n1, n2, r"
        )
        result = tx.run(query, node_1=node_1, node_2=node_2, edge=edge)
        return [record for record in result]  # Collecting results as a list to avoid ResultConsumedError

    # 데이터를 조회하는 함수
    def read_data(self, node_1):
        with self.driver.session(database=self.database) as session:
            results = session.read_transaction(self._find_and_return_data, node_1)
            for record in results:
                print(f"Found relationship: {record['r.description']} between {record['n1.name']} and {record['n2.name']}")


    @staticmethod
    def _find_and_return_data(tx, node_1):
        query = (
            "MATCH (n1 {name: $node_1})-[r:RELATED_TO]->(n2) "
            "RETURN n1.name, n2.name, r.description"
        )
        result = tx.run(query, node_1=node_1)
        return [record for record in result]  # Collecting results as a list to avoid ResultConsumedError

    def delete_all_data(self):
        with self.driver.session(database=self.database) as session:
            session.write_transaction(self._delete_all_data)

    @staticmethod
    def _delete_all_data(tx):
        query = (
            "MATCH (n) DETACH DELETE n"
        )
        tx.run(query)

# Neo4j 데이터베이스 정보 설정
uri = "bolt://localhost:7687"  # 데이터베이스 URI
user = "neo4j"  # 사용자명
password = "12345678"  # 비밀번호
database = "reuters"  # 데이터베이스 이름

# 데이터베이스 객체 생성
db = Neo4jDatabase(uri, user, password, database)