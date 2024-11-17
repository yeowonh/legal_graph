# legal_graph
GraphRAG 기반의 온톨로지 지식 법률 QA 챗봇 시스템 구축

Ontology-based Legal Knowledge QA Chatbot with GraphRAG System


```shell
├─codes
│  ├─configs
│  ├─dart // 공시길라잡이 크롤링
│  │  └─prompt // pdf parser code
│  ├─gradio // 시연용 gradio
│  ├─GraphDB
│  │  ├─prompt // 프롬프트
│  │  └─summarization_test
│  ├─legal // 법률 preprocess
│  └─pdf2docx2md // gpt parser test
│      └─korean
├─data
│  ├─codes
│  ├─DCM
│  │  ├─DCM_json // DCM data 가공 버전
│  │  │  ├─01
│  │  │  │  └─jsonl
│  │  │  ├─02
│  │  │  │  ├─chunk
│  │  │  │  └─jsonl
│  │  │  ├─03
│  │  │  │  └─jsonl
│  │  │  ├─04
│  │  │  │  ├─chunk
│  │  │  │  └─jsonl
│  │  │  └─08
│  │  │      ├─chunk
│  │  │      └─jsonl
│  │  └─DCM_original // DCM data 원본
│  │      ├─01_자본시장과금융투자업에관한법률
│  │      ├─02_금융지주회사감독규정
│  │      ├─03_증권의발행및공시등에관한규정
│  │      ├─04_kofia
│  │      ├─05_전자단기사채등의발행및유통에관한법률
│  │      ├─06_기업공시서식작성기준
│  │      ├─07_기업공시실무안내
│  │      ├─08_은행법
│  │      ├─09_DART
│  │      ├─10_개인정보보호법
│  │      ├─11_전자금융거래법
│  │      └─12_전자서명법
│  └─graph
│      └─clause
│          ├─edge_triplet
│          │  ├─01
│          │  ├─02
│          │  ├─03
│          │  ├─04
│          │  └─08
│          ├─matched_pattern
│          └─retrieve
├─graphrag // test files
└─results // previous result file path
    ├─1-2
    ├─1-6
    ├─1-7
```