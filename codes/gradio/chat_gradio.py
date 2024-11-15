import sys, os
# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

import argparse
from typing import Dict
import json
import gradio as gr
from GraphDB.models import ChatModel


def main(config: Dict):
    load_dotenv("C:/Users/Shic/Downloads/legal_graph/.env")
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Settings")
    g.add_argument("--model", type=str, default=config["model"], help="GPT model name")
    g.add_argument("--embedding_model", type=str, default=config["embedding_model"], help="GPT model name")
    g.add_argument("--chunk_size", type=int, default=config["chunk_size"], help="data chunk size")
    g.add_argument("--chunk_overlap", type=int, default=config["chunk_overlap"], help="data overlap size")
    g.add_argument("--top_k", type=int, default=config["top_k"], help="Retrieve documents")
    g.add_argument("--rag", type=str, default=config['rag'], help="Graph or Vector")

    args = parser.parse_args()

    print("## Settings ##")
    print("## model : ", args.model)
    print("## embedding_model : ", args.embedding_model)
    print("## chunk_size : ", args.chunk_size)
    print("## chunk_overlap : ", args.chunk_overlap)
    print("## top_k : ", args.top_k)
    print("## rag : ", args.rag)
    
    # conversational loop
    chat_model = ChatModel()

    def inference(query, top_k=args.top_k):
        print('## inference query : ', query)
        response, documents = chat_model.get_documents_answer(query=query, top_k=top_k)
        return response, documents
    
    
    # 반환값을 2개로
    gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(lines=2, label="query", placeholder="증권신고서를 부실기재했을 때 법인의 대표자에게 부과되는 과징금은 얼마인가?"),
            gr.components.Slider(
                minimum=0, maximum=20, step=1, value=args.top_k, label="Max Top k"
            )
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Chatbot Response",
            ),
            gr.components.Textbox(
                lines=10,
                label="Retrieved Documents",
            )
        ],
        
        title="GraphRAG 기반의 온톨로지 지식 법률 QA 챗봇",
        description=f"(사용 모델 : {args.model}), (RAG Mode : {args.rag})",
    ).queue().launch(share=True, debug=True)


if __name__ == "__main__":
    try:
        with open('configs/config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found. Please check the path.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
        sys.exit(1)
    
    main(config=config)