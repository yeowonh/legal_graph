from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path
import re

# 불필요한 공백 및 개행 제거 함수
def delete_newlines(html_text: str) -> str:
    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_text, 'html.parser')

    for element in soup.find_all(text=True):
        if element.parent.name not in ['pre', 'textarea']:  # 원본 텍스트 보존 태그는 제외
            # 태그 내 공백을 유지한 채로 앞뒤 공백 및 개행 문자만 제거
            cleaned_text = element.strip()
            element.replace_with(cleaned_text)

    # soup를 다시 문자열로 변환하면서 불필요한 개행과 공백을 제거
    cleaned_html = ''.join(str(soup).splitlines())
    return cleaned_html

# GPT 응답 받아오는 함수 밖으로 빼기
# html -> markdown 
def markdown_parsing(content: str, prompt: str, model_name: str):
    # Openai 불러오기
    load_dotenv(verbose=True)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    completion = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return completion.choices[0].message.content

# sub directory 여부 확인
def has_subdirectory(directory):
    return any(entry.is_dir() for entry in Path(directory).iterdir())
