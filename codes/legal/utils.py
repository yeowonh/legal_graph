import re
import json

def parse_law(text: str, meta_info: dict)-> list[dict]:
    law_structure = []
    current_section = {"doc": None, "chapter": None, "section": None, "subsection": None, "supplementary" : None}

    # '편', '장', '절', '관' 패턴
    section_patterns = {
        "doc": re.compile(r'^제\d+편 [^\n]+'),
        "chapter": re.compile(r'^제\d+장 [^\n]+'),
        "section": re.compile(r'^제\d+절 [^\n]+'),
        "subsection": re.compile(r'^제\d+관 [^\n]+'),
        "supplementary" : re.compile(r"부칙 <제\d+(?:-\d+)?호,\d+\. \d+\. \d+\.>")
    }
    
    # 조 패턴 (제X조 or 제X조의Y)
    # 제X-1조, 제X-1조의Y 포함
    clause_pattern = re.compile(r'^(제[\d-]+조(?:의\d+)?)(?:\((.*?)\))?\s*(.*)')

    
    # 텍스트 줄 단위로 분할
    lines = text.splitlines()
    current_clause = None
    
    for line in lines:
        line = line.strip()

        # '편', '장', '절', '관' 갱신
        is_section_line = False
        for section, pattern in section_patterns.items():
            if pattern.match(line):
                current_section[section] = line.strip()
                is_section_line = True
                break  # "편", "장", "절", "관", "부칙" 중 하나를 찾으면, 더 이상 확인하지 않음

        if is_section_line:
            continue  # 섹션 관련 줄은 content에 포함하지 않음

        # 조항 확인
        clause_match = clause_pattern.match(line)
        if clause_match:
            if current_clause:
                # 이전 조항을 저장
                law_structure.append(current_clause)
            # 새 조항 시작
            clause_index = clause_match.group(1)
            subtitle = clause_match.group(2)
            content = clause_match.group(3)
            current_clause = {
                "index": clause_index,
                "subtitle": subtitle,
                "content": content,
                "metadata": {
                    "document_title" : meta_info["document_title"],
                    "date":meta_info["date"],
                    "revise_info":meta_info["revise_info"],
                    "source":meta_info["source"],
                    "title":{
                        "doc": current_section["doc"],
                        "chapter": current_section["chapter"],
                        "section": current_section["section"],
                        "subsection": current_section["subsection"],
                        "supplementary" : current_section["supplementary"]
                    }
                }
            }
        
        # 내용 추가
        elif current_clause and line:
            if current_clause["content"]:
                current_clause["content"] += "\n" + line
            else:
                current_clause["content"] = line

    if current_clause:
        law_structure.append(current_clause)

    return law_structure

def parse_supplementary(text: str, meta_info: dict)->list[dict]:
    law_structure = []
    current_section = {"supplementary": None}

    section_patterns = {
        "supplementary" : re.compile(r"부칙 <제\d+(?:-\d+)?호,\d+\. \d+\. \d+\.>")
    }
        
    # 조 패턴 (제X조 or 제X조의Y)
    # 제X-1조, 제X-1조의Y 포함
    clause_pattern = re.compile(r'^(제[\d-]+조(?:의\d+)?)(?:\((.*?)\))?\s*(.*)')
    
    # 텍스트 줄 단위로 분할
    lines = text.splitlines()
    current_clause = None
    
    for line in lines:
        line = line.strip()


        # '편', '장', '절', '관' 갱신
        is_section_line = False
        for section, pattern in section_patterns.items():
            if pattern.match(line):
                current_section[section] = line.strip()
                is_section_line = True
                break  # "편", "장", "절", "관" 중 하나를 찾으면, 더 이상 확인하지 않음

        if is_section_line:
            continue  # 섹션 관련 줄은 content에 포함하지 않음

        # 조항 확인
        clause_match = clause_pattern.match(line)
        
        if clause_match:
            if current_clause:
                # 이전 조항을 저장
                law_structure.append(current_clause)
            # 새 조항 시작
            clause_index = clause_match.group(1)
            subtitle = clause_match.group(2)
            content = clause_match.group(3)
            
            current_clause = {
                "index": clause_index,
                "subtitle": subtitle,
                "content": content,
                "metadata": {
                    "document_title" : meta_info["document_title"],
                    "date":meta_info["date"],
                    "revise_info":meta_info["revise_info"],
                    "source":meta_info["source"],
                    "title":{
                        "supplementary": current_section["supplementary"]
                    }
                }
            }
        
        # 내용 추가
        elif current_clause and line:
            if current_clause["content"]:
                current_clause["content"] += "\n" + line
            else:
                current_clause["content"] = line

    if current_clause:
        law_structure.append(current_clause)

    return law_structure



def parse_contents(data : list[str]) -> list:
    for idx, row in enumerate(data):
        if bool(re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", row["content"])):
            split_content = re.split(r'(?=①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)', row["content"])
            split_content = [x for x in split_content if x]
            data[idx]["content"] = split_content
        
    return data