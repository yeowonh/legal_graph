import argparse
import json
import re
import sys, os
import copy

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from codes.legal.utils import split_with_check, split_articles, complete_articles, process

parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--file_path", type=str, default="DCM/1-2/DCM_1-2_main.txt", help="txt file path")


# 편 > 장 > 절 > 관 > 조
pattern = [r"(제\d+[-\d]*편\s\w+)\n\n(.*?)((?=제\d+[-\d]*편\s\w+)|\Z)", r"(제\d+[-\d]*장\s\w+)\n\n(.*?)((?=제\d+[-\d]*장\s\w+)|\Z)", r"(제\d+[-\d]*절\s\w+)\n\n(.*?)((?=제\d+[-\d]*절\s\w+)|\Z)", r"(제\d+[-\d]*관\s\w+)\n\n(.*?)((?=제\d+[-\d]*관\s\w+)|\Z)"]
def main(args):
    with open(args.file_path, "r", encoding="utf-8") as f:
        contents = f.readlines()

    document_title = contents[0].replace("\ufeff", "")
    date, revise_info = re.split(r" \[", contents[1])
    date = date.replace("[", "").replace("]", "")
    revise_info = revise_info.replace("[", "").replace("]", "")

    source = contents[-2].strip()

    metadata = {
        "document_title": document_title,
        "date": date,
        "revise_info": revise_info,
        "source": source,
        "title": {
            "doc" : None,
            "chapter" : None,
            "section" : None,
            "subsection" : None,
        },
    }

    row = {
        "index" : None,
        "subtitle" : None,
        "content" : None,
        "metadata" : metadata
    }

    articles_list = [] # 전체 조항 리스트

    # 부칙 : \n\n\n\n 단위로 부칙이 나뉨
    if 'supplementary' in args.file_path:
        print('## 부칙 법령 처리 ##')
        contents = ''.join(contents[2:]).split("\n" * 4)[1:]
        for content in contents:
            # 부칙은 바로 조부터 시작
            content_list = content.split('\n')
            metadata["title"] = content_list[0]
            content = '\n'.join(content_list[1:])
            articles_list += process(None, content, metadata, 4)

    
    else:
        print('## main 법령 처리 ##')
        # main : 하나만
        content_text = ''.join(contents[2:]).split("\n" * 4)[1]

        docs_title, docs = split_with_check(0, content_text)
        docs_text = [doc for doc in docs if doc != '' and doc != None]

        if docs_title and docs_text: # 편이 있을 경우
            for doc_title, doc_text in zip(docs_title, docs_text):
                chapters_title, chapters = split_with_check(1, doc_text)
                chapters_text = [chapter for chapter in chapters if chapter != '' and chapter != None]
                
                if chapters_title and chapters_text: # 장이 있는 경우
                    for chapter_title, chapter_text in zip(chapters_title, chapters_text):
                        sections_title, sections = split_with_check(2, chapter_text)
                        sections_text = [section for section in sections if section != '' and section != None]

                        if sections_title and sections_text: # 절이 있는 경우
                            for section_title, section_text in zip(sections_title, sections_text):
                                subsections_title, subsections = split_with_check(3, section_text)
                                subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]
                                
                                if subsections_title and subsections_text: # 관이 있을 경우
                                    for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                        print("# (편 - 장 - 절 - 관)")
                                        articles = split_articles(subsection_text) # (편 - 장 - 절 - 관)
                                        articles_list += complete_articles(row, articles, [doc_title, chapter_title, section_title, subsection_title])
                                
                                elif subsections_title:
                                    print("# (편 - 장 - 절 - 관)")
                                    subsection_title = subsections_title # (편 - 장 - 절 - 관)
                                    articles = split_articles(subsection_text)
                                    articles_list += complete_articles(row, articles, [doc_title, chapter_title, section_title, subsection_title])
                                
                                else: # 관이 없을 경우
                                    print("# (편 - 장 - 절)")
                                    articles = split_articles(section_text) # (편 - 장 - 절)
                                    articles_list += complete_articles(row, articles, [doc_title, chapter_title, section_title, None])
                        
                        else: # 절이 없는 경우
                            subsections_title, subsections = split_with_check(3, chapter_text)
                            subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]
                            
                            if subsections_title and subsections_text: # 관이 있는 경우
                                for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                    print("# (편 - 장 - 관))")
                                    articles = split_articles(subsection_text)  # (편 - 장 - 관)
                                    articles_list += complete_articles(row, articles, [doc_title, chapter_title, None, subsection_title])
                            elif subsections_title:
                                print("# (편 - 장 - 관)")
                                subsection_title = subsections_title # (편 - 장 - 관)
                                articles = split_articles(subsection_text)
                                articles_list += complete_articles(row, articles, [doc_title, chapter_title, section_title, subsection_title])

                            else: # 관이 없는 경우
                                print("# (편 - 장)")
                                articles = split_articles(chapter_text) # (편 - 장)
                                articles_list += complete_articles(row, articles, [doc_title, chapter_title, None, None])
                
                elif chapters_title:
                    print("# (편 - 장)")
                    chapter_title = chapters_title # ()
                    

                else: # 장이 없는 경우
                    sections_title, sections = split_with_check(2, doc_text)
                    sections_text = [section for section in sections if section != '' and section != None]

                    if sections_title and sections_text: # 절이 있는 경우
                        for section_title, section_text in zip(sections_title, sections_text):
                            subsections_title, subsections = split_with_check(3, section_text)
                            subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]
                            
                            if subsections_title and subsections_text: # 관이 있는 경우
                                for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                    print("# (편 - 절 - 관)")
                                    articles = split_articles(subsection_text) #(편 - 절 - 관)
                                    articles_list += complete_articles(row, articles, [doc_title, None, section_title, subsection_title])
                            elif subsections_title:
                                subsection_title = subsections_title # (편 - 절 - 관)
                                print("# (편 - 절 - 관)")
                                articles = split_articles(subsection_text) #(편 - 절 - 관)
                                articles_list += complete_articles(row, articles, [doc_title, None, section_title, subsection_title])

                            else: # 관이 없는 경우
                                print("# (편 - 절)")
                                articles = split_articles(section_text) # (편 - 절)
                                articles_list += complete_articles(row, articles, [doc_title, None, section_title, None])
                    
                    elif sections_title:
                        section_title = sections_title



                    else: # 절이 없는 경우
                        subsections_title, subsections = split_with_check(3, doc_text)
                        if subsections_title and subsections_text:  # 관이 있는 경우
                            for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                articles = split_articles(subsection_text) # 관이 있는 경우 (편 - 관)
                                articles_list += complete_articles(row, articles, [doc_title, None, None, subsection_title])
                        elif subsections_title:
                            subsection_title = subsections_title # (편 - 관)
                            print("# (편 - 관))")
                            articles = split_articles(subsection_text) # (편 - 관)
                            articles_list += complete_articles(row, articles, [doc_title, None, section_title, subsection_title])
                        else: # 관이 없는 경우
                            articles = split_articles(doc_text) # 관이 없는 경우 (편)
                            articles_list += complete_articles(row, articles, [doc_title, None, None, None])

        elif docs_title:
            doc_title = docs_title

        else: # 편이 없을 경우
            chapters_title, chapters = split_with_check(1, content_text)
            chapters_text = [chapter for chapter in chapters if chapter != '' and chapter != None]

            if chapters_title and chapters_text: # 장이 있을 경우
                for chapter_title, chapter_text in zip(chapters_title, chapters_text):
                    sections_title, sections = split_with_check(2, chapter_text)
                    sections_text = [section for section in sections if section != '' and section != None]

                    if sections_title and sections_text: # 절이 있는 경우
                        for section_title, section_text in zip(sections_title, sections_text):
                            subsections_title, subsections = split_with_check(3, section_text)
                            subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]
                            
                            if subsections_title and subsections_text: # 관이 있을 경우
                                for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                    articles = split_articles(subsection_text) # (장 - 절 - 관)
                                    articles_list += complete_articles(row, articles, [None, chapter_title, section_title, subsection_title])
                            elif subsections_title:
                                subsection_title = subsections_title # (장 - 절 - 관)
                            else: # 관이 없을 경우
                                articles = split_articles(section_text) # (장 - 절)
                                articles_list += complete_articles(row, articles, [None, chapter_title, section_title, None])
                    elif sections_title:
                        section_title = sections_title
                    else: # 절이 없을 경우
                        subsections_title, subsections = split_with_check(3, chapter_text)
                        subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]

                        if subsections_title and subsections_text:  # 관이 있는 경우
                            for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                articles = split_articles(subsection_text) # 장 - 관
                                articles_list += complete_articles(row, articles, [None, chapter_title, None, subsection_title])
                        elif subsections_title:
                            subsection_title = subsections_title # (장 - 관)
                        else: # 관이 없을 경우
                            print("# (장)")
                            print(chapter_text)
                            articles = split_articles(chapter_text) # (장)
                            articles_list += complete_articles(row, articles, [None, chapter_title, None, None])
            elif chapters_title:
                chapter_title = chapters_title

            else: # 장이 없는 경우
                sections_title, sections = split_with_check(2, content_text)
                sections_text = [section for section in sections if section != '' and section != None]

                if sections_title and sections_text: # 절이 있는 경우
                    for section_title, section_text in zip(sections_title, sections_text):
                        subsections_title, subsections = split_with_check(3, section_text)
                        subsections_text = [subsection for subsection in subsections if subsection != '' and subsection != None]
                        
                        if subsections_title and subsections_text:  # 관이 있을 경우
                            for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                                articles = split_articles(subsection_text) # (절 - 관)
                                articles_list += complete_articles(row, articles, [None, None, section_title, subsection_title])
                        elif subsections_title:
                                subsection_title = subsections_title # (절 - 관)
                        else: # 관이 없을 경우
                            articles = split_articles(section_text) # (절)
                            articles_list += complete_articles(row, articles, [None, None, section_title, None])

                elif sections_title:
                    section_title = sections_title
                else: # 절이 없을 경우
                    subsections_title, subsections = split_with_check(3, section_text)
                    if subsections_title and subsections_text: # 관이 있는 경우
                        for subsection_title, subsection_text in zip(subsections_title, subsections_text):
                            articles = split_articles(subsection_text) # (편 - 장 - 관)
                            articles_list += complete_articles(row, articles, [doc_title, chapter_title, None, subsection_title])
                    elif subsections_title:
                        subsection_title = subsections_title # (편 - 장  - 관)
                    else: # 관이 없는 경우
                        articles = split_articles(chapter_text) # (편 - 장)
                        articles_list += complete_articles(row, articles, [doc_title, chapter_title, None, None])


    path = args.file_path.split('/')

    os.makedirs(os.path.join("results", path[1]), exist_ok=True)
    RESULT_PATH = os.path.join("results", path[1], path[2].split('.')[0] + ".json")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    exit(main(parser.parse_args()))
