"""
DART_html_crawler.ipynb
의 py 버전 -> 터미널 실행 가능

python3 codes/DART_html_cralwer.py --URL "https://dart.fss.or.kr/info/selectGuide.do"

"""
ROOT_PATH = '../DCM/dart/'

import requests               
from bs4 import BeautifulSoup as bs
import json
import os
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import re
import sys, os
import argparse

# For Chrome
#Selenium - Webdriver version 호환성
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def create_directory(path):
    if not os.path.exists(path):
        print(f'## Make directory in {path}')
        os.makedirs(path)

# 'dart/' 폴더 내에서 '{url_num}_'으로 시작하는 디렉토리 확인
def find_directory(directory, url_num):
    if not os.path.exists(directory):
        return False  # 디렉토리가 없으면 False 반환
    
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if os.path.isdir(full_path) and name.startswith(url_num):
            return full_path  # 조건에 맞는 디렉토리가 있으면 full_path 반환
        
    return None  # 찾지 못하면 None 반환


def make_dir_path(url_num:str, title:str, tab_num=None, chapter_num=None, subtitle=None):
    print(f"## url_num {url_num}, title {title}, tab_num {tab_num}, chapter_num {chapter_num}, subtitle {subtitle}")

    # 상위 디렉토리 (e.g. 110_공시서류확인및서명)
    if tab_num == None and chapter_num == None:
        path = os.path.join(ROOT_PATH, url_num + '_' + title)
        create_directory(path)
    
    # 하위 디렉토리 (e.g. 110_1_제도안내)
    elif tab_num != None and chapter_num == None:
        # url_num이 같은 상위 디렉토리 서칭하기
        parent_dir = find_directory(ROOT_PATH, url_num)
        if parent_dir != None:
            path = os.path.join(parent_dir, url_num + '_' + tab_num + '_' + title)
            create_directory(path)
        else:
            raise ValueError("No parent directory exists!")
    
    # 탭 안에 하위 탭 (이하 챕터) 있는 경우 (e.g. 220_1-1_제도안내_개요)
    # # ../DCM/dart/220_2-1_세부업무_Ⅰ 개요 
    else:
        # url_num이 같은 상위 디렉토리 서칭하기
        parent_dir = find_directory(ROOT_PATH, url_num)
        if parent_dir != None:
            path = os.path.join(parent_dir, url_num + '_' + tab_num + '-' + chapter_num  + '_' + title + '_' + subtitle)
            create_directory(path)
        else:
            raise ValueError("No parent directory exists!")

    return path


def get_page_html(url):
    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
        soup = bs(html, 'html.parser')
        return soup.prettify()

    else: 
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

def get_url_num(url):
    return re.findall(r'\d+', url)[0]

def html_parsing(html, url_num, parent_title, parent_path):
    soup = bs(html, 'html.parser')

    li_elements = soup.select("#contents > div.content > div.page-tab > ul > li")

    # 탭 없는 경우 존재 (menu=500)
    if len(li_elements) == 0:
        print(f'## Tabs not exists!')
        
        content_element = soup.select("#contents > div.content > div")[0].prettify()

        with open(os.path.join(parent_path, parent_title+".txt"), "w") as file:
            print(f"## {parent_path} 에 {os.path.join(parent_path, parent_title+'.txt')} 저장 ##")
            file.write(content_element)

    else:
        # 탭 단위
        for idx, li in enumerate(li_elements, start=1):
            tab_title = li.get_text(strip=True).strip().replace(' ', '')
            print(f'## 현재 크롤링 중인 탭 : {tab_title}')
            path = make_dir_path(url_num, tab_title, str(idx))
            content = soup.select_one(f"#content-tab-{idx}").prettify()
            
            sub_soup = bs(content, 'html.parser')

            sub_li_elements = sub_soup.select(f"#content-tab-{idx} > div.sub-top-tab > ul > li")

            if sub_li_elements:
                print("## 하위 탭 존재 ##")
                for sub_idx, sub_li in enumerate(sub_li_elements, start=1):
                    subtab_title = sub_li.get_text(strip=True).strip().replace(' ', '')
                    path = make_dir_path(url_num, tab_title, str(idx), str(sub_idx), subtab_title)
                    sub_content = soup.select_one(f"#content-tab-{idx}-{sub_idx}").prettify()

                    with open(os.path.join(path, tab_title + "_" + subtab_title +".txt"), "w") as file:
                        print(f"## {path} 에 {os.path.join(path, tab_title + '_' + subtab_title +'.txt')} 저장 ##")
                        file.write(sub_content)
                    
            else:
                with open(os.path.join(path, tab_title+".txt"), "w") as file:
                    print(f"## {path} 에 {os.path.join(path, tab_title+'.txt')} 저장 ##")
                    file.write(content)

# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--URL", type=str, required=True, default="https://dart.fss.or.kr/info/selectGuide.do", help="site URL")


def main(args):
    options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # URL 설정
    URL = args.URL


    # 페이지 열기
    driver.get(URL)
    time.sleep(2)


    # 메뉴 열기 버튼 클릭
    menu = driver.find_element(By.CLASS_NAME, 'btn-menu')
    menu.click()
    time.sleep(1)


    # ul li 안의 ul li 안의 a 태그를 가져오기
    li_elements = driver.find_elements(By.CSS_SELECTOR, 'ul li ul li')

    for idx, li in enumerate(li_elements):
        a = li.find_element(By.TAG_NAME, 'a')
        link_text = a.get_attribute('innerText').strip()
        link_href = a.get_attribute('href')
        
        if link_text:
            print(f"텍스트: {link_text}, 링크: {link_href}")

            # url 넣고 bs4로 파싱
            # 탭 전체 기준 디렉토리 생성
            try:
                url_num = re.findall(r'\d+', link_href)[0]
                
            except:
                print(f"공시업무 게시판 제외")
                continue

            path = make_dir_path(url_num, link_text.replace(' ', '')) # 110_공시서류확인및서명
            # 전처리는 bs4단에서 하기
            html = get_page_html(link_href)
            html_parsing(html, url_num, link_text.replace(' ', ''), path)
            
        else:
            print(f"빈 텍스트: 링크: {link_href}")

    print("## 크롤링 완료! ##")

    driver.quit()

if __name__ == "__main__":
    exit(main(parser.parse_args()))