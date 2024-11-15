"""
DART_crawler.ipynb의 py 버전 -> 터미널 실행 가능
python3 codes/dart/DART_crawler.py --URL "https://dart.fss.or.kr/info/selectGuide.do"
"""

import os
import shutil
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
from bs4 import BeautifulSoup as bs
import requests



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
import shutil

# For Chrome
#Selenium - Webdriver version 호환성
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait



# 현재 스크립트의 경로를 기준으로 절대 경로로 ROOT_PATH 설정
#ROOT_PATH = '../DCM/dart/'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_PATH = os.path.join(PARENT_DIR, 'DCM', 'dart')


def create_directory(path):
    if not os.path.exists(path):
        print(f'## Make directory in {path}')
        os.makedirs(path)


def find_directory(directory, url_num):
    if not os.path.exists(directory):
        return False  # 디렉토리가 없으면 False 반환
    
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if os.path.isdir(full_path) and name.startswith(url_num):
            return full_path  # 조건에 맞는 디렉토리가 있으면 full_path 반환
        
    return None  # 찾지 못하면 None 반환


def make_dir_path(url_num: str, title: str, tab_num=None, chapter_num=None, subtitle=None):
    print(f"## url_num {url_num}, title {title}, tab_num {tab_num}, chapter_num {chapter_num}, subtitle {subtitle}")

    # 상위 디렉토리 (e.g. 110_공시서류확인및서명)
    if tab_num is None and chapter_num is None:
        path = os.path.join(ROOT_PATH, url_num + '_' + title)
        create_directory(path)
    
    # 하위 디렉토리 (e.g. 110_1_제도안내)
    elif tab_num is not None and chapter_num is None:
        parent_dir = find_directory(ROOT_PATH, url_num)
        if parent_dir:
            path = os.path.join(parent_dir, url_num + '_' + tab_num + '_' + title)
            create_directory(path)
        else:
            raise ValueError("No parent directory exists!")
    
    # 탭 안에 하위 탭 (이하 챕터) 있는 경우 (e.g. 220_1-1_제도안내_개요)
    else:
        parent_dir = find_directory(ROOT_PATH, url_num)
        if parent_dir:
            path = os.path.join(parent_dir, url_num + '_' + tab_num + '-' + chapter_num + '_' + title + '_' + subtitle)
            create_directory(path)
        else:
            raise ValueError("No parent directory exists!")

    return path


def get_page_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = bs(response.text, 'html.parser')
        return soup.prettify()
    else: 
        print(f"Failed to retrieve the page. Status code: {response.status_code}")


def html_parsing(html, url_num, parent_title, parent_path):
    soup = bs(html, 'html.parser')
    li_elements = soup.select("#contents > div.content > div.page-tab > ul > li")

    if len(li_elements) == 0:
        print(f'## Tabs not exists!')
        content_element = soup.select("#contents > div.content > div")[0].prettify()
        with open(os.path.join(parent_path, parent_title + ".txt"), "w") as file:
            file.write(content_element)
    else:
        for idx, li in enumerate(li_elements, start=1):
            tab_title = li.get_text(strip=True).strip().replace(' ', '')
            path = make_dir_path(url_num, tab_title, str(idx))
            content = soup.select_one(f"#content-tab-{idx}").prettify()

            sub_soup = bs(content, 'html.parser')
            sub_li_elements = sub_soup.select(f"#content-tab-{idx} > div.sub-top-tab > ul > li")

            if sub_li_elements:
                for sub_idx, sub_li in enumerate(sub_li_elements, start=1):
                    subtab_title = sub_li.get_text(strip=True).strip().replace(' ', '')
                    path = make_dir_path(url_num, tab_title, str(idx), str(sub_idx), subtab_title)
                    sub_content = soup.select_one(f"#content-tab-{idx}-{sub_idx}").prettify()
                    with open(os.path.join(path, tab_title + "_" + subtab_title + ".txt"), "w") as file:
                        file.write(sub_content)
            else:
                with open(os.path.join(path, tab_title + ".txt"), "w") as file:
                    file.write(content)




#### hwp parsing #### 
def process_sub_tab(driver, number, sub_tab, default_download_dir, path):

    #save_path = ../DCM/dart/220_주요사항보고서/220_1_제도안내
    time.sleep(1)  # 페이지 로드 대기
    driver.execute_script("arguments[0].click();", sub_tab)
    time.sleep(1)  # 페이지 로드 대기

    # SUB TAB 제목 추출 
    sub_tab_title = sub_tab.text
    # 다운로드 버튼 처리
    try:
        # 모든 content-tab 관련 요소 찾기
        tab_elements = driver.find_elements(By.XPATH,  "//div[contains(@id, 'content-tab')]")
        
        # 각 요소에서 id 속성 추출하고 number와 매칭
        for tab_element in tab_elements:
            parent_id = tab_element.get_attribute("id")  # id 속성 값 추출

            # id에서 숫자 부분 추출 (정규 표현식 사용)
            match = re.search(r'\d+-\d+', parent_id)  # 예: content-tab-2-3에서 '2-3' 추출
            if match:
                id_number = match.group()  # 숫자 부분만 추출
                # 추출한 id_number와 number 매칭
                if id_number == number:
                    # 매칭되는 요소에서 .btn-download 찾기
                    download_buttons = tab_element.find_elements(By.CSS_SELECTOR, ".btn-download")
                    
                    # 각 다운로드 버튼에 대해 다운로드 처리
                    for num, download_button in enumerate(download_buttons, start=0):
                        download_hwp_file(num, download_button, default_download_dir, path)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    

def menu_crawler(url, url_num, path):
    default_download_dir = os.path.abspath(os.path.join(path, "temp_downloads")) 
    #e.g.,  c:\Users\Shic\2_development\GIB\legal_graph\DCM\dart\110_공시서류확인및서명\temp_downloads 
   
    
    # 기본 다운로드 경로가 없으면 생성
    if os.path.exists(default_download_dir):
        shutil.rmtree(default_download_dir)
    os.makedirs(default_download_dir)

    # Chrome의 다운로드 경로 설정
    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory' : default_download_dir}
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": default_download_dir,  # 절대 경로 사용
        "download.prompt_for_download": False,  # 다운로드 확인 창 비활성화
        "download.directory_upgrade": True,  # 기존 디렉토리 사용 허용
        "safebrowsing.enabled": True , # 안전 브라우징 활성화
        'prefs': default_download_dir
    })

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options) 
    driver.get(url)
    time.sleep(2)  # 페이지 로드 대기

    try :
        tabs = driver.find_elements(By.CSS_SELECTOR, '.page-tab .nav-tabs li a') 
        tabs_len = len(tabs)
        print(f"{url_num}의 tab 개수  : {tabs_len} ") # 220의 길이가 15? 
        for idx, tab in enumerate(tabs, start=0):
            driver.execute_script("arguments[0].click();", tab)
            tab_title = tab.text
            path = make_dir_path(url_num, tab_title, str(idx+1))

            sub_tabs = driver.find_elements(By.CSS_SELECTOR, ".tab-pane.active .sub-top-tab .nav-tabs li a")
            sub_tabs_len = len(sub_tabs)
            print(f"{url_num}의 {tab_title}의 sub tab 개수 : {sub_tabs_len}")

            if sub_tabs:
                print("## 하위 탭 존재 ##")
                for sub_idx, sub_tab in enumerate(sub_tabs, start=0):
                    driver.execute_script("arguments[0].click();", sub_tab)
                    subtab_title = sub_tab.text
                    subtab_title = subtab_title.replace(' ', '')
                    path = make_dir_path(url_num, tab_title, str(idx+1), str(sub_idx+1), subtab_title)
                    print("## sub path loaded in")
                    process_sub_tab(driver, f"{idx+1}-{sub_idx+1}" ,sub_tab, default_download_dir, path)
                continue
                        
            else:
                print("## 하위 탭 없음 ##")
                download_buttons = driver.find_elements(By.CSS_SELECTOR, ".tab-pane.active .btn-download")
                for idx, download_button in enumerate(download_buttons, start=0):
                    download_hwp_file(idx, download_button, default_download_dir, path)
                
            #다운로드 폴더 삭제
            #shutil.rmtree(default_download_dir)

    except Exception as e:
        print(f"tab이 존재하지 않습니다. ") 
        download_buttons = driver.find_elements(By.CSS_SELECTOR, ".tab-pane.active .btn-download")        
        for idx, download_button in enumerate(download_buttons, start=0 ) :
            download_hwp_file(idx, download_button, default_download_dir, path)
        

    

def download_hwp_file(idx, download_button, default_download_dir, path):
    try:
        #다운로드 버튼이 click 가능한지 확인 
        if not download_button.is_enabled():
            time.sleep(2) 
        # 다운로드 버튼 클릭
        download_button.click()
        time.sleep(3)  # 파일 다운로드를 위한 대기 시간
        #다운로드가 완료되지 않은 .crowded 경우 처리

        # 다운로드된 파일 목록 출력 및 이동
        downloaded_file_name = os.listdir(default_download_dir)[0]
        print(downloaded_file_name," 다운로드 완료 ")
        downloaded_file_path  = os.path.join(default_download_dir, downloaded_file_name)  #eg., legal_graph/DCM/dart/220_주요사항보고서/220_1-1_개요/temp_downloads/공시진행.hwp
        move_file_path  = os.path.join(path, f"{idx}_{downloaded_file_name}")
        #rename = os.path.join(default_download_dir, f"{downloaded_file.split('.')[0]}_{index}_{downloaded_file.split('.')[1]}"
        
        #같은 파일 있으면 다음 번호 붙여주기 
        index = 0 
        if os.path.exists(move_file_path):
            index += 1
            os.rename(downloaded_file_path, f"{downloaded_file_path.split('.')[0]}_{index}_{downloaded_file_path.split('.')[1]}")
        
        # 파일 이동
        shutil.move(downloaded_file_path, move_file_path)
        time.sleep(2)  # 파일 move 대기 
        #print(f"파일 {downloaded_file_path}을 {move_file_path}로 이동")

        # 다운로드 폴더 비우기 
        shutil.rmtree(default_download_dir)
    except Exception as e:
        pass 


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
    menu_btn = driver.find_element(By.CLASS_NAME, 'btn-menu')
    menu_btn.click()
    time.sleep(1)


    # 메뉴 별 폴더 생성 
    #기존에 있는 경우 지우고 재생성 
    if os.path.exists(ROOT_PATH):
        print(f"{ROOT_PATH} 폴더 삭제")
        shutil.rmtree(ROOT_PATH)
    os.makedirs(ROOT_PATH, exist_ok=True)


    # ul li 안의 ul li 안의 a 태그를 가져오기
    menu_list = driver.find_elements(By.CSS_SELECTOR, 'ul li ul li')

    for idx, menu in enumerate(menu_list):
        a = menu.find_element(By.TAG_NAME, 'a')
        link_text = a.get_attribute('innerText').strip().replace(" ", "")
        link_href = a.get_attribute('href')

        if link_text:
            print(f"텍스트: {link_text}, 링크: {link_href}")
            try:
                url_num = re.findall(r'\d+', link_href)[0]
            except:
                print(f"공시업무 게시판 제외")
                continue
            
            path = make_dir_path(url_num, link_text) # 110_공시서류확인및서명     
            print("path", path)
            #html = get_page_html(link_href)
            #html_parsing(html, url_num, link_text.replace(' ', ''), path)
            menu_crawler(link_href, url_num, path)
            

        else:
            print(f"빈 텍스트: 링크: {link_href}")
    print("## 크롤링 완료! ##")
    driver.quit()



# 프로젝트의 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser()

g = parser.add_argument_group("Arguments")
g.add_argument("--URL", type=str, required=True, default="https://dart.fss.or.kr/info/selectGuide.do", help="site URL")



if __name__ == "__main__":
    
    exit(main(parser.parse_args()) )
