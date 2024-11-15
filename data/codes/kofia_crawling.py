import requests
import re
import json
from bs4 import BeautifulSoup

def get_code() -> dict:
    # URL 설정
    url = 'https://law.kofia.or.kr/service/law/lawCurrentPartTree.do'

    # HTTP 요청 헤더 설정
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Cookie': '3=1; WMONID=sTNhbbTn6x6; JSESSIONID=R15mYXwXZodIMTrb5HlfJfRZ6DXXLaxe5oKwEaSRmGBdagblzh7ujIMXpbUb4sUM.ap5_servlet_serviceEngine',
        'Dnt': '1',
        'Host': 'law.kofia.or.kr',
        'Referer': 'https://law.kofia.or.kr/service/law/lawView.do?seq=308&historySeq=0&gubun=cur&tree=part',
        'Sec-Ch-Ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'iframe',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
    }

    # HTTP GET 요청 보내기
    response = requests.get(url, headers=headers)

    # 요청 성공 여부 확인
    if response.status_code == 200:
        print("요청 성공")
        # 응답 내용 출력 또는 파일로 저장
        # print(response.text)  # 여기서 데이터를 원하는 대로 파싱하거나 저장할 수 있습니다.
    else:
        print(f"요청 실패: {response.status_code}")

    # pattern = re.compile(r"gotoLawList\('(\d+)', '.*?', '(.+?)', '.*?', '.*?', '.*?', '.*?', '(.+?)'")
    pattern = re.compile(r"gotoLawList\('(\d+)', '.*?', '(.+?)', '.*?', '.*?', '.*?', '.*?', '.*?', '.*?', '.*?', '(\d+)'")

    matches = pattern.findall(response.text)

    law_dict = {title: [code, history_seq] for code, title, history_seq in matches if len(str(code)) > 1}
    return law_dict


def get_content(seq, history_seq):
    import requests

    url = 'https://law.kofia.or.kr/service/law/lawFullScreenContent.do'

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'connection': 'keep-alive',
        'cookie': '3=1; WMONID=sTNhbbTn6x6; JSESSIONID=Oi3B3mu0QmBN3TeZLWafQoYDu8zMlisqf9AXDTEaNjf1Mip0zv810ozvtXA1tUDj.ap5_servlet_serviceEngine',
        'dnt': '1',
        'host': 'law.kofia.or.kr',
        'referer': 'https://law.kofia.or.kr/service/law/lawView.do?seq=135&historySeq=0&gubun=cur&tree=part',
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'iframe',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
    }

    # 요청할 파라미터 설정
    params = {
        'seq': str(seq),
        'historySeq': str(history_seq)
    }

    # GET 요청 보내기
    response = requests.get(url, headers=headers, params=params)

    # 요청이 성공적으로 이루어졌는지 확인
    if response.status_code == 200:
        print("요청 성공")
        # 응답 내용 출력 또는 파일로 저장
        return response.text  # 응답 내용을 HTML로 출력합니다.
    else:
        print(f"요청 실패: {response.status_code}")

    return

def find_jo(item, tag='td'):
    jo_title = None
    jo_num = None
    for idx, finded_tag in enumerate(item.find_all(tag)):
        if idx > 2:
            return jo_num, jo_title
        
        cand_text = finded_tag.text.replace('\xa0', '').replace('&nbsp;', '').strip()
        match = re.match(r'^\d+\.', cand_text)
        if match:
            jo_num = match.group(0)[:-1]
            jo_title = cand_text
            return jo_num, jo_title

        match = re.search(r'제(\d+)조', cand_text)
        if match:
            # jo_num = match.group(0)[:-1]
            jo_title = cand_text
            return jo_num, jo_title
    
        match = re.search(r'제(\d+)-(\d+)조', cand_text)
        if match:
            # jo_num = match.group(0)[:-1]
            jo_title = cand_text
    return jo_num, jo_title


def make_data(html_text):
    soup = BeautifulSoup(html_text)
    total_items = []

    cand_class = ['hang', 'none', 'dann', 'ho']

    for item in soup.find_all('div', class_='JO'):
        
        for tag in ['td', 'a']:
            # print(item)
            jo_num, jo_title = find_jo(item, tag=tag)
            if jo_title is not None:
                break
        
        if jo_title is None and jo_num is not None:
            jo_title = jo_num

        # print(jo_title, jo_num)
        contents = item.find_all('div', class_=cand_class)
        first_content = {"항": "", "내용": ""}
        items = []
        current_item = None
        for cont in contents:
            if cont['class'][0] == 'hang':
                # 항일 경우
                current_item = {"항": cont.text, "내용": ""}
                items.append(current_item)
            else:
                if current_item:
                    items[-1]['내용'] += cont.text.strip() + "\n"
                # 항 안나왔는데 내용이 먼저 나온 경우
                else:
                    first_content['내용'] += cont.text.strip() + "\n"
        if len(first_content['내용']) == 0:
            total_items.append({jo_title: items}    )
        else:
            total_items.append({jo_title: [first_content] + items})
    return total_items

if __name__ == "__main__":
    code_dict = get_code()

    # 경로 수정 필요함
    with open('/mnt/c/Users/Shic/legal_graph/data/DCM/DCM_original/04_kofia/kofia_law_code_dict.json', 'w', encoding='utf-8') as f:
        print("## json file saved in /mnt/c/Users/Shic/legal_graph/data/DCM/DCM_original/04_kofia/kofia_law_code_dict.json")
        json.dump(code_dict, f, ensure_ascii=False)

    code_dict = json.load(open('/mnt/c/Users/Shic/legal_graph/data/DCM/DCM_original/04_kofia/kofia_law_code_dict.json', encoding='utf-8'))

    crawled = []
    for key in code_dict:
        seq, history_seq = code_dict[key]
        print(key)
        html_text = get_content(seq, history_seq)
        datas = make_data(html_text)
        crawled.append({"title": key, "contents": datas})

    with open('/mnt/c/Users/Shic/legal_graph/data/DCM/DCM_original/04_kofia/증권 인수업무 등에 관한 규정_통합.jsonl', 'w', encoding='utf-8') as f:
        for data in crawled:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + '\n')
