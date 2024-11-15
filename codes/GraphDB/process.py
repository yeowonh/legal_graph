import re

def replace_hanja(items: list[dict]) -> list[dict]:
    """
    text 받아서 한자 제거 (전처리)
    """
    # 한자 범위 정의
    hanja_range = (
        r'\u3400-\u4DBF'   # CJK Unified Ideographs Extension A
        r'\u4E00-\u9FFF'   # CJK Unified Ideographs
        r'\uF900-\uFAFF'   # CJK Compatibility Ideographs
        # 아래 두 범위는 Python의 re 모듈에서 지원하지 않는 범위이므로 제외합니다.
        # r'\U00020000-\U0002EBEF'  # CJK Unified Ideographs Extensions B-F
        # r'\U0002F800-\U0002FA1F'  # CJK Compatibility Ideographs Supplement
    )

    # 한자를 포함한 괄호 패턴
    hanja_parens_pattern = re.compile(
        rf'\(\s*[{hanja_range}]+\s*\)'
    )

    # 한자만 제거하는 패턴
    hanja_pattern = re.compile(
        rf'[{hanja_range}]'
    )

    for idx, item in enumerate(items):
        # 한자를 포함한 괄호와 그 안의 내용을 제거
        preprocessed_content = hanja_parens_pattern.sub('', item['content'])
        # 남은 한자 제거
        preprocessed_content = hanja_pattern.sub('', preprocessed_content)
        items[idx]['content']= preprocessed_content

    return items

import re

def postprocess(triplets:list, manual_remove_idx=[]):
    clause_pattern = r'^제\d+조(제\d+항)?$'
    pattern = r'\d+조(의\d+)*제\d+항$'

    new_triplets = []
    remove_idx = []
    
    for idx, triplet in enumerate(triplets):
        subject = triplet[0]
        relation = triplet[1]
        object_ = triplet[2]

        subject_keyword = subject.get('keyword_name', '')
        object_keyword = object_.get('keyword_name', '')

        # Rule 1: Exclude if keyword ends with '한다'
        if re.search('한다$', subject_keyword) or re.search('한다$', object_keyword):
            remove_idx.append(idx)

        # Rule 2: Exclude if keywords are the same except for '등' appended
        elif subject_keyword == object_keyword + '등' or object_keyword == subject_keyword + '등':
            remove_idx.append(idx)

        # Rule 3: Exclude if keywords are the same except for '각' prepended
        elif subject_keyword == '각 ' + object_keyword or object_keyword == '각 ' + subject_keyword:
            remove_idx.append(idx)

        # Rule 4: Exclude if keyword contains '이 절' or '그'
        elif '이 절' == subject_keyword or '그' == subject_keyword or '이 절' == object_keyword or '그' == object_keyword:
            remove_idx.append(idx)

        # Rule 6 : '각 호'가 들어가는 경우 제거 - keyword 보다는 본질적으로 content에 가까운 듯
        elif "각 호" in subject_keyword or "각 호" in object_keyword :
            remove_idx.append(idx)
            
        # Rule 6: 한쪽이 'X조, X조X항' 인 경우 앞에 '제' 붙이기
        if re.match(pattern, subject_keyword):
             subject_keyword = '제' + subject_keyword
        
        if re.match(pattern, object_keyword):
             object_keyword = '제' + object_keyword

        # Rule 5: Exclude if both keywords are '제X조' or '제X조제X항' (exception : 제63조제1항-[INCLUDED_IN]-제289조)
        if re.match(clause_pattern, subject_keyword) and re.match(clause_pattern, object_keyword):
            remove_idx.append(idx)

        # 숫자 (1.), 대시 (-) 제거, output: 부분 제거
        subject['keyword_name'] = re.sub(r'^(- |[0-9]+\. |output: )', '', subject_keyword)
        object_['keyword_name']  = re.sub(r'^(- |[0-9]+\. |output: )', '', object_keyword)

        new_triplets.append((subject, relation, object_))

    remove_idx = list(set(remove_idx).union(manual_remove_idx))

    new_triplets = [x for idx, x in enumerate(new_triplets) if idx not in remove_idx]
    return remove_idx, new_triplets
