'''

1. 특정 문자열을 직접 리터럴로 사용하여 해당 문자열을 검색
compile
search() : 처음 매칭되는 문자열
findall() : 매칭되는 모든 경우
group() : 결과 문자열

2. 특정 패턴의 문자열
^	이 패턴으로 시작해야 함	^abc : abc로 시작해야 함 (abcd, abc12 등)
$	이 패턴으로 종료되어야 함	xyz$ : xyz로 종료되어야 함 (123xyz, strxyz 등)
[문자들]	문자들 중에 하나이어야 함. 가능한 문자들의 집합을 정의함.	[Pp]ython : "Python" 혹은 "python"
[^문자들]	[문자들]의 반대로 피해야할 문자들의 집합을 정의함.	[^aeiou] : 소문자 모음이 아닌 문자들
|	두 패턴 중 하나이어야 함 (OR 기능)	a | b : a 또는 b 이어야 함
?	앞 패턴이 없거나 하나이어야 함 (Optional 패턴을 정의할 때 사용)	\d? : 숫자가 하나 있거나 없어야 함
+	앞 패턴이 하나 이상이어야 함	\d+ : 숫자가 하나 이상이어야 함
*	앞 패턴이 0개 이상이어야 함	\d* : 숫자가 없거나 하나 이상이어야 함
패턴{n}	앞 패턴이 n번 반복해서 나타나는 경우	\d{3} : 숫자가 3개 있어야 함
패턴{n, m}	앞 패턴이 최소 n번, 최대 m 번 반복해서 나타나는 경우 (n 또는 m 은 생략 가능)	\d{3,5} : 숫자가 3개, 4개 혹은 5개 있어야 함
\d	숫자 0 ~ 9	\d\d\d : 0 ~ 9 범위의 숫자가 3개를 의미 (123, 000 등)
\w	문자를 의미	\w\w\w : 문자가 3개를 의미 (xyz, ABC 등)
\s	화이트 스페이스를 의미하는데, [\t\n\r\f] 와 동일	\s\s : 화이트 스페이스 문자 2개 의미 (\r\n, \t\t 등)
.	뉴라인(\n) 을 제외한 모든 문자를 의미	.{3} : 문자 3개 (F15, 0x0 등)

3. 정규식 그룹(Group)
 ( ) 괄호는 그룹을 의미한다. 
 예를 들어, 전화번호의 패턴을 \d{3}-\d{3}-\d{4} 와 같이 표현하였을 때, 
 지역번호 3자를 그룹1으로 하고 나머지 7자리를 그룹2로 분리하고 싶을 때, 
 (\d{3})-(\d{3}-\d{4}) 와 같이 둥근 괄호로 묶어 두 그룹으로 분리

'''

import re

# 특정 문자열을 직접 리터럴로 사용하여 해당 문자열을 검색
text = "에러 1122 : 레퍼런스 오류\n 에러 1033: 아규먼트 오류"
regex = re.compile("에러 1033")
mo = regex.search(text)
if mo != None:
    print(mo.group()) 


# 특정 패턴의 문자열 : 전화번호
patternText = "문의사항이 있으면 032-232-3245 으로 연락주시기 바랍니다."
 
regex_pattern = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
matchobj = regex.search(patternText)
phonenumber = matchobj.group()
print(phonenumber)     

# 출력: ['에러 1122', '에러 1033']
regex2 = re.compile("에러\s\d+")
mc = regex2.findall(text)
print(mc)

grouptext = "문의사항이 있으면 032-232-3245 으로 연락주시기 바랍니다."
 
regex = re.compile(r'(\d{3})-(\d{3}-\d{4})')
matchobj = regex.search(grouptext)
areaCode = matchobj.group(1)
num = matchobj.group(2)
fullNum = matchobj.group()
print(areaCode, num) # 032 232-3245

regex = re.compile(r'(?P<area>\d{3})-(?P<num>\d{3}-\d{4})')
matchobj = regex.search(text)
areaCode = matchobj.group("area")
num = matchobj.group("num")
print(areaCode, num)  # 032 232-3245
