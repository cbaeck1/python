import datetime
import re
from client.app_utils import getTimezone
from semantic.dates import DateService

WORDS = ["TIME"]	
# 각 모듈에 대한 구분을 위해 명시하는 코드입니다.

def handle(text, mic, profile):
    tz = getTimezone(profile)
    now = datetime.datetime.now(tz=tz)	
    service = DateService()
    response = service.convertTime(now)
    mic.say("현재 시각은 %s 입니다." % response)
    # Profile에 등록된 지역에 따른 시간정보를 불러와 STT에 대한 TTS 음성출력을 위한 코드입니다.

def isValid(text):
    return bool(re.search(r'\btime\b', text, re.IGNORECASE))
    # 입력된 음성과 비교하기 위한 코드입니다. 한글을 사용하기 위해서는 
    # return bool(re.search(ur'시간', phrase, re.UNICODE))
    # 과 같은 방식으로 수정해야합니다.