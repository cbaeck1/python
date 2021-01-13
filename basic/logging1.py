'''

1. Level 설정
logging은 level 설정을 통해 메시지의 중요도를 구분한다. 총 5개의 기본 level이 제공  WARNING이 기본 level로 지정
DEBUG	간단히 문제를 진단하고 싶을 때 필요한 자세한 정보를 기록함
INFO	계획대로 작동하고 있음을 알리는 확인 메시지
WARNING	소프트웨어가 작동은 하고 있지만, 예상치 못한 일이 발생했거나 할 것으로 예측된다는 것을 알림
ERROR	중대한 문제로 인해 소프트웨어가 몇몇 기능들을 수행하지 못함을 알림
CRITICAL    작동이 불가능한 수준의 심각한 에러가 발생함을 알림

2. logging work flow 확인
2-1. Logger: 어플리케이션 코드가 직접 사용할 수 있는 인터페이스를 제공함
logging은 Logger class의 Instance (=logger)를 선언하는 것으로 부터 시작
logger = logging.getLogger("logging_sample")
2-2. Handler: logger에 의해 만들어진 log 기록들을 적합한 위치로 보냄
handler의 종류는 15개 정도가 있는데, 가장 기본적인 것은 StreamHandler와 FileHandler
2-3. Filter: 어떤 log 기록들이 출력되어야 하는지를 결정함
2-4. Formatter: log 기록들의 최종 출력본의 레이아웃을 결정함
logging.Formatter(
  fmt = None,     # 메시지 출력 형태. None일 경우 raw 메시지를 출력.
  datefmt = None, # 날짜 출력 형태. None일 경우 '%Y-%m-%d %H:%M:%S'.
  style = '%'     # '%', '{', '$' 중 하나. `fmt`의 style을 결정.
)
cf) https://docs.python.org/3.7/library/logging.html#filter

'''

import logging, os, sys

# os.path.basename(__file__) 
# os.path.basename((sys.argv[0]) 
# os.path.split(sys.argv[0])[1]
logger = logging.getLogger(os.path.split(sys.argv[0])[1])
logger.setLevel(logging.DEBUG)

# file log handler
fh = logging.FileHandler("./basic/logs/"+ os.path.split(sys.argv[0])[1] + ".log")
fh.setLevel(logging.INFO)
# console log handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formmater
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
logger.debug("this is debugging")
logger.info("this is info")
logger.warning("this is warning")
logger.error("this is error")
