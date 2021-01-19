'''
1. 현재 날짜, 시간 가져오기
다음과 같이 datetime 모듈을 이용하여 현재 시간을 가져올 수 있습니다.

from datetime import datetime

current_time = datetime.now()
print(current_time)
Output: 2021-01-02 22:41:10.181347

2. 현재 시간 가져오기
날짜는 필요없고, 시간만 필요하다면 now().time()으로 시간 정보만 가져올 수 있습니다.

from datetime import datetime

current_time = datetime.now().time()
print(current_time)
Output: 22:57:11.070602

3.1 원하는 format으로 출력
다음과 같이 strftime()을 이용하여 원하는 형태로 출력할 수 있습니다.

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(current_time)
Output: 22:41:10

3.2 year, month, day 정보 가져오기
datetime 객체는 year, month, day 등의 변수를 갖고 있습니다. 이것으로 특정한 날짜, 시간 정보를 가져올 수 있습니다.

from datetime import datetime

current_time = datetime.now()
print(current_time)
print(f'Year : {current_time.year}')
print(f'Month : {current_time.month}')
print(f'Day : {current_time.day}')
print(f'Hour : {current_time.hour}')
print(f'Minute : {current_time.minute}')
print(f'Second : {current_time.second}')
Output: 2021-01-02 22:41:10.181418
Year : 2021
Month : 1
Day : 2
Hour : 22
Minute : 41
Second : 10

3.3 NewYork의 시간 정보 가져오기
다음과 같이 pytz 모듈을 이용하여 특정 지역의 Timezone 정보를 가져올 수 있습니다. datetime.now()에 Timezone을 인자로 전달하면 해당 지역의 시간이 리턴됩니다. 리턴된 datetime 객체에서 필요한 시간 정보를 가져오면 됩니다.

from datetime import datetime
import pytz

tz = pytz.timezone('America/New_York')
cur_time = datetime.now(tz)
simple_cur_time = cur_time.strftime("%H:%M:%S")
print(f'NY time: {cur_time}')
print(f'NY time: {simple_cur_time}')
Output: NY time: 2021-01-02 08:41:10.200024-05:00
NY time: 08:41:10

3.4 Timezone을 가져올 때 사용되는 String
America/New_York와 같이, Timezone을 가져올 때 사용되는 문자열은 GitHub - heyalexej
( https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568#file-pytz-time-zones-py )에 소개되어있습니다. 
이 내용을 참고하시면 좋을 것 같습니다.

for tz in pytz.all_timezones:
    print(tz)

'''

from datetime import datetime
import pytz

# 
current_time = datetime.now()
print(current_time)

#
current_time = datetime.now().time()
print(current_time)

# 
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(current_time)

#
current_time = datetime.now()
print(current_time)
print(f'Year : {current_time.year}')
print(f'Month : {current_time.month}')
print(f'Day : {current_time.day}')
print(f'Hour : {current_time.hour}')
print(f'Minute : {current_time.minute}')
print(f'Second : {current_time.second}')

#
tz = pytz.timezone('America/New_York')
cur_time = datetime.now(tz)
simple_cur_time = cur_time.strftime("%H:%M:%S")
print(f'NY time: {cur_time}')
print(f'NY time: {simple_cur_time}')

#
for tz in pytz.all_timezones:
    print(tz)
