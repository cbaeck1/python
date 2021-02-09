'''
메일 발송을 위해서는 SMTP 프로토콜을 활용

메일 발송 자동화를 위해서 네이버 메일을 활용하여 코드를 작성해본다. 
가장 먼저 네이버 메일 설정에서 SMTP 사용을 활성화한다. 
그리고 네이버 메일 설정 화면에서 메일 발송에 필요한 SMTP 서버명, 포트번호, 아이디 그리고 패스워드까지 네 가지 정보를 확인한다.






'''


import smtplib
from email.message import EmailMessage

#import getpass
#password = getpass.getpass('Password: ')
userid = 'cbaeck'
password = 'yale1004!!'

message = EmailMessage()
message['Subject'] = 'Title'
message['From'] = userid + '@naver.com'
message['To'] = 'cbaeck1@gmail.com'

message.set_content('업무자동화 - 이메일 자동으로 보내기')

with smtplib.SMTP_SSL('smtp.naver.com', 465) as server:
    server.ehlo()
    server.login(userid, password)
    server.send_message(message)

print('이메일을 발송했습니다.')