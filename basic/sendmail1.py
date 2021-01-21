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
message.add_alternative('''
<h1> 업무자동화 - 이메일 자동으로 보내기 </h1>
''', subtype='html')

with smtplib.SMTP_SSL('smtp.naver.com', 465) as server:
    server.ehlo()
    server.login(userid, password)
    server.send_message(message)
print('이메일을 발송했습니다.')