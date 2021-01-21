import os
import smtplib
from email.message import EmailMessage
from email.mime.application import MIMEApplication

#import getpass
#password = getpass.getpass('Password: ')
userid = 'cbaeck'
password = 'yale1004!!'

message = EmailMessage()
message['Subject'] = 'Title'
message['From'] = userid + '@naver.com'
message['To'] = 'cbaeck1@gmail.com'

message.add_alternative('''
<h1> 업무자동화 - 이메일 자동으로 보내기 </h1>
''', subtype='html')

# file attach
filepath = './basic/브로드캐스트.png'
with open(filepath, 'rb') as f:
    filename = os.path.basename(filepath)
    img_data = f.read()
    part = MIMEApplication(img_data, name=filename)
    message.attach(part)

with smtplib.SMTP_SSL('smtp.naver.com', 465) as server:
    server.ehlo()
    server.login(userid, password)
    server.send_message(message)
    
print('이메일을 발송했습니다.')