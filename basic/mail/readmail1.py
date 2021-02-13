import os
import email
import imaplib
import configparser
from email.mime.text import MIMEText
from datetime import datetime
 
# 문자열의 인코딩 정보 추출 후, 문자열, 인코딩 얻기
def find_encoding_info(txt):
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding
 
# Email 설정정보 불러오기
config = configparser.ConfigParser()
config.read('basic/mail/config.ini')
 
# gmail imap 세션 생성
session = imaplib.IMAP4_SSL('imap.gmail.com', 993)
 
# 로그인
id = config['Gmail']['ID']
password = config['Gmail']['Password']
print(id, password)

session.login(id, password)
 
# 받은편지함
session.select('Inbox')
# m.select("[Gmail]/All Mail")
 
# 받은 편지함 내 모든 메일 검색
# UIDs = imap_obj.search(['FROM', 'mailer-daemon@googlemail.com'])
result, data = session.search(None, 'ALL')
if result == 'OK': 
    # 여러 메일 읽기
    all_email = data[0].split()
 
    for mail in all_email:
        # #fetch 명령을 통해서 메일 가져오기 (RFC822 Protocol)
        result, data = session.fetch(mail, '(RFC822)')
        #raw_email = data[0][1]
        #raw_email_string = raw_email.decode('utf-8')
        #email_message = email.message_from_string(raw_email_string)

        email_message = email.message_from_bytes(data[0][1])
        while email_message.is_multipart():
            email_message = email_message.get_payload(0)

        content = email_message.get_payload(decode=True)
        print(content)
        #f = open("email_" + str(datetime.now()) + ".html", "wb")
        #f.write(content)
        #f.close()

        # 메일 정보
        print('From: ', email_message['From'])
        if email_message['From'] != 'mailer-daemon@googlemail.com' :
            continue

        print('Sender: ', email_message['Sender'])
        print('To: ', email_message['To'])
        print('Date: ', email_message['Date'])
    
        subject, encode = find_encoding_info(email_message['Subject'])
        print('Subject', subject)
    
        message = ''    
        print('[Message]')
        #메일 본문 확인
        if email_message.is_multipart():
            for part in email_message.get_payload():
                if part.get_content_type() == 'text/plain':
                    bytes = part.get_payload(decode=True)
                    encode = part.get_content_charset()
                    message = message + str(bytes, encode)
        else:
            if email_message.get_content_type() == 'text/plain':
                bytes = email_message.get_payload(decode=True)
                encode = email_message.get_content_charset()
                message = str(bytes, encode)
        print(message)
        
        #첨부파일 존재 시 다운로드
        # for part in email_message.walk():
        #     if part.get_content_maintype() == 'multipart':
        #         continue
        #     if part.get('Content-Disposition') is None:
        #         continue
        #     file_name = part.get_filename()
    
        #     if bool(file_name):
        #         file_path = os.path.join('C:/Temp/', file_name)
        #         if not os.path.isfile(file_path):
        #             fp = open(file_path, 'wb')
        #             fp.write(part.get_payload(decode=True))
        #             fp.close()
        # else:
        #     continue
 
session.close()
session.logout()

