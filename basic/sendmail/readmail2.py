'''
IMAP 서버 연결
IMAP 서버 로그인
이메일 검색하기
1)폴더 선택하기
    각 튜플에는 세가지 값이 있다.
    ex. ((‘\HasNoChildren’,), ‘/’, ‘INBOX’)
    1) 폴더의 플래그 튜플
    2) 이름 문자열에서 상위폴더와 하위폴더를 구분하는데 쓰이는 문자
    3) 폴더 전체의 이름
    검색할 폴더를 선택하려면 IMAPClient 개체의 select_folder()메소드에 문자열로 폴더의 이름을 전달한다.

    선택한 폴더가 없는 경우 파이썬은 imaplib.error를 일으킨다
    readonly = True 키워드 매개변수는 그 이후로 다른 메소드를 호출할 때 이 폴더 안에 있는 이메일을 실수로 변경하거나 지우는 사고를 막아준다

검색 수행하기 : 폴더 선택하면 IMAPClient 객체의 search() 메소드로 이메일을 검색
크기 제한
이메일 내용 : ‘BODY[]’ 키는 이메일의 실제 본문에 대응된다.
    메세지의 내용은 IMAP 서버가 읽어들이기 위한 목적으로 설계된 RFC 822라는 형식이라 한다.
    각 메세지는 두 개의 키, ‘BODY[]’ 및 ‘SEQ’를 가진 사전에 저장
    pyzmail 모듈을 사용하면 해당 메세지를 볼 수 있다.
본문 가져오기

'''

import imapclient
import imaplib
# 크기 제한
imaplib._MAXLINE = 10000000
imap_obj = imapclient.IMAPClient('imap.gmail.com', ssl=True)

imap_obj.login('cbaeck1@gmail.com', 'oiudntprqfmafdta')
'cbaeck1@gmail.com authenticated (Success)'

import pprint
pprint.pprint(imap_obj.list_folders())

imap_obj.select_folder('INBOX', readonly=True)
UIDs = imap_obj.search(['FROM', 'mailer-daemon@googlemail.com'])
pprint.pprint(UIDs)

# UID 리스트를 받았다면 IMAPClient 객체의 fetch() 메소드를 호출해서 실제 이메일 내용을 가져올 수 있다.
raw_msg = imap_obj.fetch(UIDs, ['BODY[]'])
pprint.pprint(raw_msg)

import pyzmail
msg = pyzmail.Pyzmessage.factory(raw_msg[6]['BODY[]'])

# 본문 가져오기
msg.text_part != None
msg.text_part.get_payload().decode(message.text_part.charset)
msg.html_part != None
msg.html_part.get_payload().decode(message.html_part.charset)


imap_obj.logout()
