# 플라스크(Flask)는 파이썬으로 작성된 마이크로 웹 프레임워크의 하나로, Werkzeug 툴킷과 Jinja2 템플릿 엔진에 기반을 둔다. BSD 라이선스이다.

from flask import Flask

app = Flask(__name__)

@app.route('/') #http://127.0.0.1:5000
def index():
    return '루트 페이지'

@app.route('/company') #http://127.0.0.1:5000/company
def company():
    return '회사 소개'

@app.route('/company/history') #http://127.0.0.1:5000/company/history
def company_history():
    return '회사 연혁'


app.run(host='127.0.0.1', port=5000, debug=False)