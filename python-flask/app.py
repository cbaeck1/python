# 1. 어플리케이션 인스턴스(Application Instance)를 생성
# 웹 서버는 클라이언트로부터 수신한 모든 리퀴스트를 이 오브젝트에서 처리하는데 
# 이 때 웹 서버 게이트 웨이 인터페이스(WSGI)라는 프로토콜을 사용
# `Flask` 클래스로부터 간단한 인스턴스를 만들어서 웹 서비스를 제공합니다. 
# 여기서 `Flask` 클래스에 전달된 `__name__` 파라미터는 Flask 어플리케이션을 구분하기 위한 구분자로 사용되며 
# 거의 모든 경우에 `__name__`를 전달해주면 완벽하게 동작합니다.

# 2. 라우트
# 웹 브라우저와 같은 클라이언트는 웹 서버에 리퀘스트를 전송하여 플라스크 어플리케이션 인스턴스에 교대로 전송합니다.
# 어플리케이션 인스턴스는 각 URL 리퀴세트 실행을 위해 어떤 코드가 필요한지 알아야 하며, 
# 따라서 URL을 파이썬 함수에 매핑시켜야 하는데 이 URL을 처리하는 함수를 라우트라고 합니다.
# 그 다음 함수에 `@app.route` 데코레이터가 달려있습니다. 

# 3. 뷰 함수
# 이 데코레이터는 URL과 함수를 이어주는 역할을 하는데, 위의 경우에는 라우트가 `/`와 `/index`로 두 개 달려 있습니다. 
# 따라서 웹 브라우저가 `/` 또는 `/index` 페이지를 요청하게 되면 
# 어플리케이션에서 `index`함수를 호출하여 그 결과를 웹 브라우저에 전달하게 됩니다.

# 참고 
# 무료호스팅 : https://ide.goorm.io/
# https://ndb796.tistory.com/134
# https://has3ong.tistory.com/300

from flask import Flask

app = Flask(__name__)
 
@app.route('/')
@app.route('/index')
def index():
    return 'Hello Flask!'


# 추가
@app.route('/about')
def about():
  return 'About 페이지'


app.run(host='127.0.0.1', port=5000, debug=False)