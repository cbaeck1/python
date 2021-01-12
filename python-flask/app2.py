# `Flask` 클래스로부터 간단한 인스턴스를 만들어서 웹 서비스를 제공합니다. 
# 여기서 `Flask` 클래스에 전달된 `__name__` 파라미터는 Flask 어플리케이션을 구분하기 위한 구분자로 사용되며 
# 거의 모든 경우에 `__name__`를 전달해주면 완벽하게 동작합니다.

# 그 다음 함수에 `@app.route` 데코레이터가 달려있습니다. 
# 이 데코레이터는 URL과 함수를 이어주는 역할을 하는데, 위의 경우에는 라우트가 `/`와 `/index`로 두 개 달려 있습니다. 
# 따라서 웹 브라우저가 `/` 또는 `/index` 페이지를 요청하게 되면 
# 어플리케이션에서 `index`함수를 호출하여 그 결과를 웹 브라우저에 전달하게 됩니다.

# 1. HTML
# 웹 브라우저는 HTML을 읽어 해석해서 사용자에게 보여주기 때문에 다양한 기능을 추가하기 위해서는 모든 컨텐츠를 HTML의 형태로 만들어서 전달

# 2. htlm 
# 1) index2.html
# 하지만 이렇게 긴 HTML 코드 블럭을 어플리케이션 코드 내에 삽입한다는 것은 가독성이 너무 떨어져서 유지보수하기 어렵게 만듭니다. 
# 어플리케이션의 규모가 커지면서 점점 더 많은 기능들을 제공하게 되고, 점점 더 많은 HTML 파일들을 처리해야 하는데 
# 이를 모두 하나의 코드에 삽입한다면 관리하기 어렵게 됩니다. 
# 따라서 구조적으로 사용자에게 보여질 부분(HTML)과 실제로 처리하는 부분(어플리케이션 코드)을 나눌 필요가 있습니다.

# 2) 템플릿
# Flask에서는 보여지는 부분과 처리하는 부분을 나누기 위해 템플릿이라는 기능을 제공합니다.
# 템플릿에 사용되는 파일들은 `templates` 디렉터리에 저장되며 일반적으로 .html 파일을 사용합니다. 
# 또한 css 같은 파일들은 `static` 디렉터리에 저장합니다. 
# 어플리케이션 상에서 이러한 html 파일들을 렌더링할 수 있도록 Flask에서는 `render_template`를 제공하는데요. 
# Jinja2 템플릿 엔진을 사용해서 html 문서 내에 코드 조각들을 삽입하여 웹 페이지를 동적으로 생성할 수 있습니다.
# 템플릿 기능을 사용하기 위해 먼저 프로젝트 디렉터리 하위에 `templates`과 `static`디렉터리를 생성합니다.
# 
# flask_blog/
# ├── templates/
# ├── static/
# ├── venv/
# └── app.py

from flask import Flask, render_template
 
app = Flask(__name__)
 
@app.route('/')
@app.route('/index')
def index():
  return render_template('index2.html')


# 추가
@app.route('/about')
def about():
  return 'About 페이지'


app.run(host='127.0.0.1', port=5000, debug=False)