# 플라스크(Flask)는 파이썬으로 작성된 마이크로 웹 프레임워크의 하나로, 
# Werkzeug 툴킷과 Jinja2 템플릿 엔진에 기반을 둔다. BSD 라이선스이다.

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
# │   ├── about5.html
# │   ├── index6.html
# │   └── layout6.html
# ├── static/
# │   ├── profile_imgs/
# │   │   └── default.png
# │   └── layout.css
# ├── venv/
# └── app6.py
# └── site.db

# 3. layout 
# 또한 Flask 템플릿은 강력한 기능들이 있는데 그 중 하나가 바로 계층 구조를 지원한다는 점입니다.
# 하나의 웹사이트는 일반적으로 동일한 테마를 갖습니다. 
# 각각의 페이지는 동일한 구성요소를 갖는데 이를 매번 html 파일에 삽입하게 되면 구성요소를 변경할 때, 
# 모든 파일에 수정사항을 반영해야 합니다. 
# 이번 프로젝트에서는 사이트에서 사용되는 동일한 구성요소는 `layout.html`파일에 넣어두고, 
# 나머지 페이지들은 이를 상속 받아서 사용하도록 하겠습니다.
# 이를 통해 `layout.html` 파일 하나를 변경함으로써 전 페이지에 변경사항을 적용할 수 있습니다.
# 템플릿 디렉터리 하위에 `layout.html` 파일을 만듭니다.

# Jinja2 템플릿 엔진은 구문(Statement)의 경우 `{% %}`로, 표현(Expression)은 `{{ }}`로 감싸서 표시합니다. 
# 주석의 경우에는 `{# #}`을 이용해 작성할 수 있습니다.
# 파일을 보면 `{% block <이름> %}` 태그가 보이는데 이 태그는 상속받은 템플릿에서 사용할 수 있는 공간을 정의합니다.
# 상속 받은 템플릿에서 부모의 block 태그와 동일한 이름의 block 태그를 사용하면 템플릿이 렌더링 될 때, 
# 부모의 block 태그는 상속 받은 템플릿에서 작성한 코드로 대치됩니다.
# 이제 나머지 모든 템플릿은 `layout.html`을 상속받아 사용할 것입니다. 
# 그렇게 되면 모든 템플릿이 `layout.html`에 정의되어 있는 요소들 상속받게 됩니다.

# `index3.html` 파일을 보면 {% extends 'layout.html' %}` 태그가 보입니다.
# 이는 우리가 작성했던 `layout.html` 파일을 상속 받겠다는 의미입니다.
# 그리고 `{% block content %} {% endblock %}`사이에 우리가 페이지에 표현할 내용들을 작성했습니다.

# 이와 동일하게 about 페이지도 한 번 만들어 보겠습니다.
# 템플릿 디렉터리 하위에 `about.html` 파일을 만들고 다음과 같은 코드를 작성합니다.

# 4. html 에 값 전달
# 1) 템플릿에 파라미터 전송
# (1) 제목 동적 생성
# `layout.html`을 보면 `<title>`이 고정값입니다. 
# 이를 페이지마다 다르게 지정하기 위해 `{{ title }}`변수를 사용하고 페이지마다 이를 넘겨주도록 하겠습니다. 
# `layout4.html`에 `<title>`태그가 있는 곳을 다음과 같이 변경합니다.
# 메인 페이지는 제목을 그대로 두고 About 페이지만 제목을 변경해보겠습니다.

# (2) 게시물 동적 생성
# `index4.html` 파일을 보면 게시물 정보가 하드코드 되어 있는 것을 확인할 수 있습니다.
# 실제 웹 페이지는 db에서 가져온 포스트를 기반으로 보여줘야 하기 때문에 
# `app.py`에서 포스트에 대한 정보를 넘겨줄 수 있도록 작성하겠습니다.
# 게시물은 개수가 얼마든지 가능하기 때문에 템플릿상에서 동적으로 생성할 수 있도록 작성해야 합니다.
# 따라서 반복문을 사용하도록 하겠습니다.
# 하드코드된 `<article>` 블럭을 다음과 같이 수정합니다.

# 2) URL 동적 생성
# layout.html파일을 보면 `href` 속성이 `/`, `/about`과 같이 하드코드 되어 있는 것을 확인할 수 있습니다. 
# 지금이야 이러한 속성들이 문제되지 않지만 `app.py`에서 라우트가 변경되거나 
# 전체 프로젝트가 특정 홈페이지의 하위로 이동하는 경우 URL이 달라질 수 있어 제대로 된 URL을 가르키지 못할 수 있습니다.
# 따라서 이를 동적으로 생성할 수 있도록 변경해야 하는데 flask에서는 `url_for`라는 함수로 지원합니다. 
# 기본적으로 `url_for`함수는 엔드포인트 함수명을 인자로 받습니다. 
# 따라서 `/` 주소를 생성하고 싶다면 app.py에서 `/` 라우트에 대한 함수 이름을 `index`로 정의했으므로 `url_for('index')`를 사용해야 합니다.
# 만약 `/about` 주소를 생성하고 싶다면 `url_for('about')`을 사용해야 합니다.
# 또한 `url_for`함수로 `static` 폴더 내에 있는 리소스의 주소를 생성할 수도 있는데, 
# 이는 `url_for('static', filename=<파일 이름>>)`를 이용합니다.
# 이제 layout4.html 파일에서 하드 코드 된 url들을 모두 `url_for` 함수를 이용해 동적생성 하도록 변경하겠습니다.

# 5. 부트 스트랩(Bootstrap)
# 이번에는 어플리케이션의 디자인을 수정하도록 하겠습니다. 
# 기능을 변경하는 것이 아닌 디자인을 변경하는 만큼 아래 코드를 복사 붙여 넣기 하는 것을 추천합니다.
# 1) layout5.html
# 부트스트랩은 웹을 개발하기 위한 프레임워크로 손쉽게 반응형 웹을 만들 수 있습니다.
# 먼저 `layout5.html`에 홈페이지의 기본이 될 테마를 작성합니다.
# 2) css
# 웹 사이트가 제대로 보이기 위해서는 다음과 같은 스타일 시트 파일이 필요합니다. 
# 스타일 시트는 `static`폴더에 `layout.css` 파일을 하나 만듭니다.
# 3) index
# 메인 페이지를 수정하도록 하겠습니다. `index5.html`을 수정합니다.

# 6. 데이터베이스
# Python의 객체에 데이터 모델을 정의하고 이를 데이터베이스와 매핑해주는 것을 ORM(Object Relaition Model)이라고 합니다. 
# 덕분에 코드는 특정 데이터베이스에 종속되지 않고, 기본 객체 만으로 데이터를 기술할 수 있기 때문에 조금 더 OOP 스러운 코드를 작성할 수 있습니다.
# Python에서 ORM으로 많이 쓰이는 것 중 SQLAlchemy를 Flask에서 플러그인 처럼 사용하기 쉽게 만들어진 Flask-SQLALchemy를 사용

# 1) from flask_sqlahcmey import SQLAlchemy
# 2) # app`객체에 몇 가지 설정을 추가해야 합니다. 
# 먼저 어플리케이션의 시크릿 키를 추가 합니다. .
# 다음으로는 `SQLAlchemy`에서 사용할 데이터베이스의 위치를 알려줘야 합니다. 
# 마지막으로 `SQLALCHEMY_TRACK_MODIFICATIONS`의 경우에는 추가적인 메모리를 필요로 하므로 꺼두는 것을 추천합니다. 
# 설정을 완료했으면 `SQLAlchemy` 객체를 하나 만듭니다.

# 3) 사용자 모델 추가
# 프로젝트에서 사용자는 사용자 계정 이름을 갖고, 이메일과 암호가 있으며 프로필 사진을 저장할 수 있는 공간이 필요합니다.
# 먼저 사용자 데이터 모델을 나타내는 객체를 하나 선언 합니다. 
# 그리고 SQLAlchemy의 기능을 사용하기 위해 `db.Model`을 상속 받습니다. 
# 기본적으로 데이터베이스 테이블 이름은 자동으로 정의되지만 `__table_name__`을 이용해 명시적으로 정할 수 있습니다.
# 사용자 데이터 모델을 나타내는 객체를 선언했는데, 이제 여기에 모델이 갖고 있어야 하는 필드와 관련된 제약사항들을 적어줘야 합니다
# `id` 필드는 대부분의 모델에서 기본 키로 사용합니다.
# `username`, `email`, `password`, `profile_image` 필드는 문자열로 정의를 하고, 
# 최대 길이를 명시하여 공간을 절약할 수 있도록 합니다. 
# 또한 `username`과 `email` 필드는 서로 중복되지 않아야 하고, 비어있지 않아야 합니다. 
# `password` 필드의 경우에는 중복되는 것은 괜찮지만 비어있지 않아야 합니다. 
# 그리고 보안을 위해 평문으로 저장하는 것이 아니라 암호화를 해서 저장을 해야 합니다. 
# `profile_image` 필드는 이미지 데이터를 DB에 직접 저장하는 것이 아니라 파일 시스템에 저장한 다음 그 파일 이름만 저장할 예정입니다. 
# 그리고 프로필 이미지는 모든 사람이 처음부터 넣는 것은 아니기 때문에 기본 이미지 파일을 가리킬 필요가 있습니다. 
# 기본 이미지 파일은 `default.png`로 설정하도록 하겠습니다.
# 테이블의 컬럼을 만들기 위해서는 `db.Column()`을 이용합니다. 컬럼의 이름은 기본적으로 변수 이름을 사용합니다. 
# `db.Column()`은 데이터 타입에 대한 정보와 제약 조건들을 넣어줄 수 있습니다. 
# 데이터 모델 객체의 경우에도 일반적인 Python 객체처럼 `__repr__`과 같은 메소드를 사용할 수 있습니다.
# 비밀번호를 암호화된 해시로 저장 : `werkzeug.security`에 있는 `generate_password_hash`와 `check_password_hash`를 이용해 
# 비밀번호를 암호화 할 수 있습니다.
# `generate_password_hash`함수는 문자열을 암호화된 해시로 바꿔주는 역할을 합니다. 
# `check_password_hash` 함수는 함호화된 해시와 문자열을 비교해서 이 문자열이 동일한 해시를 갖는 경우 참을 반환합니다.
# `generate_password_hash` 함수로 암호화 하고 : hash = generate_password_hash('password')
# `check_password_hash` 함수를 이용해 맞는 비밀번호화 틀린 비밀번호를 넣었을 경우 결과 값을 비교했습니다.
# check_password_hash(hash, 'password')
# check_password_hash(hash, 'wrong password')

# 4) 게시물 모델 추가
# 게시물은 제목과 내용으로 이루어 질 것이고, 추가적으로 언제 게시되었는지, 누가 게시했는지에 대한 정보가 필요
# `title`은 게시물 제목을, `content`는 게시물 내용을, `date_posted`는 게시일을 나타냅니다. 
# 게시일의 경우에는 기본값을 `datetime.utcnow()`를 사용함으로써 명시적으로 게시일을 나타내지 않은 경우 
# 현재 시간을 게시일로 하도록 하였습니다. 
# 그 다음 게시자에 대한 내용을 나타내기 위해 `user` 테이블의 id를 외래키로 하는 `user_id`라는 컬럼을 만들었습니다. 
# 여기서 중요한 점은 `db.ForeignKey`는 테이블 이름을 인자로 받습니다. 
# SQLAlchemy에서 테이블 이름은 기본적으로 소문자를 사용하고 여러 단어의 조합인 경우에는 스네이크 케이스를 사용합니다.
# `User` 객체를 보면 `posts` 컬럼이 추가되어 있는 것을 확인할 수 있습니다. 
# `posts` 컬럼은 `db.relationship`를 사용하는데 이는 실제 데이터베이스에 나타나는 필드는 아닙니다. 
# 이 가상 필드는 데이터베이스를 좀 더 높은 추상화 수준에서 바라볼 수 있게 도와주는 역할을 하는데요. 
# 예를 들어 사용자를 `user`이라는 변수에 저장했다고 한다면, 
# 이 사용자가 작성했던 모든 게시물에 대한 정보는 `user.posts`를 이용해 접근할 수 있습니다. 
# `db.relationship`의 첫 번째 인자는 `db.ForeignKey`와는 다르게 객체 이름을 받습니다.
# 그리고 `backref`는 `Posts` 객체에 삽입되는 가상 필드 이름입니다. 
# 즉, 게시물을 `post`라는 변수에 저장했다고 한다면 이 게시물을 작성한 게시자를 `post.author`을 이용해 접근할 수 있음을 의미합니다. 
# 이를 통해 데이터베이스의 데이터를 Python 코드 상에서 접근할 때, 고수준의 추상화된 레벨에서 사용할 수 있습니다.

# 5) 데이터베이스 초기화 및 데이터 추가
# 데이터베이스에 어떤 내용을 체워 넣을지 구조적인 부분을 코드상으로 나타내었습니다. 
# 하지만 실제 데이터베이스에 테이블을 만들고 데이터를 넣어준 것은 아니라 데이터베이스에서 데이터를 접근하려 하면 에러를 나타낼 것입니다. 
# 따라서 데이터베이스를 초기화해 줄 필요가 있습니다.
# 먼저 프로젝트 디렉터리에서 python 터미널을 이용 `db.create_all()`으로 데이터베이스를 초기화
# from app6 import app, db, User, Post
# db.create_all() 를 실행 -> site.db 생성
# 데이터베이스를 초기화했으면 터미널을 이용하여 테스트를 위한 데이터를 작성
# user = User(username='user', email='user@blog.com', password='password')
# db.session.add(user) 
# db.session.commit() #  db.session.rollback()을 이용해 변경 사항을 취소
# post1 = Post(title='첫 번째 게시물', content='첫 번째 게시물 내용', author=user)
# post2 = Post(title='두 번째 게시물', content='두 번째 게시물 내용', author=user)
# db.session.add(post1)
# db.session.add(post2)
# db.session.commit()
# 데이터베이스에 저장한 데이터를 가져오기 위해서는 각 모델에 `query`를 이용합니다. 
# 예를 들어 전체 사용자를 가져오고 싶으면 `User.query.all()`을 입력, Post.query.all()
# 삭제 : User.query.filter_by(id=1).delete() 
#        User.query.filter(User.id == 1).delete() 

# 6) 더미 데이터 삭제
#   `db.query` 명령을 이용해 데이터베이스에 저장되어 있는 모든 게시물 데이터를 불러와 `render_template` 함수에 전달
#   `index6.html` 파일을 수정해 `holder.js`를 통해 이미지를 보여줄 공간만 마련했던 것을 실제 데이터와 연결
#   `<article>` 태그가 있는 부분에 `<img>` 태그를 보면 `src="#"`로 비어있는 것을 확인할 수 있습니다.
#   이를 `static/profile_imgs` 하위에 저장 되어 있는 프로필 사진들로 연결
# 7) 기본 프로필 이미지 저장
# `User` 모델에 기본 프로필 이미지로 `default.png`를 적어주었습니다. 
# 이 사진 파일을 저장하기 위해서 `static` 디렉터리 하위에 `profile_imgs` 디렉터리를 만들고 프로필 이미지 파일들을 저장하겠습니다. 
# 폴더를 나누는 이유는 `static` 폴더에는 프로필 사진 뿐 아니라 다양한 자바스크립트 파일, css 스타일 시트 파일 등이 저장될텐데 
# 여러 파일들이 혼합되면 정리하는데 불편하기 때문입니다.

from flask import Flask, render_template
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.config['SECRET_KEY'] = 'this is secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 사용자 모델
class User(db.Model):
    __table_name__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    profile_image = db.Column(db.String(100), default='default.png')
    # `User` 객체를 보면 `posts` 컬럼이 추가
    # `posts` 컬럼은 `db.relationship`를 사용하는데 이는 실제 데이터베이스에 나타나는 필드는 아닙니다. 
    # 이 가상 필드는 데이터베이스를 좀 더 높은 추상화 수준에서 바라볼 수 있게 도와주는 역할을 하는데요.
    # 예를 들어 사용자를 `user`이라는 변수에 저장했다고 한다면, 
    # 이 사용자가 작성했던 모든 게시물에 대한 정보는 `user.posts`를 이용해 접근할 수 있습니다. 
    # `db.relationship`의 첫 번째 인자는 `db.ForeignKey`와는 다르게 객체 이름을 받는다.
    # `backref`는 `Posts` 객체에 삽입되는 가상 필드 이름입니다. 
    # 즉, 게시물을 `post`라는 변수에 저장했다고 한다면 이 게시물을 작성한 게시자를 `post.author`을 이용해 접근할 수 있음을 의미합니다. 
    # 이를 통해 데이터베이스의 데이터를 Python 코드 상에서 접근할 때, 고수준의 추상화된 레벨에서 사용
    posts = db.relationship('Post', backref='author', lazy=True)

    def __init__(self, username, email, password, **kwargs):
      self.username = username
      self.email = email
      self.set_password(password)

    def __repr__(self):
      return f"<User('{self.id}', '{self.username}', '{self.email}')>"

    def set_password(self, password):
      self.password = generate_password_hash(password)
 
    def check_password(self, password):
      return check_password_hash(self.password, password)

# 게시물
class Post(db.Model):
    __table_name__ = 'post'
 
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), unique=True, nullable=False)
    content = db.Column(db.Text)
    date_posted = db.Column(db.DateTime, default=datetime.utcnow())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
 
    def __repr__(self):
      return f"<Post('{self.id}', '{self.title}')>"


# 현재는 db가 구성되지 않았으므로 테스트를 위해 더미 데이터를 만들고 이를 템플릿에 넘겨준다
'''
posts = [
    {
        'author': {
            'username': 'test-user'
        },
        'title': '첫 번째 포스트',
        'content': '첫 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2018-08-01', '%Y-%m-%d')
    },
    {
        'author': {
            'username': 'test-user'
        },
        'title': '두 번째 포스트',
        'content': '두 번째 포스트 내용입니다.',
        'date_posted': datetime.strptime('2018-08-03', '%Y-%m-%d')
    },
]
'''
 
@app.route('/')
@app.route('/index')
def index():
  posts = Post.query.all()
  return render_template('index6.html', posts=posts)


@app.route('/about')
def about():
  return render_template('about5.html', title='About')


app.run(host='127.0.0.1', port=5000, debug=False)

