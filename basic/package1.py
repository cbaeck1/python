'''
16. 패키지
1. 패키지란 무엇인가?
패키지(Packages)는 도트(.)를 사용하여 파이썬 모듈을 계층적(디렉터리 구조)으로 관리
# 가상의 game 패키지 예
game, sound, graphic, play는 디렉터리 이름
game/
    __init__.py
    sound/
        __init__.py
        echo.py
        wav.py
    graphic/
        __init__.py
        screen.py
        render.py
    play/
        __init__.py
        run.py
        test.py

2. 패키지 만들기
3. 패키지 안의 함수 실행하기
>>> import game.sound.echo
>>> game.sound.echo.echo_test()

4. __init__.py 의 용도
다음과 같이 echo_test 함수를 사용하는 것은 불가능하다.
>>> import game
>>> game.sound.echo.echo_test()
※ python3.3 버전부터는 __init__.py 파일이 없어도 패키지로 인식한다(PEP 420). 
   하지만 하위 버전 호환을 위해 __init__.py 파일을 생성하는 것이 안전한 방법이다
해당 디렉터리의 __init__.py 파일에 __all__ 변수를 설정하고 import할 수 있는 모듈을 정의

5. relative 패키지
1) from game.sound.echo import echo_test를 전체 경로를 입력해 사용
2) from ..sound.echo import echo_test 상대 경로를 입력해 사용





'''

import game.sound.echo
game.sound.echo.echo_test()

# NameError: name 'echo' is not defined
from game.sound import *
echo.echo_test()

from game.graphic.render import render_test
render_test()
