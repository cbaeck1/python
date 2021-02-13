import os, sys
print(os.path.dirname(__file__))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + '/manimlib' )
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + '/manimlib')
from manimlib.imports import *

# 매님의 Scene 클래스를 상속받는 Hello_World 클래스 생성
class Hello_World(Scene):
    # 자동 실행되는 메서드
    def construct(self):
        #TextMobject는 문자열을 표현하는 매님의 대표적인 객제
        #text=TextMobject("Hello World")
        text = Text("Hello World")
        # Scene 클래스에는 객체를 화면에 표출하는 여러 메이메이션들이 있고,
        # Write 는 화면에 글자를 펜으로 쓰듯이 보여주는 애니메이션
        self.play(Write(text))
        self.wait()

# hello = Hello_World()