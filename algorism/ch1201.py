import sys
from PyQt5.QtWidgets import *

app = QApplication(sys.argv)
print(sys.argv)
label = QLabel("Hello PyQt")
label.show()

print("Before event loop")
app.exec_()

# 한 가지 중요한 점은 파이썬 코드는 파일의 위에서부터 아래쪽으로 순차적으로 실행되는데 app.exec_ 앞쪽의 코드는 실행됐지만 
# 그다음 줄의 코드는 아직 실행되지 않았다는 점입니다. 이는 exec_ 메서드가 호출되면서 이벤트 루프가 생성됐고 
# 이 때문에 프로그램이 계속해서 이벤트 루프 안에서 실행되고 있기 때문
print("After event loop")
