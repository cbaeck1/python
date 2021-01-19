'''
다른 경로에 있는 파일을 import 하는 방법

1. 동일 경로 파일
실행파일(path1.py)과 동일한 경로에 있는 python 파일들은 현재 경로를 의미하는 .를 사용하여 import
from . import my_module

$ tree
.
├── my_module.py
└── main.py

2. 하위 경로 파일
하위 경로의 파일은 from 하위 폴더 처럼 폴더를 지정해주어 import
from subdir import my_module

$ tree
.
├── subdir
│   └── my_module.py
└── main.py

3. 상위 경로 파일
os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
from . import library

$ tree
.
├── main
│   └── main.py
└── library.py

4. 다른 경로의 파일
import sys
sys.path.append(다른 경로의 파일)

from . import library



'''