'''
https://pillow.readthedocs.io/en/5.1.x/handbook/overview.html

Pillow와 PIL은 동일한 환경에서 공존 할 수 없습니다. Pillow를 설치하기 전에 PIL을 제거하십시오.
pip install Pillow

외부 라이브러리
    libjpeg 는 JPEG 기능을 제공
    zlib 는 압축 된 PNG에 대한 액세스를 제공
    libtiff 는 압축 된 TIFF 기능을 제공
    libfreetype 은 유형 관련 서비스를 제공
    littlecms 는 색상 관리를 제공
    libwebp 는 WebP 형식을 제공
    tcl / tk 는 tkinter 비트 맵 및 사진 이미지에 대한 지원을 제공
    openjpeg 는 JPEG 2000 기능을 제공
    libimagequant 는 향상된 색상 양자화를 제공
    libraqm 은 복잡한 텍스트 레이아웃 지원을 제공


이미지 아카이브 : 축소판을 만들고, 파일 형식 간을 변환하고, 이미지를 인쇄하는 등의 작업을 수행
이미지 디스플레이 : Tk PhotoImage및 BitmapImage인터페이스가 포함, 
이미지 처리 : 포인트 연산, 내장 된 컨볼 루션 커널 세트를 사용한 필터링, 색상 공간 변환을 포함한 기본 이미지 처리 기능, 이미지 크기 조정, 회전 및 임의의 아핀 변환

개념 : Python Imaging Library는 래스터 이미지를 처리합니다 . 즉, 픽셀 데이터의 사각형입니다.
1) 밴드 : 빨간색, 녹색, 파란색 및 알파 투명도 값에 대해 'R', 'G', 'B'및 'A'밴드가 있다
2) 모드 : 이미지의 화소의 타입과 깊이를 정의
    1 (1 비트 픽셀, 흑백, 바이트 당 1 픽셀로 저장 됨)
    L (8 비트 픽셀, 흑백)
    P (색상 팔레트를 사용하여 다른 모드에 매핑 된 8 비트 픽셀)
    RGB (3x8 비트 픽셀, 트루 컬러)
    RGBA (4x8 비트 픽셀, 투명 마스크가있는 트루 컬러)
    CMYK (4x8 비트 픽셀, 색상 분리)
    YCbCr (3x8 비트 픽셀, 컬러 비디오 형식)
    이것은 ITU-R BT.2020, 표준이 아닌 JPEG를 나타냅니다.
    LAB (3x8 비트 픽셀, L * a * b 색 공간)
    HSV (3x8 비트 픽셀, 색조, 채도, 값 색 공간)
    I (32 비트 부호있는 정수 픽셀)
    F (32 비트 부동 소수점 픽셀)
3) 크기
4) 좌표계
5) 팔레트
6) 정보 : info
7) 필터 : NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS









'''


from PIL import Image, ImageDraw, ImageFont
# get an image
base = Image.open('Pillow/Tests/images/hopper.png').convert('RGBA')

# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))

# get a font
fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
# get a drawing context
d = ImageDraw.Draw(txt)

# draw text, half opacity
d.text((10,10), "Hello", font=fnt, fill=(255,255,255,128))
# draw text, full opacity
d.text((10,60), "World", font=fnt, fill=(255,255,255,255))

out = Image.alpha_composite(base, txt)

out.show()