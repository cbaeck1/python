'''
 pip install Pillow

1. 이미지 파일 읽고 쓰기

2. Thumbnail 이미지 생성

3. 이미지 부분 잘라내기
crop() 
copy() : 이미지를 복사
paste() : 이미지를 붙여넣기

4. 이미지 회전 및 Resize
5. 이미지 필터링

'''


from PIL import Image, ImageFilter
 
# 이미지 열기
im = Image.open('basic/python.png')
 
# 이미지 크기 출력 : 너비와 높이 (픽셀 단위)를 포함하는 2-튜플
print(im.size)

# format 속성은 이미지의 소스를 식별
# mode 특성 수 및 이미지의 밴드의 이름과 같은 픽셀 타입과 깊이를 정의
# gray scale 이미지의 경우 "L"(휘도), true color 이미지의 경우 "RGB", free press 이미지의 경우 "CMYK"입니다.

# 이미지 JPG로 저장
# cannot write mode RGBA as JPEG
# im.save('python.jpg')
im = im.convert('RGB')
im.save('basic/media/python.jpg')

# Thumbnail 이미지 생성
size = (64, 64)
imThumbnail = im.thumbnail(size)  
imThumbnail.save('basic/media/python-thumb.jpg')

# 이미지 잘라 내기, 붙여 넣기 및 병합
cropImage = im.crop((100, 100, 150, 150))
cropImage.save('basic/media/python-crop.jpg')

# 크기를 600x600 으로
img2 = im.resize((600, 600))
img2.save('basic/media/python-600.jpg')
 
# 90도 회전
img3 = im.rotate(90)
img3.save('basic/media/python-rotate.jpg')

# 하위 직사각형을 처리하고 다시 붙여 넣기
# 영역을 다시 붙여 넣을 때 영역의 크기는 지정된 영역과 정확히 일치해야합니다
region = im.transpose(Image.ROTATE_180)
box = (100, 100, 400, 400)
imRegion = im.paste(region, box)
imRegion.save('basic/media/python-Region.jpg')

# 이미지 롤링
# cropping is a lazy operation.
# paste 메서드는 투명 마스크를 선택적 인수로 사용
def roll(image, delta):
    """Roll an image sideways."""
    xsize, ysize = image.size

    delta = delta % xsize
    if delta == 0: return image

    part1 = image.crop((0, 0, delta, ysize))
    part2 = image.crop((delta, 0, xsize, ysize))
    part1.load()
    part2.load()
    image.paste(part2, (0, 0, xsize-delta, ysize))
    image.paste(part1, (xsize-delta, 0, xsize, ysize))

    return image

# 밴드 분할 및 병합
r, g, b = im.split()
im = Image.merge("RGB", (b, g, r))

# 이미지 향상
# filter
blurImage = im.filter(ImageFilter.BLUR)
blurImage.save('basic/media/python-blur.png')

# 이미지 파일 식별 : 이미지 파일 집합을 빠르게 식별
import sys
for infile in sys.argv[1:]:
    try:
        with Image.open(infile) as im:
            print(infile, im.format, "%dx%d" % im.size, im.mode)
    except IOError:
        pass


# 이미지 시퀀스

# 포스트 스크립트 인쇄

# 이미지 읽기에 대한 추가 정보

# 디코더 제어

