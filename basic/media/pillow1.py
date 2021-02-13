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
 
# 이미지 크기 출력
print(im.size)
 
# 이미지 JPG로 저장
# cannot write mode RGBA as JPEG
# im.save('python.jpg')
im = im.convert('RGB')
im.save('basic/media/python.jpg')

# Thumbnail 이미지 생성
size = (64, 64)
im.thumbnail(size)  
 
im.save('basic/media/python-thumb.jpg')

# 이미지 부분 잘라내기
cropImage = im.crop((100, 100, 150, 150))
cropImage.save('basic/media/python-crop.jpg')

# 크기를 600x600 으로
img2 = im.resize((600, 600))
img2.save('basic/media/python-600.jpg')
 
# 90도 회전
img3 = im.rotate(90)
img3.save('basic/media/python-rotate.jpg')

# 
blurImage = im.filter(ImageFilter.BLUR)
 
blurImage.save('basic/media/python-blur.png')



