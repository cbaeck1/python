
import matplotlib as mpl
import matplotlib.pyplot as plt
  
import matplotlib.font_manager as fm
fontpath = 'c:/Windows/Fonts/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)

font_name = fm.FontProperties(fname=fontpath).get_name()
mpl.rc('font', family=font_name)

plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()


import numpy as np
import pylab as pl
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import cv2
from IPython.display import clear_output

from lucid.misc.io import showing as show

import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import tensorflow as tf
tf.enable_eager_execution()
def gen_points(s, font_size=42):
  #font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', font_size)
  font = PIL.ImageFont.truetype('c:/Windows/Fonts/malgun.ttf', font_size)
  w, h = font.getsize(s)
  im = PIL.Image.new('L', (w, h))
  draw  = PIL.ImageDraw.Draw(im)
  draw.text((0, 0), s, fill=255, font=font)
  im = np.uint8(im)
  y, x = np.float32(im.nonzero())
  pos = np.column_stack([x, y])
  if len(pos) > 0:
    pos -= (w/2, h/2)
    pos /= font_size
  return pos
# along the lines of
# https://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/python/optimaltransp_6_entropic_adv.ipynb

@tf.function
def pdist(x, y):
  dx = x[:, None, :] - y[None, :, :]
  return tf.reduce_sum(tf.square(dx), -1)

@tf.function
def Sinkhorn_step(C, f):
  g = tf.reduce_logsumexp(-f-tf.transpose(C), -1)
  f = tf.reduce_logsumexp(-g-C, -1)
  return f, g

def Sinkhorn(C, f=None, niter=1000):
  n = tf.shape(C)[0]
  if f is None:
    f = tf.zeros(n, np.float32)
  for i in range(niter):
    f, g = Sinkhorn_step(C, f)
  P = tf.exp(-f[:,None]-g[None,:]-C)/tf.cast(n, tf.float32)
  return P, f, g
import seaborn as sns
colors = sns.color_palette("Paired")
VIDEO_SIZE = 512

def draw_points(p , color):
  w = VIDEO_SIZE
  img = np.zeros((w, w, 3), np.uint8)
  #img[:] = 255
  p = np.int32((w/2+p*w*0.9)*4)
  for idx, (x, y) in enumerate(p):
    cv2.circle(img, (x, y), 12, color, -1, cv2.CV_AA, shift=2)
  return img

t = np.linspace(0, 2*np.pi, 256)
x = 16*np.sin(t)**3
y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
pos0 = np.float32(np.column_stack([x*0.03, -y*0.03]))

colors = sns.color_palette("Paired")
colors = [tuple(np.array(list(color)) * 255) for color in colors]
show.image(draw_points(pos0, colors[1])/255.0)


pos = pos0.copy()
colors = sns.color_palette("Paired")
colors = [tuple(np.array(list(color)) * 255) for color in colors]

with FFMPEG_VideoWriter('out.mp4', (VIDEO_SIZE, VIDEO_SIZE), 60.0) as video:
  video.write_frame(draw_points(pos , colors[0]))
  f = None
  for idx , s in enumerate('ILOVEYOU'):
    target = gen_points(s) if s != ' ' else pos0
    for i in range(80):
      C = pdist(pos, target)/(0.01)**2
      P, f, g = Sinkhorn(C, f=f, niter=20)
      P = P.numpy()
      g = P.dot(target)*len(pos)-pos
      pos += 0.1*g
      frame = draw_points(pos , colors[idx])
      video.write_frame(frame)
    print(s, end='', flush=True)
mvp.ipython_display('out.mp4', loop=True)