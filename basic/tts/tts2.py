# -*- coding: utf-8 -*-
from gtts import gTTS
from pydub import AudioSegment
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.io.wavfile
import numpy as np

__author__ = 'info-lab'

text_kr = ""
with open("basic/tts/input.txt", "r", encoding='utf-8') as f:
    for line in f:
        text_kr += line.replace(".", "          ")

print(text_kr)
tts = gTTS(
    text=text_kr,
    lang='ko', 
    slow=False
)
tts.save('ex_ko.mp3')

