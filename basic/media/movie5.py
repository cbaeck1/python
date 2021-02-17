'''
클립 믹싱

'''

from moviepy.editor import *

# ... make some audio clips aclip1, aclip2, aclip3
concat = concatenate_audioclips([aclip1, aclip2, aclip3])
compo = CompositeAudioClip([aclip1.volumex(1.2),
                            aclip2.set_start(5), # start at t=5s
                            aclip3.set_start(9)])

