import sys, numpy as np
mod = sys.modules[__name__]
for idx in range(5) :
    setattr(mod, 'object_{}'.format(idx),  idx )

print(object_0)

# 25251 -> 25***
v5 = '25251'
print(v5)
v5 = v5[:2]+"***"
print(v5)

v1 = 'v5[:2]+"***"'
print(v1)
abc = 'v10'
setattr(mod, abc, v1)
print(v10)
v5 = eval(v10)
print(v5)

