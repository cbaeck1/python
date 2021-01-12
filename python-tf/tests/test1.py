import numpy as np

a = np.array([[-3.8892615], [-4.8355017]])
print(a, a.shape)

b = a.reshape(2, order='C')
print(b, b.shape)
print(b.tolist())

