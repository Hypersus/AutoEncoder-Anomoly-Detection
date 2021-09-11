import numpy as np
rng = np.random.RandomState(42)
# print(rng)
# X = rng.random_sample((100, 1))
X=np.array([1,2,3,4,5])

print(X.reshape(X.size,1))
