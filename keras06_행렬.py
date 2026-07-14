import numpy as np

x1 = np.array([1, 2, 3])
print("x1 = ", x1.shape)    # x1 =  (3,)
                            # => 스칼라 3개의 백터 묶음

x2 = np.array([[1, 2, 3]])
print("x2 = ", x2.shape)    # x2 =  (1, 3)
                            # => 1행 3열
                            
x3 = np.array([[1, 2, 3], [4, 5, 6]])
print("x3 = ", x3.shape)                   # x3 = (2, 3)

x4 = np.array([[1, 2], [3, 4]])            
print("x4 = ", x4.shape)                   # x4 = (2, 2)

x5 = np.array([[[[1]]], [[[2]]]])
print("x5 = ", x5.shape)
