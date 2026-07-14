import numpy as np

# 0차원 데이터 : 스칼라

# 1차원 데이터 : 벡터
x1 = np.array([1, 2, 3])        # array() 함수의 parameter : object -> 대괄호 유의
print(x1.shape)                 # shape 속성 : 차원과 차원의 크기를 반환
                                # 속성이므로 함수처럼 ()가 붙지 않음
                                # (3,) 출력

# 2차원 데이터 : 행렬, Matrix                                
x2 = np.array([[1, 2, 3], [1, 2, 3]])
print(x2.shape)                 # (2, 3) 출력

# 3차원 데이터 : Tensor
x3 = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]], [[1, 2], [3, 4], [5, 6], [7, 8]], [[1, 2], [3, 4], [5, 6], [7, 8]]])
print(x3.shape)                 # (3, 4, 2) 출력 -> 면, 행, 열
