import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 1. 데이터 전처리
# x는 [1,7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]
# y는 [1, 2, 3, 4, 5, 6]
x = np.array([[1, 7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])       # array() 함수의 매개변수 type : 객체
                                                                        # 여러 개의 백터(1차원) -> []로 묶어 매트릭스(2차원) '객체'로
                                                                        
                                                                        # 0차원 : 스칼라,   1차원 : 백터,   2차원 : 매트릭스    3차원 : tensor
y = np.array([1, 2, 3, 4, 5, 6])

# 열의 개수가 중요 -> 열 : 특성(feature)
print(x.shape)                                      # (6, 2) 출력
print(y.shape)                                      # (6, ) 출력
exit()                                              # 프로세스 종료

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(2, )))            # input_shape는 매개변수의 키워드인자가 들어가야하므로 input_shape=(n차원 데이터) 형식으로 지정
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([[7, 13]]))         # x = [7, 13]일 때, y의 예측값
print("result = ", result)
