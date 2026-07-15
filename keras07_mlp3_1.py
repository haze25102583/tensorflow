import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터 전처리
# x은 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# y은 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape)      # .T전 -> (3, 10)     ----->      (10, 3)
print(y.shape)      # .T전 -> (10,)

# 2. 모델 구성
model = Sequential()                                 # Sequential : 순차적인 모델에 대한 클래스
model.add(Dense(100, input_shape=(3,)))              # input_shape=(특징 수,)
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([[10, 1.3, 0]]))    # 목표치 : 10.0
print("result = ", result)                          # 출력 : result =  [[9.997517]]
