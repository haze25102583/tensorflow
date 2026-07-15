import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 1. 데이터 전처리
# x은 [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]
# y은 [1, 2, 3, 4, 5, 6]
x = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).T     # .T 와 x=x.T 동일
                                                                # 코드의 간결화
y = np.array([1, 2, 3, 4, 5, 6])

# print(x.shape)      # (6, 2)
# print(y.shape)      # (6,)
# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(2,)))         # keras모델의 1차원 데이터의 형태 : (특성수, )
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')     # 인덱스의 대괄호 잊지 말기!!
model.fit(x, y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([[7, 13]]))
print("result = ", result)
