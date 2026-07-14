# 로스 : 모수 - 예측값 (-> 평가기준. 상대적이나 가장 이상적인 값은 0
# 데이터 과적합 : 너무 잘맞음


import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # 행렬 연산에 특화

#1. 데이터
# x = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])           # (2, 6)
x = np.array([[1,7], [2, 8], [3, 9], [4, 10], [5, 11], [6, 12]])    # (6, 2)
y = np.array([1, 2, 3, 4, 5, 6])

#2. 모델구성
# depth, node가 증가하면, 속도가 느려짐 (GPU로 보완 가능)
model = Sequential()
#model.add(Dense(40, input_dim=2))    # dim = dimension(차원)
#         Dense(output dimension, input_dim=차원의 수)
model.add(Dense(200, input_shape=(2,)))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(900))
model.add(Dense(2200))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)       # batch 단위의 데이터 : (목적) 성능 향상, 데이터 과적합 방지

#4. 평가, 예측
# *** 목적 : 최소의 loss, 최적의 weight ***
loss = model.evaluate(x, y)
print("loss= ", loss)

result = model.predict(np.array([[7, 13]])) #(1, 2)
print("7의 예측값 : ", result)
