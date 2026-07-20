import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # 행렬 연산에 특화

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])    # 두개 이상은 묶어서 리스트로
y = np.array([1, 2, 3, 4, 5, 6])    # 여러개의 데이터를 묶어서 numpy 형태로 만듦

#2. 모델구성
# depth, node가 증가하면, 속도가 느려짐 (GPU로 보완 가능)
model = Sequential()
model.add(Dense(40, input_dim=1))    # dim = dimension(차원)
#         Dense(output dimension, input_dim=차원의 수)
model.add(Dense(90, input_dim=40))
model.add(Dense(10, input_dim=90))
model.add(Dense(30, input_dim=10))
model.add(Dense(10, input_dim=30))
model.add(Dense(40, input_dim=10))
model.add(Dense(100, input_dim=40))
model.add(Dense(900, input_dim=100))
model.add(Dense(2200, input_dim=900))
model.add(Dense(1, input_dim=2200))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
result = model.predict(np.array([7]))
print("7의 예측값 : ", result)
