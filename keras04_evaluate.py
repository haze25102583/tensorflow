# 로스 : 모수 - 예측값 (-> 평가기준. 상대적이나 가장 이상적인 값은 0


import tensorflow as tf             # tensorflow를 임포트
print(tf.__version__)               # tensorflow의 버전을 출력

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense           # Dense : 얼마나 촘촘한 지
import numpy as np                                  # 행렬 연산에 특화

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6])    # 두개 이상이 데이터는 묶어서 '리스트'로
y = np.array([1, 2, 3, 4, 5, 6])    # numpy 형태로 만듦

#2. 모델구성
# depth, node가 증가하면, 속도가 느려짐 (GPU로 보완 가능)
model = Sequential()                 # 연속적인 모델
model.add(Dense(40, input_dim=1))    # dim = dimension(차원)
#         Dense(output dimension, input_dim=차원의 수)                  hidden layer -> 최적의 depth값, node 개수는 알 수 없다
model.add(Dense(90))                                                 # 1에서 있단 히든부분인 input_dim 생략
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(900))
model.add(Dense(2200))
model.add(Dense(1))



#3. 컴파일, 훈련
# 목적 : 최소의 loss (**1순위**), 최적의 weight
# loss = (모수) - (예측값) = y-y'                       # 상대적
# 순전파 -> 역전파 -> 순전파를 반복하며 weight, bias (y = wx+b) 값을 계속해서 갱신해 최적의 값을 찾는다.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)                           # 통상적으로 epochs(횟수)가 많을 수록 예측 성능이 향상되나 최적의 epochs 값은 알 수 없다 (hidden layer)
#                                                       데이터 과적합 : 데이터에 지나치게 최적화되어 예측 성능이 떨어지는 것


#4. 평가, 예측
# *** 목적 : 최소의 loss, 최적의 weight ***
loss = model.evaluate(x, y)
print("loss= ", loss)

result = model.predict(np.array([7]))
print("7의 예측값 : ", result)
