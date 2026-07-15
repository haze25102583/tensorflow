import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np                                   # 행렬 연산에 특화

#1. 데이터
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train = np.array([1, 2, 3, 4, 5, 6, 7, ])         # 리스트의 제일 끝 , -> 뒤에 더 올 수도 있음
# x_train = np.array([[1, 2, 3, 4, 5, 6, 7, ]])     # 도 가능 -> 열중요 행무시
y_train = np.array([1, 2, 3, 4, 5, 6, 7, ])

x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

print(x_train.shape, x_test.shape)        # (7,) (3,)
print(y_train.shape, y_test.shape)        # (7,) (3,)

# exit()

#2. 모델구성
model = Sequential()

# hidden layer, 하이퍼 파라미터 튜닝
model.add(Dense(200, input_shape=(1,)))     # (None, 1)
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
model.fit(x_train, y_train, epochs=100, )       # batch 단위의 데이터 : (목적) 성능 향상, 데이터 과적합 방지

#4. 평가, 예측
# *** 목적 : 최소의 loss, 최적의 weight ***
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)

result = model.predict(np.array([11]))
print("11의 예측값 :  ", result)
