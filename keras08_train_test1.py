import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np




# 1. 데이터 전처리
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Training data(70%), test data(30%)                # 데이터 과적합(overfit) 해결, 성능 향상

x_train = np.array([1, 2, 3, 4, 5, 6, 7, ])         # , : 앞으로 데이터가 더 올 것 -> 명시적, 문법적 허용
# x_train = np.array([[1, 2, 3, 4, 5, 6, 7, ]]) 도 가능   ->    열 중요(feature = attribute = 특성 = 속성)
x_test = np.array([8, 9, 10, ])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, ])
y_test = np.array([8, 9, 10, ])

print(x_train.shape, x_test.shape)                  # (7,) (3,)
print(y_train.shape, y_test.shape)                  # (7,) (3,)




# 2. 모델 구성
model = Sequential()

# hidden layer  ->  Hyperparameter Tuning 
model.add(Dense(1, input_shape=(1, )))               # 0차원 데이터 : input_shape=(feature, )
                                                     # (None, 1)
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(300))
model.add(Dense(1))





# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100)




# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                               # loss =  0.0005944736767560244

result = model.predict(np.array([11]))
print("11의 예측값 : ", result)                       # 11의 예측값 :  [[11.026599]]
