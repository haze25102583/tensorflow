import tensorflow as tf
# print(tf.__version__)       # 2.14.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 1. 데이터 전처리
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)       # batch 단위의 데이터
                                                # 다량의 데이터가 가능. 데이터 과적합 방지. 성능 향상. 
                                                # 시행 3회 : 1/3 > 2/3 > 3/3
    
# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss = ", loss)
result = model.predict(np.array([7]))
print("result = ", result)                      # 7의 예측값 : result =  [[6.999337]] 출력
