# 깃허브에 '정리'되어 있는 잔디밭을 깔 것

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# *** 암기
#1. 데이터 ***
# 수가 많을수록, 원칙적으로는 훈련이 잘 될 '확률'이 높다 (가장 중요)

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5, 6])

#2. 모델구성 ***    
# y = wx+b (목적 : 최적의 weight)
# 첫 값은 랜덤. epochs값이 증가할 때마다 weight, bias값이 갱신
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련 ***
# 더 많이 훈련시킬 수록, 훈련이 잘 될 확률이 높다
model.compile(loss='mse', optimizer='adam')     # loss : 손실 (loss >= 0이며, 궁극의 loss값은 0)
model.fit(x, y, epochs=940)

#4. 평가, 예측 ***
result = model.predict(np.array([7]))
print("7의 예측값 : ", result)

