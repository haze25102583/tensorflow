# 11_2을 es 적용해서 성능 향상
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_breast_cancer                                # 교육용 데이터 제공

import numpy as np
import pandas as pd
import time


# 1. 데이터
datasets = load_breast_cancer()

print(datasets.feature_names)

x = datasets.data
y = datasets.target


print(x.shape, y.shape)                 # (569, 30) (569,)
print(np.unique(y, return_counts=True))


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,        # default
                                                    shuffle=True,            # default
                                                    random_state=100,
                                                    )

print(x_train.shape, x_test.shape)  # (331, 10) (111, 10)
print(y_train.shape, y_test.shape)  # (331,) (111,)


# 2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu',input_shape=(30,)))
model.add(Dense(28, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=['accuracy'],
              )

es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',
    patience=100,                
    restore_best_weights=True,
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=128, batch_size=64, verbose=3,
          validation_split=0.2,
          callbacks = [es])                   
end_time = time.time()

print("=========================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)           # binary_crossentropy 값 나옴
print("loss = ", loss)                      
y_predict = model.predict(x_test)               # 0과 1 **사이** -> sigmoid
y_predict = np.round(y_predict)

# print("x_test의 예측값 : ", y_predict)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")
acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
