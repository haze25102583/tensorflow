# 16 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris                                # 교육용 데이터 제공

import numpy as np
import pandas as pd
import time


# 1. 데이터
datasets = load_iris()

# print(datasets)                     # (None, 4) -> 이후 input_shape=(4,)
print(datasets.DESCR)
print(datasets.feature_names)         # 열, 속성, 특성, 컬럼 

x = datasets.data
y = datasets.target


print(x.shape, y.shape)                             # (150, 4) (150,)
print(np.unique(y, return_counts=True))             # (array([0, 1, 2]), array([50, 50, 50]))


########### OneHotEncoding (1) - 판다스 ###########
# import pandas as pd
# y = pd.get_dummies(y)
# print(y)                        # 0 > False, 1 > True

########## OneHotEncoding (2) -> Tensorflow
from tensorflow.keras.utils import to_categorical   # 버전에 따라 warning
y = to_categorical(y)

print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,         # default
                                                    shuffle=True,            # default
                                                    random_state=100,
                                                    )

print(x_train.shape, x_test.shape)                  
print(y_train.shape, y_test.shape)                  
print(np.unique(y_train, return_counts=True))      



# 2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu',input_shape=(4,)))
model.add(Dense(28, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'],
              )

es = EarlyStopping(
    monitor='val_loss',                 # loss, accuracy도 가능
    mode = 'min',                       #       accuracy일 경우, mode = 'max'
                                        # 'auto' -> min으로 해야할 지, max로 해야할 지 자동으로 정함
    patience=100,                
    restore_best_weights=True,          # 통상적으로, True가 더 성능이 좋으나 간혹 False가 성능 좋기도.. (<- hidden layer)
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, # verbose=3,
          validation_split=0.2,
          callbacks = [es])                   
end_time = time.time()



print("=========================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)           # binary_crossentropy 값 나옴
print("loss = ", loss)                      

y_predict = model.predict(x_test)               # 0과 1 **사이** -> sigmoid
# y_predict = np.round(y_predict)               # 다중분류에서는 사용 xx
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)        # 세 개중 가장 큰 값을 1로 : [0.4  0.3  0.2] -> [1  0  0]
# exit()
y_test = np.argmax(y_test, axis=1)
print(y_test)
# exit()

print("y_test의 원값", y_test, y_test.shape)
print("[x_test]의 예측값 : ", y_predict, y_predict.shape)     # y_predict의 예측값이 0~1 사이 값



# print("x_test의 예측값 : ", y_predict)
print("걸린 시간 : ", round(end_time-start_time, 2), "초")
acc_score = accuracy_score(y_test, y_predict)   # accuracy_score() 함수 : (ValueError)Classfication can't handle a mix of
                                                # binary and continous targets
print('accuracy_score : ', acc_score)
