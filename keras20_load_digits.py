# 18 카피
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error          # accuracy_score -> 다중 분류, r2_score -> 회귀 분석
from sklearn.datasets import load_digits

import numpy as np
import pandas as pd
import time


# 1. 데이터
datasets = load_digits()
# print(datasets)         
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target


# print(x.shape, y.shape)                             # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))               # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
                                                      # => 10개의 클래스 : 클래스 별 데이터의 개수를 최대한 균일하게 해야 편향되지 않는다.
                                                      #                  특정 label의 데이터가 너무 적을 때
                                                      #                  특정 데이터를 중폭시켜 이를 해결하기도..


########### OneHotEncoding (1) - 판다스 ###########     -> 2차원 형태의 데이터로 바꿈
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
model.add(Dense(10, activation='relu',input_shape=(64,)))
model.add(Dense(28, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(10, activation='softmax'))


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
model.fit(x_train, y_train, epochs=1000000, batch_size=64, # verbose=3,
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
