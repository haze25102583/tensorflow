# mse, mae, rmse(제곱으로 인해 숫자가 너무 커지는 것을 보완)              <- 지표는 동일해야 함
# loss, 오차, 에러, cost
# accracy(정확도)

# 모델의 종류 : 1. 회귀모델 (수치형 모델)       2. 분류모델

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np                                           
from sklearn.model_selection import train_test_split         
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error               # 평가하다


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(x, y,                                       
                                                    
                                                    train_size=0.7,                             # test_size=0.3 도 가능
                                                                                                # train_size + test_size =1 이어야 함
                                                                                                # (1초과:value error, 1미만: 데이터 누락)
                                                                                                # default 값이 있으나 명시하는 게 좋음
                                                                                                
                                                    #shuffle=False,                             # True -> 랜덤하게 섞음. (default값 : True)
                                                    random_state=38                            # 랜덤난수 : 랜덤을 뽑는 기준값. 이후 같은 수를 인자값으로 주면 동일한 결과가 나옴
                                                                                                #           -> 랜덤값을 고정(=> 같은 조건)하여, 성능이 그때마다 변하지 않게 함
                                                                                                
                                                                                                #           조정 반복 -> 성능향상 (랜덤값 메모 필수!)
                                                    )                                       


print(x_train, x_test)
print(y_train, y_test)
# exit()


print(x_train.shape, x_test.shape)        # (7,) (3,)
print(y_train.shape, y_test.shape)        # (7,) (3,)

# exit()

#2. 모델구성
model = Sequential()

model.add(Dense(200, input_shape=(1,)))     # (None, 1)
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(1000))
model.add(Dense(40))
model.add(Dense(900))
model.add(Dense(200))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, )       # batch 단위의 데이터 : (목적) 성능 향상, 데이터 과적합 방지

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)                          # loss =  7.451475620269775
                                                # 어느정도의 오차가 있는 것이 정확

# 원값과 예측값을 확인하고, 지표를 구한다
y_predict = model.predict([x_test])             # 100번째 w를 이용해서 예측값 : wx+b
print("y_test의 원값 : ", y_test)
print("[x]의 예측값 :  ", y_predict)


r2 = r2_score(y_test, y_predict)                # 모수, 예측값
print("r2_score : ", r2)

rmse = root_mean_squared_error(y_test, y_predict)
print("rmse : ", rmse)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)
