# Train data, Test data 범위 -> 전체
# 전체에서 70% Train data, 30% Test             -> 랜덤하게 추출
# xx 앞의 70억개 Train, 뒤의 30억개 Test xx      -> 특정 부분만 반영

# import 가장 위에!                             -> 가독성
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np                                           # 행렬 연산에 특화
from sklearn.model_selection import train_test_split         # 앞글자 소문자 -> 함수 or 메서드


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [실습] train과 test를 섞어서 랜덤하게 7:3을 뽑는다            -> 데이터 정제, 데이터 전처리
# 힌트 : 사이킷런
x_train, x_test, y_train, y_test = train_test_split(x, y,                                       # 하이퍼파라미터 튜닝 by 매개변수(parameter)
                                                    
                                                    train_size=0.7,                             # test_size=0.3 도 가능
                                                                                                # train_size + test_size =1 이어야 함
                                                                                                # (1초과:value error, 1미만: 데이터 누락)
                                                                                                # default 값이 있으나 명시하는 게 좋음
                                                                                                
                                                    #shuffle=False,                             # True -> 랜덤하게 섞음. (default값 : True)
                                                    random_state=368                            # 랜덤난수 : 랜덤을 뽑는 기준값. 이후 같은 수를 인자값으로 주면 동일한 결과가 나옴
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
print("loss = ", loss)                          # loss =  0.00031298567773774266

result = model.predict(np.array([11]))
print("11의 예측값 :  ", result)                 # 11의 예측값 :   [[10.977603]]
