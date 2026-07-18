# verbose
"""목적 : epochs의 부가적인 옵션을 off 해 프로그램 실행 속도를 높임
        0 :  침묵
        1 : 과정이 보임
        2 : 실행바 xx
        3 : 각 시간 xx  (epochs값만) -> time() 활용해서 시간 체크    
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
import time

# 1. 데이터 전처리
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle = True,
                                                    random_state=100,)

# print(x_train.shape, y_train.shape)             # (309, 10) (309,)
# print(y_train.shape, y_test.shape)              # (309,) (133,)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(10,)))
model.add(Dense(26, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=32)
end_time = time.time()

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss = ", loss)
y_predict = model.predict(x_test)
print("y_predict = ", y_predict)
r2 = r2_score(y_test, y_predict)
print("r2 = ", r2)

print("걸린시간 = ", round(end_time - start_time, 2), "초")