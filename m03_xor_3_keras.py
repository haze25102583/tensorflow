# 다층 퍼셉트론 구성으로 xor 인공지능 겨울 문제 해결

import numpy as np
from sklearn.svm import LinearSVC                         # support vector machine : 옛날의 단순한 단층 선형회귀 모델
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# print(x_data.shape, y_data.shape)                         # (4, 2) (4,)


# 2. 모델
model = Sequential()
model.add(Dense(20, input_dim=2, activation='relu'))      # hidden 층 없음
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)                                 

# 4. 평가, 예측
results = model.evaluate(x_data, y_data)
print("results : ", results)
y_predict = np.round(model.predict(x_data))

print(results)

# 회귀식에서 r2_score가 높을수록(1) 좋음                                                         
acc = accuracy_score(y_data, y_predict)
print("acc : ", acc)                                      # acc :  0.75