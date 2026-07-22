import numpy as np
from sklearn.svm import LinearSVC                         # support vector machine : 옛날의 단순한 단층 선형회귀 모델
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 0, 0, 1])

# print(x_data.shape, y_data.shape)                         # (4, 2) (4,)


# 2. 모델
model = LinearSVC()                                       # loss 도 이미 구현되어 있음

# 3. 훈련
model.fit(x_data, y_data)                                 # 모델 자체가 순전파 알고리즘이라 epochs 가 없음

# 4. 평가, 예측
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)                     # score : accuracy 예측 
print(results)

# 회귀식에서 r2_score가 높을수록(1) 좋음                                                         
acc = accuracy_score(y_data, y_predict)
print("acc : ", acc)                                      # 1.0 출력