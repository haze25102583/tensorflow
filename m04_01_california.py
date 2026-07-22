import numpy as np
from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape)                                         # (20640, 8) (20640,)

# 2. 모델 구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearSVC()
# model = LogisticRegression()                                    # 분류
# model = DecisionTreeRegressor()
model = RandomForestRegressor()                                   # Tree를 assemble 했기 때문에 속도는 느리지만 성능을 훨씬 좋음
                                                                  # 0.9741354512895658

# 3. 컴파일, 훈련
model.fit(x, y)

# 4. 평가, 예측
results = model.score(x, y)                                       # 회귀 : r2 score, 분류 : acc
print(results)