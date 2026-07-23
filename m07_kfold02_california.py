# 7_1 copy
# default 여서 이후, HyperParameterTuning이 필요함
# 단층 퍼셉트론 구조로 역전파도 없음
# 과적합(overfit) 문제 : 
from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=123) # 전체 데이터를 5등분 / 섞어서 데이터를 자르도록 함

# 2. 모델
model = DecisionTreeRegressor()
model = RandomForestRegressor(max_depth=5, max_leaf_nodes=3, n_estimators=100,
                              max_features=33,)      # hyperparametertuning

# 3. 컴파일, 훈련
# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1) # cpu 전체를 다 쓰겠다
print('ACC : ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4)
      )
