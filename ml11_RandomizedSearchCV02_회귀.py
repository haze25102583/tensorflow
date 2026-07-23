# 09_02 카피해서 수정, 성능 비교
# 09_1 카피
# HyperParameterTuning의 자동화
from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.metrics import r2_score
import time
from xgboost import XGBRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.95,         
    # stratify=y                   # y의 분류를 균형있게 자름
)

print(np.unique(y_train))

kfold = KFold(n_splits=5, shuffle=True, random_state=123)


# 2. 모델
parameters = [
    # xg_boost의 parameters  ->  외울 것
    {'n_estimators':[100, 200], "max_depth":[6, 10, 12], 'learning_rate':[0.1, 0.01, 0.001]}, # 18개
       # 100이 default                 # 트리의 깊이                # 학습률 (HyperParameterTuning 활용)<0 (매우 작은 수)
    {'max_depth':[6, 8, 10, 12], 'learning_rate':[0.1, 0.01, 0.001]},                         # 12개
    {'min_child_weight':[2, 3, 5, 10], 'learning_rate':[0.1, 0.01, 0.001]},                   # 12개
]                                                                                                            # 총 42번


model = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold, verbose=1,)           # Grid : 격자(모든 경우의 수를 다 해보겠다는 의미)
# Fitting 5 folds for each of 42 candidates, totalling 210 fits                    # GridSearchCV -> 모델의 랩핑

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}

# 4. 평가, 예측
print("best_score : ", model.best_score_)
# best_score :  0.8414193439947031
# test 까지 반영 -> 성능 조금 낮음

print("model.score : ", model.score(x_test, y_test))        # 0.875
# model.score :  0.8516948379364527
# Train의 KFold 형태(과적합) -> 성능 조금 높음


# model.score와 best_score의 차이
# model.score가 더 중요하고, 이에 test data를 적용해야 함
# train data에 과적합 되어있는 score 이기 때문에

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))  # r2_score :  0.8516948379364527
print("걸린 시간 : ", round(end_time-start_time, 2), "초")  # 걸린 시간 :  84.12 초