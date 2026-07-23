# 09_1 카피
from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_iris
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import r2_score
import time
from xgboost import XGBRegressor


# 1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.95,         
    stratify=y                      # y의 분류를 균형있게 자름
)

print(np.unique(y_train))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


# 2. 모델
# parameters = [
#     {"C" : [1, 10, 100, 1000], "kernel":['linear', 'sigmoid'], "degree":[3, 4, 5] },                        
#     {"C" : [1, 10, 100], "kernel":['rbf'], 'gamma':[0.001, 0.0001]},                                         
#     {"C" : [1, 10, 100, 1000], "kernel":['sigmoid'], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4, 5]},    
# ]                                                                                                            
# KFold마다 candidate가 바뀌며 10개만 쓴다
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
parameters = [
    # xg_boost의 parameters  ->  외울 것
    {'n_estimators':[100, 200], "max_depth":[6, 10, 12], 'learning_rate':[0.1, 0.01, 0.001]}, # 18개
       # 100이 default                 # 트리의 깊이                # 학습률 (HyperParameterTuning 활용)<0 (매우 작은 수)
    {'max_depth':[6, 8, 10, 12], 'learning_rate':[0.1, 0.01, 0.001]},                         # 12개
    {'min_child_weight':[2, 3, 5, 10], 'learning_rate':[0.1, 0.01, 0.001]},                   # 12개
]                                                                                                            # 총 42번


# model = GridSearchCV(XGBRegressor(), parameters, cv = kfold, verbose=1,)
model = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold, verbose=1,)             

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear') -> default임. 입력한 것중에 없음
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1}
# 4. 평가, 예측
print("best_score : ", model.best_score_)                   # best_score :  0.9248728275299072

print("model.score : ", model.score(x_test, y_test))        # model.score :  0.8337117433547974

y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))                # r2_score :  0.8337117433547974
print("걸린 시간 : ", round(end_time-start_time, 2), "초")        # 걸린 시간 :  1.12 초