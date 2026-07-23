# HyperParameterTuning의 자동화
from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# 1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.95,         
    stratify=y                      # y의 분류를 균형있게 자름
)

print(np.unique(y_train))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


# 2. 모델
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel":['linear', 'sigmoid'], "degree":[3, 4, 5] },                        # 딕셔너리, 24번
    {"C" : [1, 10, 100], "kernel":['rbf'], 'gamma':[0.001, 0.0001]},                                         # 6번
    {"C" : [1, 10, 100, 1000], "kernel":['sigmoid'], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4, 5]},    # 36번
]                                                                                                            # 총 66번


model = GridSearchCV(SVC(), parameters, cv = kfold, verbose=1,)             # 66*5 = 330번, Grid : 격자(모든 경우의 수를 다 해보겠다는 의미)
                                                                            # GridSearchCV -> 모델의 랩핑

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear') -> default임. 입력한 것중에 없음
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

# 4. 평가, 예측
print("best_score : ", model.best_score_)
# best_score :  0.9862068965517242

print("model.score : ", model.score(x_test, y_test))        # 0.875
# model.score :  0.875

# model.score와 best_score의 차이
# model.score가 더 중요하고, 이에 test data를 적용해야 함
# train data에 과적합 되어있는 score 이기 때문에

y_predict = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_predict))  # 0.875 -? model.score와 동일
print("걸린 시간 : ", round(end_time-start_time, 2), "초")