from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.95,         
    stratify=y                      # y의 분류를 균형있게 자름
)
print(x_train.shape, x_test.shape)  # (540, 30) (29, 30)
print(y_train.shape, y_test.shape)  # (540,) (29,)

# 2. 모델
parameters = {                          # key-value 형태. key -> 문자열 형태
    'learning_rate':0.1,
    'max_depth':6,
    'n_estimators':200,
}
model = XGBClassifier(**parameters)      # * : 리스트              ** : 딕셔너리


# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


# 4. 평가, 예측

print("model.score : ", model.score(x_test, y_test))       
#  model.score :  1.0
# Train의 KFold 형태(과적합) -> 성능 조금 높음


# model.score와 best_score의 차이
# model.score가 더 중요하고, 이에 test data를 적용해야 함
# train data에 과적합 되어있는 score 이기 때문에

y_predict = model.predict(x_test)
print("acc_score : ", accuracy_score(y_test, y_predict))  # acc_score :  1.0
print("걸린 시간 : ", round(end_time-start_time, 2), "초")  # 걸린 시간 :  0.14 초