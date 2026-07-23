from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
# PCA : 요즘엔 CNN을 주로 사용해 차원 축소 개념으로만 쓴다. 희미한 이미지를 축소시켜 뚜렷하게 하는 것과 비슷.

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

pca = PCA(n_components=25)
x = pca.fit_transform(x)
print(x.shape)      # n_components=25 -> (569, 25)                  # 하이퍼 파라미터 튜닝
                    # 컬럼이 많을 경우, 압축시켜 훈련시킨다
                    # 성능은 좋아질 수도, 안 좋아질 수도 but 시간 단축 


exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,         
    stratify=y                      # y의 분류를 균형있게 자름
)

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
print(y_train.shape, y_test.shape)  # (455,) (114,)




# 2. 모델
parameters = {                          # key-value 형태. key -> 문자열 형태
    'learning_rate':0.5,
    'max_depth':11,
    'n_estimators':100,
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
print("걸린 시간 : ", round(end_time-start_time, 2), "초")  # 걸린 시간 :  0.11 초
