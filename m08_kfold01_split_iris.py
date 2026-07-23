# 분류 모델의 문제 : 클래스의 불균형
# 암환자일 확률은 희박하므로, 암인지에 관계없이 암이 아니라고 판명해도 정확도가 높음
from sklearn.model_selection import train_test_split, KFold, cross_val_score    # cross validation : 교차 검증한 score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.95,          # shuffle=False -> 시계열 데이터는 가장 최근의 데이터가 가장 잘 맞음
    stratify=y                      # y의 분류를 균형있게 자름
)

# print(y_train)
# print(y_test)
                                    # stratify 전 >>> [0 1 0 0 0 2 0 1] : 2가 적은데 비해 0이 너무 많다 -> 외도
                                    # 분류데이터는 균형적일 수 있도록, 표본의 수가 적은 수의 클래스를 증폭 or 표본을 더 수집
                                    # stratify 후 >>> [1 2 2 0 0 0 1 2]
print(np.unique(y_train))

# kfold = KFold(n_splits=5, shuffle=True, random_state=123) # 전체 데이터를 5등분 / 섞어서 데이터를 자르도록 함
                                                            # 단 분포까지 정확화지는 않음 (회귀에서 사용)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


# 2. 모델
model = DecisionTreeClassifier()

# 3. 컴파일, 훈련
# 4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, n_jobs=-1) # cpu 전체를 다 쓰겠다
print('ACC : ', scores,
      '\n cross_val_score 평균 : ', round(np.mean(scores), 4)
      )
