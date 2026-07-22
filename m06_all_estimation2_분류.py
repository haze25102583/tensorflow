# estimator 측정자

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=100, test_size=0.2)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# model = RandomForestRegressor()

all_models = all_estimators(type_filter='classifier')

# print("all_models : ", all_models)
print("모델의 갯수 : ", len(all_models))                        # len : 개체의 갯수 출력

max_score = 0
max_name = '~히히 못가~'

for (name, algorithm) in all_models:
    try:    # python의 예외처리
        model = algorithm()
        
        # 3. 훈련
        model.fit(x_train, y_train)
        
        # 4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, '의 정답률 : ', results)
        
        if results > max_score :
            max_score = results
            max_name = name
        
    except: 
        print(name, '은(는) 에러 발생')
    
print("=======================================================")
print("최고모델 : ", max_name, max_score)
print("=======================================================")
    
#     ValueError: `n_components` upper bound is 1. Got 2 instead. Reduce `n_components`.
# PS C:\nsu_work> 
# n_components는 디폴트 불가능. 반드시 값을 넣어야한다

