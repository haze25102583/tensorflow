# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
from sklearn.datasets import load_diabetes                                                 # 교육용 데이터 제공
import pandas as pd                                                                        # csv파일을 데려오는 라이브러리

# rainbow_csv, edit_csv
# 1. 데이터
path = "./_data/"           # . : 현재 폴더

train_csv = pd.read_csv(path + "train.csv", index_col=0)            # "./_data/train.csv"
test_csv = pd.read_csv(path + "test.csv", index_col=0)                           # "./data/test.csv"
submit_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

# print(train_csv)
# print(train_csv.shape)        # (10886, 11)

# print(test_csv)
# print(test_csv.shape)         # (6493, 8)

# print(submit_csv)
# print(submit_csv.shape)         # (6493, 1)

print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='str')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)             # pandas는 numpy 파일 형태
                                                                          # 열을 뺴려면 axis=1
                                                                          
# print(x)                                                                    

print("=============================================")
y = train_csv['count']
print(y)
print(y.shape)