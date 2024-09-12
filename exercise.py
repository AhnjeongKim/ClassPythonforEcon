import numpy as np
import pandas as pd
import matplotlib.pyplot as plt          #matplotlib의 pyplot을 plt라는 이름으로 임포트
from statsmodels.formula.api import ols  # statsmodel 라이브러리 불러오기

import os           # 시스템 라이브러리 
os.getcwd()         #디렉토리 경로 확인

# 파일 불러오기
df = pd.read_csv('SATandGPA_LinearRegression.csv')
print (df)
df.head(10)

# 기초 통계량
df.describe()
print(df.mean())
print(df.min())

x=df.loc[:,'SAT']  #df.loc: 데이터 프레임을 라벨을 통해 가져오기 (분석 편의를 위함)
y=df.loc[:,'GPA']  

# 데이터 시각화
plt.scatter(x,y)
plt.show()

plt.scatter(x,y, color='green')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# 단순선형회귀 분석
model= ols(formula='SAT~GPA',data=df).fit()
print(model.sumary())
print('parameters',model.params)
print('R2',model.rsquared)