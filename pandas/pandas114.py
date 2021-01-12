# Python으로 하는 탐색적 자료 분석 (Exploratory Data Analysis)
# Python을 통해 탐색적 자료분석
# 1. 탐색적 자료분석의 기본은 바로 변수 별로 분포를 그려보는 것
#    수치형 데이터의 경우는 히스토그램을, 명목형 데이터의 경우는 빈도표를 통해 데이터의 분포를 관찰
#  유명한 데이터셋인 타이타닉 데이터 :  titanic.csv
# 기본적인 탐색적 자료 분석의 순서는 아래와 같이 정리해보았습니다. 
# 1. 데이터를 임포트하여 메모리에 올린다.
# 2. 데이터의 모양을 확인 한다.
# 3. 데이터의 타입을 확인한다.
# 4. 데이터의 Null 값을 체크한다. 
# 5. 종속변수의 분포를 살펴본다.
# 6. 독립변수 - 명목형 변수의 분포를 살펴본다. 
# 7. 독립변수 - 수치형 변수의 분포를 살펴본다. 
# 8. 수치형, 명목형 변수간의 관계를 파악한다. 

# 1. 데이터를 임포트한다.
# 아래와 같이 패키지와 데이터를 임포트합니다. numpy, pandas, matplotlib, seaborn은 이 4가지의 패키지는 파이썬을 통한 EDA에서 거의 필수적으로 사용하는 라이브러리입니다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
titanic = pd.read_csv("data/titanic3.csv")

# 2. 데이터의 모양을 확인한다. 
print(titanic.head())
'''
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
'''

# 3. 데이터의 타입을 체크한다.
# 데이터의 타입을 체크하는 이유는 해당 변수의 타입을 제대로 맞추어주기 위해서입니다. 
# 범주형 변수의 경우 object 또는 string, 수치형 변수의 경우 int64 혹은 float 64로 맞추어주면 됩니다.
# 범주형 변수의 경우 값이 문자열로 들어가 있으면 알아서 object 타입이 되지만, 
# 만약의 숫자로된 범주형 변수의 경우 int64 등으로 잘못 타입이 들어가 있는 경우가 있습니다.  
print(titanic.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
Pclass         891 non-null int64
# PassengerId    891 non-null int64
Survived       891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            891 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
'''

# 위 데이터의 경우 Survived와 PClass 변수가 범주형 int64로 잘못 되어있으므로 형변환을 합니다.
titanic['Survived'] = titanic['Survived'].astype(object)
titanic['Pclass'] = titanic['Pclass'].astype(object)

# 4. 데이터의 Null 값을 체크한다. 
# Null Check도 매우 중요한 작업 중 하나입니다. 
# 단순히 Tutorial이나 학습을 위해 제작된 데이터셋이아닌 현실의 데이터셋의 경우, 많은 부분이 Null 인 경우가 많습니다. 
# 따라서 이 Null 값을 어떻게 처리하느냐가 매우 중요합니다. 
print(titanic.isnull().sum())
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age              0
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''

# Cabin 변수가  687 행이 missing이고 Embarked가 2개의 행이 missing인 것을 확인하였습니다.
# Null 값이 있는 경우, 크게 그 값을 빼고 하는지, 혹은 결측치를 대치하는지 2개의 방법으로 나눌 수 있습니다. 
# 각각의 방법에 대한 이름이 다르긴한데 보통 첫 번째 방법을 complete data analysis, 두 번째 방법을 Imputation이라고 이름 붙입니다. 

missing_df = titanic.isnull().sum().reset_index()
missing_df.columns = ['column', 'count']
missing_df['ratio'] = missing_df['count'] / titanic.shape[0]
missing_df.loc[missing_df['ratio'] != 0]
print(missing_df)
# column	count	ratio
# 10	Cabin	687	0.771044
# 11	Embarked	2	0.002245
# 위 명령어를 통해 전체의 몇 %가 missing 인지를 확인할 수 있습니다. 

# 5. 종속변수 체크
# 기본적으로 종속변수의 분포를 살펴봅니다. 종속변수란 다른 변수들의 관계를 주로 추론하고, 최종적으로는 예측하고자 하는 변수입니다. 
titanic['Survived'].value_counts().plot(kind='bar')
plt.show()

# 6. 명목형 변수의 분포 살펴보기
#    단변수 탐색
category_feature = [ col for col in titanic.columns if titanic[col].dtypes == "object"]
print(category_feature) # ['Survived', 'Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
# 앞에서 명목형 변수의 형을 object로 모두 변경했기 때문에 이처럼 컬럼 중에서 object 타입을 가진 컬럼만 뽑아서 명목형 변수의 리스트를 만듭니다. 
# 이 때, 데이터의 기본키(인덱스), 종속변수 등을 제외하고 분석하는 것이 좋습니다. 

category_feature = list(set(category_feature) - set(['PassengerId','Survived','boat','body','home.dest']))
print(category_feature) #  ['Cabin', 'Embarked', 'Ticket', 'Sex', 'Name', 'Pclass']

# 다음으로는 그래프를 통해 명목형 변수의 분포를 살펴보는 것입니다. 
for col in category_feature:
    titanic[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

# 이렇게 살펴봄으로써 명목형 변수를 어떻게 다룰지를 판단할 수 있습니다. 
# 예를 들어, 카테고리수가 너무 많고, 종속변수와 별로 관련이 없어보이는 독립 변수들은 빼고 분석하는 것이 나을 수도 있습니다.
# 이변수 탐색
sex_df = titanic.groupby(['Sex','Survived'])['Survived'].count().unstack('Survived')
sex_df.plot(kind='bar', figsize=(20,10))
plt.title('Sex')
plt.show()
# 성별-생존의 관계 파악처럼 두 변수의 관계를 파악하기 위해서는 위와 같이 확인할 수 있습니다.

# 7. 수치형 변수의 분포 살펴보기
# 단변수 탐색 seaborn 패키지의 distplot 함수를 이용하면 매우 편합니다.
# 우선 이와 같이 전체 변수 중에서 범주형 변수와 기타 인덱스 변수, 종속변수들을 제외하고 수치형 변수만 골라냅니다.
numerical_feature = list(set(titanic.columns) - set(category_feature) - set(['PassengerId','Survived']))
numerical_feature = np.sort(numerical_feature)
print(numerical_feature)
# 변수별로 for문을 돌면서 distplot을 그립니다
for col in numerical_feature:
    sns.distplot(titanic.loc[titanic[col].notnull(), col])
    plt.title(col)
    plt.show()

# 이변수, 삼변수 탐색
# seaborn 패키지의 pairplot을 통해 종속변수를 포함한 3개의 변수를 한 번에 볼 수 있도록 플로팅합니다.
sns.pairplot(titanic[list(numerical_feature) + ['Survived']], hue='Survived', 
             x_vars=numerical_feature, y_vars=numerical_feature)
plt.show()

# pairplot은 어러 변수의 관계를 한 번에 파악할 수 있으며,  hue 파라미터를 통해 종속변수를 지정함으로써 세 변수의 관계를 파악할 수 있습니다.

# 8. 수치형, 명목형 변수 간의 관계 탐색
# 앞서서 수치형-수치형  간의 관계, 그리고 명목형-명목형 간의 관계에 종속변수까지 포함해서 보았습니다. 
# 이 번에는 수치형-명목형 간의 관계를 파악해 보는 것입니다. 예를 들어, 성별, 나이, 생존여부 3개의 변수를 동시에 탐색하고 싶을 수 있습니다.
#  이 경우에 명목형 변수에 따라 수치형변수의 boxplot을 그려봄으로써 대략적인 데이터의 형태를 살펴볼 수 있습니다. 
unique_list = titanic['Sex'].unique()
for col in numerical_feature:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Sex', y=col, hue='Survived', data=titanic.dropna())
    plt.title("Sex - {}".format(col))
    plt.show()

