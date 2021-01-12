'''
출처: https://rfriend.tistory.com/282?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

1. 
(1) csv 파일 불러오기 : read_csv()
(2) 구분자 '|' 인 text 파일 불러오기 : sep='|'
(3) 파일 불러올 때 index 지정해주기 : index_col
(4) 변수 이름(column name, header) 이 없는 파일 불러올 때 이름 부여하기
          : names=['X1', 'X2', ... ], header=None
(5) 유니코드 디코드 에러, UnicodeDecodeError: 'utf-8' codec can't decode byte
(6) 특정 줄은 제외하고 불러오기: skiprows = [x, x]
(1) n 개의 행만 불러오기: nrows = n
(8) 사용자 정의 결측값 기호 (custom missing value symbols) 
(9) 데이터 유형 설정 (Setting the data type per each column)

2. DB에 접속해서 데이터 불러오기 (DB connection and SQL query)
3. DataFrame을 csv 파일로 내보내기 : df.to_csv()
4. pd.DataFrame 만들고 Attributes 조회, 행(row) 기준으로 선택, 열(column) 기준으로 선택, index 재설정, reindex 과정에서 생기는 결측값 채우기

(1) pandas DataFrame 만들기
pd.DataFrame() 에서 사용하는 Parameter 들에는 (1) data, (2) index, (3) columns, (4) dtype, (5) copy 의 5가지가 있습니다.
     (1-1) data : numpy ndarray, dict, DataFrame 등의 data source
     (1-2) index : 행(row) 이름, 만약 명기하지 않으면 np.arange(n)이 자동으로 할당 됨
     (1-3) column : 열(column) 이름, 만약 명기하지 않으면 역시 np.arnage(n)이 자동으로 할당 됨
     (1-4) dtype : 데이터 형태(type), 만약 지정하지 않으면 Python이 자동으로 추정해서 넣어줌
     (1-5) copy : 입력 데이터를 복사할지 지정. 디폴트는 False 임. (복사할 거 아니면 메모리 관리 차원에서 디폴트인 False 설정 사용하면 됨)
(2) DataFrame 의 Attributes 조회하기
     (2-1) T : 행과 열 전치 (transpose)
     (2-2) axes : 행과 열 이름을 리스트로 반환
     (2-3) dtypes : 데이터 형태 반환
     (2-4) shape : 행과 열의 개수(차원)을 튜플로 반환
     (2-5) size : NDFrame의 원소의 개수를 반환
     (2-6) values : NDFrame의 원소를 numpy 형태로 반환
(3) 행(row) 기준으로 선택해서 가져오기
(4) 열(column) 기준으로 선택해서 가져오기
(5) index 재설정하기 (reindex)
(6) 시계열 데이터 index 재설정

5. DataFrame 합치기
(1) 여러개의 동일한 형태 DataFrame 합치기 : pd.concat()
     (1-1) 위 + 아래로 DataFrame 합치기(rbind) : axis = 0
     (1-2) 왼쪽 + 오른쪽으로 DataFrame 합치기(cbind) : axis = 1
     (1-3) 합집합(union)으로 DataFrame 합치기 : join = 'outer'
     (1-4) 교집합(intersection)으로 DataFrame 합치기 : join = 'inner'
     (1-5) axis=1일 경우 특정 DataFrame의 index를 그대로 이용하고자 할 경우 : join_axes
     (1-6) 기존 index를 무시하고 싶을 때 : ignore_index
     (1-7) 계층적 index (hierarchical index) 만들기 : keys 
     (1-8) index에 이름 부여하기 : names
     (1-9) index 중복 여부 점검 : verify_integrity
(2) DataFrame과 Series 합치기 : pd.concat(), append()
     (2-1) DataFrame에 Series '좌+우'로 합치기 : pd.concat([df, Series], axis=1)
     (2-2) DataFrame에 Series를 '좌+우'로 합칠 때
          열 이름(column name) 무시하고 정수 번호 자동 부여 : ignore_index=True
     (2-3) Series 끼리 '좌+우'로 합치기 : pd.concat([Series1, Series2, ...], axis=1)
          Series의 이름(name)이 있으면 합쳐진 DataFrame의 열 이름(column name)으로 사용
     (2-4) Series 끼리 합칠 때 열 이름(column name) 덮어 쓰기 : keys = ['xx', 'xx', ...]
     (2-5) DataFrame에 Series를 '위+아래'로 합치기 : df.append(Series, ignore_index=True)

6. Database처럼 DataFrame Join/Merge 하기 : pd.merge()
(1) DataFrame을 key 기준으로 합치기 
     (1-1) Merge method : left (SQL join name : LEFT OUTER JOIN)
     (1-2) Merge method : right (SQL join name : RIGHT OUTER JOIN)
     (1-3) Merge method : inner (SQL join name : INNER JOIN)
     (1-4) Merge method : outer (SQL join name : FULL OUTER JOIN)
     (1-5) indicator = True : 병합된 이후의 DataFrame에 left_only, right_only, both 등의 출처를 알 수 있는 부가정보 변수 추가
     (1-6) 변수 이름이 중복될 경우 접미사 붙이기 : suffixes = ('_x', '_y')
(2) DataFrame을 index 기준으로 합치기 (merge, join on index)
     (2-1) index를 기준으로 Left Join 하기 (Left join on index)
     (2-2) index를 기준으로 Right Join 하기 (Right join on index)
     (2-3) index를 기준으로 inner join 하기 (inner join on index)
     (2-4) index를 기준으로 outer join 하기 (outer join on index)
     (2-5) index와 Key를 혼합해서 DataFrame 합치기 (Joining key columns on an index)

7. DataFrame 결측값 여부 확인, 결측값 개수 
   isnull(), notnull(), df.isnull().sum(), df.notnull().sum(), df.isnull().sum(1), df.notnull().sum(1)
(1) DataFrame 전체의 결측값 여부 확인 : df.isnull(), isnull(df), df.notnull(), notnull(df)
(2) 특정 변수, 원소에 결측값 추가하기, 결측값 여부 확인하기 : indexing & None
(3) 칼럼별 결측값 개수 구하기 : df.isnull().sum()
(4) 행(row) 단위로 결측값 개수 구하기 : df.isnull().sum(1)
    행(row) 단위로 실측값 개수 구하기 : df.notnull().sum(1)

(5) 결측값 연산 (calculations with missing data)
     (5-1) sum(), cumsum() methods 계산 시 : NaN은 '0'으로 처리
     (5-2) mean(), std() 연산 시 : NaN은 분석 대상(population)에서 제외
     (5-3) DataFrame 칼럼 간 연산 시 : NaN이 하나라도 있으면 NaN 반환
     (5-4) DataFrame 간 연산 : 동일한 칼럼끼리는 NaN을 '0'으로 처리하여 연산,
          동일한 칼럼이 없는 경우(한쪽에만 칼럼이 있는 경우)는 모든 값을 NaN으로 반환
(6) 결측값 채우기, 결측값 대체하기, 결측값 처리 (filling missing value, imputation of missing values) : df.fillna()
     (6-1) 결측값을 특정 값으로 채우기 (replace missing values with scalar value) : df.fillna(0)
     (6-2) 결측값을 앞 방향 혹은 뒷 방향으로 채우기 (fill gaps forward or backward)
     (6-3) 앞/뒤 방향으로 결측값 채우는 회수를 제한하기 (limit the amount of filling)
     (6-4) 결측값을 변수별 평균으로 대체하기(filling missing values with mean per columns)
     (6-5) 결측값을 다른 변수의 값으로 대체하기
          (filling missing values with another columns' values)
(7) 결측값 있는 행 제거, 결측값 있는 행 제거 : dropna(axis=0), dropna(axis=1)
     (7-1) 결측값이 들어있는 행 전체 삭제하기(delete row with NaN) : df.dropna(axis=0)
     (7-2) 결측값이 들어있는 열 전체 삭제하기 (delete column with NaN) : df.dropna(axis=1)
     (7-3) 특정 행 또는 열을 대상으로 결측값이 들어있으면 제거
          (delete specific row or column with missing values) : df[ ].dropna()
(8) 결측값 보간하기 (interpolation of missing values) : interpolate(), interpolate(method='time'), interpolate(method='values')
     (8-1) 시계열데이터의 값에 선형으로 비례하는 방식으로 결측값 보간
          (interpolate TimeSeries missing values linearly) : ts.interpolate()
     (8-2) 시계열 날짜 index를 기준으로 결측값 보간
          (interploate TimeSeries missing values along with Time-index) : ts.interploate(method='time')
     (8-3) DataFrame 값에 선형으로 비례하는 방식으로 결측값 보간
          (interpolate DataFrame missing values linearly)
     (8-4) 결측값 보간 개수 제한하기 (limit the number of consecutive interpolation) : limit
(9) 결측값, 원래 값을 다른 값으로 교체하기(replacing generic values) : replace()
     (9-1) 결측값, 실측값을 다른 값으로 교체하기 : replace(old_val, new_val)
     (9-2) list 를 다른 list 값으로 교체하기 : replace([old1, old2, ...], [new1, new2, ...])
     (9-3) mapping dict 로 원래 값, 교체할 값 매핑 : replace({old1 : new1, old2: new2})
     (9-4) DataFrame의 특정 칼럼 값 교체하기 : df.replace({'col1': old_val}, {'col1': new_val})
8. 중복값 확인 및 처리 : DataFrame.duplicated(), DataFrame.drop_duplicates(), keep='first', 'last', False
(1) 중복 데이터가 있는지 확인하기 : DataFrame.duplicated()
(2) 중복이 있으면 처음과 끝 중 무슨 값을 남길 것인가? : keep = 'first', 'last', False
(3) 중복값 처리(unique한 1개의 key만 남기고 나머지 중복은 제거) 
       : DataFrame.drop_duplicates()
9. 유일한 값 찾기 : pd.Series.unique(), 유일한 값별로 개수 세기 : pd.Series.value_counts()
(1) 유일한 값 찾기 : pd.Series.unique()
(2) 유일한 값별로 개수 세기 : pd.Series.value_counts()
(2-1) 유일 값 별 상대적 비율 : pd.Series.value_counts(normalize=True)
(2-2) 유일한 값 기준 정렬 : pd.Series.value_counts(sort=True, ascending=True)
(2-3) 결측값을 유일한 값에 포함할지 여부 : pd.Series.value_counts(dropna=True)
(2-4) Bins Group별 값 개수 세기 : pd.Series.value_counts(bins=[ , , ,])

10. 표준정규분포 데이터 표준화 (standardization) : (x-mean())/std(), ss.zscore(), StandardScaler(data).fit_transform(data)
(1) 표준정규분포 데이터 표준화
     (1-1) Numpy 를 이용한 표준화 : z = (x - mean())/std()
     (1-2) scipy.stats 을 이용한 표준화 : ss.zscore()
     (1-3) sklearn.preprocessing 을 이용한 표준화 : StandardScaler().fit_transform()
(2) 이상치, 특이값이 들어있는 데이터의 표준화 (Scaling data with outliers)
     (2-1) 이상치가 포함된 데이터의 표준정규분포로의 표준화 : 
          sklearn.preprocessing.StandardScaler()
     (2-2) 이상치가 포함된 데이터의 중앙값과 IQR 를 이용한 표준화
          : sklearn.preprocessing.RobustScaler()
(3) 최소 최대 '0~1' 범위 변환 (scaling to 0~1 range) : sklearn.preprocessing.MinMaxScaler()
     (3-1) 최소, 최대값을 구해서 '0~1' 범위로 변환
     (3-2) sklearn.preprocessing.MinMaxScaler() method를 사용한 최소.최대 '0~1' 범위 변환
     (3-3) sklearn.preprocessing.minmax_scale() 함수를 사용한 최소.최대 '0~1' 범위 변환

11. 이항변수화 변환 (Binarization) : sklearn.preprocessing.Binarizer(), sklearn.preporcessing.binarize()
(1) 이항변수화 변환
     (1-1) sklearn.preprocessing.Binarizer() method를 사용한 이항변수화
     (1-2) sklearn.preprocessing.binarize() 함수를 사용한 이항변수화
(2) 범주형 변수의 이항변수화 : sklearn.preprocessing.OneHotEncoder()
     (2-1) OneHotEncoder() 로 범주형 변수의 이항변수화 적합시키기 : enc.fit()
     (2-2) 적합된(fitted) OneHotEncoder()의 Attributes 확인해보기
     (2-3) 적합된 OneHotEncoder()로 새로운 범주형 데이터셋을 이항변수화 변환하기
(3) 연속형 변수의 이산형화(discretization) : np.digitize(data, bins), pd.get_dummies(), np.where(condition, 'factor1', 'factor2', ...)
     (3-1) np.digitize(data, bins)를 이용한 연속형 변수의 이산형화 (discretization)
     (3-2) pd.get_dummies() 를 이용해 가변수(dummy var) 만들기
     (3-3) np.where(condition, factor1, factor2, ...)를 이용한 연속형 변수의 이산형화
(4) 다항차수 변환, 교호작용 변수 생성 : sklearn.preprocessing.PolynomialFeatures()
     (4-1) sklearn.preprocessing.PolynomialFeatures()를 사용해 2차항 변수 만들기
     (4-2) 교호작용 변수만을 만들기 : interaction_only=True

12. 데이터 재구조화 (reshaping) : data.pivot(), pd.pivot_table(data)
(1) 데이터 재구조화 : data.pivot(index, columns, values)
(2) 데이터 재구조화 : pd.pivot_table(data, index, columns, values, aggfunc)
(3) 데이터 재구조화(reshaping data) : pd.DataFrame.stack(), pd.DataFrame.unstack()
     (3-1) pd.DataFrame.stack(level=-1, dropna=True)
     (3-2) pd.DataFrame.unstack(level=-1, fill_value=None)
(4) 데이터 재구조화(reshape) : pd.melt()
     (1) pd.melt(data, id_vars=['id1', 'id2', ...]) 를 사용한 데이터 재구조화
     (2) pd.melt() 의 variable 이름, value 이름 부여하기 : var_name, value_name
     (3) data vs. pd.melt() vs. pd.pivot_table() 비교해보기
(5) 데이터 재구조화(reshape) : pd.wide_to_long()
     (1) pd.wide_to_long(data, ["col_prefix_1", "col_prefix_2"], i="idx_1", j="idx_2")
     (2) pd.wide_to_long()에 의한 index, columns 변화 비교
(6) 데이터 재구조화(reshape) : pd.crosstab() 사용해 교차표(cross tabulation)
     (1) 교차표(contingency table, frequency table) 만들기 : pd.crosstab(index, columns)
     (2) Multi-index, Multi-level로 교차표 만들기 : pd.crosstab([id1, id2], [col1, col2])
     (3) 교차표의 행 이름, 열 이름 부여 : pd.crosstab(rownames=['xx'], colnames=['aa'])
     (4) 교차표의 행 합, 열 합 추가하기 : pd.crosstab(margins=True)
     (5) 구성비율로 교차표 만들기 : pd.crosstab(normalize=True)

13. 데이터 정렬 (sort, arrange) : DataFrame.sort_values(), sorted(), list.sort()
(1) DataFrame 정렬 : DataFrame.sort_values()
(2) Tuple 정렬하기 : sorted(tuple, key) method
(3) List 정렬하기 : sorted(list), or list.sort()

14. Series, DataFrame 행, 열 생성(creation), 선택(selection, slicing, indexing), 삭제(drop, delete)
(1) Series 생성 및 Series 원소 선택 (element selection, indexing)
(2) DataFrame 행과 열 생성, 선택, 삭제 (creation, selection, drop of row and column)


다수개의 범주형자료로 가변수 만들기 (dummy variable)
데이터프레임에서 두 개의 문자열 변수의 각 원소를 합쳐서 새로운 변수 만들기
groupby() 로 그룹별 집계하기 (data aggregation by groups)
GroupBy로 그룹별로 반복 작업하기 (Iteration over groups)
다양한 GroupBy 집계 방법 : Dicts, Series, Lists, Functions, Index Levels
GroupBy 집계 메소드와 함수 (Group by aggregation methods and functions)
여러개의 함수를 적용하여 GroupBy 집계하기 : grouped.agg()
여러개의 칼럼에 대해 다른 함수를 적용한 Group By 집계: grouped.apply(functions)
GroupBy 를 활용한 그룹 별 가중평균 구하기
결측값을 그룹 평균값으로 채우기 (Fill missing values by Group means)
데이터프레임에 그룹 단위로 통계량을 집계해서 칼럼 추가하기 : df.groupby(['group']).col.transform('count')
동일 길이로 나누어서 범주 만들기 pd.cut(), 동일 개수로 나누어서 범주 만들기 pd.qcut()
그룹 별 변수 간 상관관계 분석 (correlation with columns by groups)
그룹 별 선형회귀모형 적합하기 (Group-wise Linear Regression)
그룹 별 무작위 표본 추출 (random sampling by group)
다수 그룹 별 다수의 변수 간 상관관계 분석 (correlation coefficients with multiple columns by groups)
DataFrame, Series의 행, 열 개수 세기
DataFrame을 정렬한 후에, 그룹별로 상위 N개 행 선택하기 (sort DataFrame by value and select top N rows by group)
DataFrame index를 reset칼럼으로 가져오고 이름 부여하기
DataFrame, Series에서 조건에 맞는 값이 들어있는 행 indexing 하기 : df.isin()
DataFrame, Series에서 순위(rank)를 구하는 rank() 함수
DataFrame에서 천 단위 숫자의 자리 구분 기호 콤마(',')를 없애는 방법

출처: https://rfriend.tistory.com/463?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]
출처: https://rfriend.tistory.com/461?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]
출처: https://rfriend.tistory.com/460?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]
출처: https://rfriend.tistory.com/456?category=675917 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]




'''