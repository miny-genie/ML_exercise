# --------------------------------- Problem 1 (1) ---------------------------------
def Problem_1_1():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # HOSPITAL = pd.read_table("hospital.txt", sep="\t")

    # sns.set(font_scale=0.5)     # 폰트 사이즈 설정
    # sns.set_style('ticks')      # 축 눈금 표시 여부
    # data = HOSPITAL[['InfctRsk', 'Culture', 'MedSchool', 'Region']] # 그리려는 변수
    # sns.pairplot(data, diag_kind=None)  # https://seaborn.pydata.org/generated/seaborn.pairplot.html

    HOSPITAL = pd.read_csv("hospital.txt", delimiter="\t")
    except_InfctRsk = HOSPITAL.drop(['InfctRsk'], axis = 1, inplace = False)  # 독립 변수
    sns.pairplot(HOSPITAL, x_vars = except_InfctRsk, y_vars = ["InfctRsk"])
    plt.show()

# --------------------------------- Problem 1 (2) ---------------------------------
def Problem_1_2():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    HOSPITAL = pd.read_table("hospital.txt", sep="\t")    

    correlation_matrix = HOSPITAL.corr()
    print(correlation_matrix["InfctRsk"])   # InfctRsk와의 상관계수 출력

    # 상관계수 시각화 코드
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation_matrix,     # 위에서 뽑은 데이터
                cbar=True,              # 오른쪽 컬러 막대 표시 여부
                annot=True,             # 차트 숫자 표시 여부
                annot_kws={'size': 7},  # 숫자 출력 크기 조절
                fmt='.2f',              # 숫자 출력 소수점 자리 조절
                cmap="coolwarm",        # Heatmap 종류
                square=True)            # 차트 정사각형 여부
    plt.title("Correlation Matrix")
    plt.tight_layout()  # figure 화면에 전부 들어오게 설정
    plt.show()

# --------------------------------- Problem 1 (3) ---------------------------------
def Problem_1_3():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    HOSPITAL = pd.read_table("hospital.txt", sep="\t")
    
    Y = HOSPITAL['InfctRsk']  # 종속 변수
    X = HOSPITAL.drop(['InfctRsk'], axis = 1, inplace = False)  # 독립 변수
    
    # ---------- ①번 결측치 여부 확인 ----------
    # isNULL = HOSPITAL.isnull().any().any()
    missing_values = HOSPITAL.isnull().sum()
    print(f'missing_values\n{missing_values}\n')
    
    # ---------- ②번 데이터 전처리 필요 여부 확인 ----------
    # 결측치를 보면 전부 0으로 확인 가능
    # 데이터 전처리가 필요하지 않은 상태
    
    # ---------- ③번 훈련용, 테스트용 데이터셋 분리----------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.3,        # 훈련 데이터와 평가 데이터를 7 : 3 비율로 분할
        random_state = 156)     # random의 seed를 156으로 고정
    
    # ---------- ④번 절편과 회귀계수 구하기 ----------
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    
    print(f'Y절편 model.intercept_\n{model.intercept_}\n')
    print(f'회귀계수값 model.coef_\n{model.coef_}\n')
    
    # ---------- ⑤번 MAE, MSE , RMSE, R2 평가지표를 통해 선형회귀분석 모델 평가 ----------
    MAE = mean_absolute_error(Y_test, Y_predict)
    MSE = mean_squared_error(Y_test, Y_predict)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test, Y_predict)
    
    print('MAE: {0:.3f},\tMSE: {1:.3f},\tRMSE: {2:.3f}'.format(MAE,MSE,RMSE))
    print('R2(Variancescore): {0:.3f}'.format(R2))
    
    # ---------- ⑥번 회귀분석 결과를 산점도와 선형회귀 그래프로 시각화 ----------
    fig, axs = plt.subplots(figsize=(8,7),          # subplot figure의 전체 사이즈
                          ncols=3,                  # subplot의 열 개수
                          nrows=4,                  # subplot의 행 개수
                          constrained_layout=True)  # X_trick 보기 위한 여백 설정
    
    x_features = ['ID',
                  'Stay',
                  'Age',
                  'Culture',
                  'Xray',
                  'Beds',
                  'MedSchool',
                  'Region',
                  'Census',
                  'Nurses',
                  'Facilities']
    
    for i, feature, in enumerate(x_features):
        row = i // 3
        col = i % 3
        sns.regplot(x = feature,
                    y = 'InfctRsk',
                    data = HOSPITAL,
                    ax = axs[row][col])
        
    plt.show()
    
    # ---------- ⑦번 분석 내용에 대한 설명 ----------
    # 해당 문제에서는 감염위험과 관련된 변수들을 사용하여 선형회귀분석을 수행했다.
    # 선형회귀분석은 종속변수와 독립변수 간의 선형적인 관계를 모델링하는 통계적 분석 방법으로
    # 감염위험변수(InfctRsk)를 종속변수로 설정하고, 나머지 11개의 변수를 독립변수로 설정하여 회귀분석을 수행했다.
    # (본래 독립변수란, 종속변수에 영향을 미치는 변수이지만 특별한 언급이 없어 전부 독립변수로 설정했다.)

    # 분석 결과를 토대로 회귀식을 작성하기 위해 절편과 회귀계수를 구하고,
    # 모델의 평가를 위해 MAE, MSE, RMSE, R2 평가지표를 사용하여 모델의 성능을 평가했다.
    # 이를 통해 모델의 예측 정확도와 오차를 확인할 수 있었습니다.

    # 마지막으로 회귀분석 결과를 시각화하기 위해 산점도와 선형회귀를 시각화했다.
    # 이를 통해 독립변수와 종속변수 간의 관계를 시각적으로 파악할 수 있었습니다.

    # 위의 분석 과정을 통해 감염위험과 관련된 변수들과의 상관관계와 영향력을 파악하고,
    # 모델을 구축하여 감염위험을 예측하고 해석할 수 있다.
    
# --------------------------------- Problem 1 (4) ---------------------------------
def Problem_1_4():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    HOSPITAL = pd.read_table("hospital.txt", sep="\t")
    
    Y = HOSPITAL['InfctRsk']  # 종속 변수
    X = HOSPITAL.drop(['InfctRsk'], axis = 1, inplace = False)  # 독립 변수
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.3,        # 훈련 데이터와 평가 데이터를 7 : 3 비율로 분할
        random_state = 156)     # random의 seed를 156으로 고정
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    
    # ---------- ①번 절편과 회귀계수 구하기 ----------
    print(f'Y절편 model.intercept_\n{model.intercept_}\n')
    print(f'회귀계수값 model.coef_\n{model.coef_}\n')
    
    coef, passIndex = list(model.coef_), []
    for index, value in enumerate(coef):
        if value > 0.01:
            passIndex.append(index)
            #print(f'index : {index}, value : {value}')
    
    # ---------- ②번 MAE, MSE , RMSE, R2 평가지표를 통해 선형회귀분석 모델 평가 ----------
    MAE = mean_absolute_error(Y_test,Y_predict)
    MSE = mean_squared_error(Y_test,Y_predict)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Y_predict)
    
    print('MAE: {0:.3f},\tMSE: {1:.3f},\tRMSE: {2:.3f}'.format(MAE,MSE,RMSE))
    print('R2(Variancescore): {0:.3f}'.format(R2))
    
    fig, axs = plt.subplots(figsize=(8,7),          # subplot figure의 전체 사이즈
                          ncols=3,                  # subplot의 열 개수
                          nrows=4,                  # subplot의 행 개수
                          constrained_layout=True)  # X_trick 보기 위한 여백 설정
    
    x_features = ['ID',
                  'Stay',
                  'Age',
                  'Culture',
                  'Xray',
                  'Beds',
                  'MedSchool',
                  'Region',
                  'Census',
                  'Nurses',
                  'Facilities']
    
    # ---------- ③번 회귀분석 결과를 산점도와 선형회귀 그래프로 시각화 ----------
    for i, feature, in enumerate(x_features):
        if i in passIndex:
            row = i // 3
            col = i % 3
            sns.regplot(x = feature,
                        y = 'InfctRsk',
                        data = HOSPITAL,
                        ax = axs[row][col])
        
    plt.show()
    
    # ---------- ④번 분석 내용에 대한 설명 ----------
    # 데이터 전처리: hospital.txt 파일을 불러와서 종속 변수(Y)와 독립 변수(X)로 나눈다.
    # 데이터 분할: 훈련 데이터와 평가 데이터로 분할, 훈련 데이터는 모델 학습에, 평가 데이터는 모델의 성능 평가에 사용한다.
    # 회귀식 작성: 훈련 데이터를 사용하여 선형 회귀 모델을 학습한다. 그후 감염위험에 대한 회귀식의 절편과 회귀계수를 구한다.
    # 변수 선택: 회귀계수가 0.01 이상인 변수들을 선택한다. 이를 통해 중요한 변수들을 선정하고, 나머지 변수들을 제외한다.
    # 성능 평가: MAE, MSE, RMSE, R2의 평가지표를 이용하여 모델의 예측 성능을 확인한다.
    # 시각화: 선택한 변수들과 종속 변수 간의 산점도와 선형 회귀 그래프를 시각화한다.
    
    # 위의 분석 과정으로 선택된 변수들과 감염위험변수(InfctRsk) 간의 선형회귀분석 모델을 구축하고 평가한다.
    # 이를 통해 변수들 간의 관계와 예측 성능을 확인할 수 있다.
    
# --------------------------------- Problem 1 (5) ---------------------------------
def Problem_1_5():
    answer = ['''
    Problem_1_3() 함수는 주어진 데이터의 모든 변수를 독립 변수로 활용하여 회귀분석을 수행한다.
    따라서 선형회귀식에는 모든 변수의 회귀계수를 포함하며, 선형회귀분석의 결과와 평가지표를 기반으로 모델의 예측 성능을 평가한다.
    또한, 시각화를 통해 독립 변수와 종속 변수 간의 관계를 파악할 수 있습니다.

    Problem_1_4() 함수는 Problem_1_3() 함수와 달리 변수 선택의 관점에서 차이가 있다.
    회귀계수가 0.01 이상인 변수들만 선택하여 회귀분석을 수행한다(중요한 변수들을 선정하고 나머지 변수들은 제외한다).
    따라서 선형회귀식에는 선택한 변수들의 회귀계수만 포함하며, 선형회귀분석 결과 및 평가지표 또한 마찬가지이다.
    시각화에서는 선택한 변수들과 종속 변수 간의 관계만을 확인할 수 있다.

    Problem_1_3() 함수와 Problem_1_4() 함수는 동일한 데이터를 활용하여 선형회귀분석을 수행한다.
    하지만 이를 비교하면, 결과 해석 및 변수 선택의 관점에서 차이가 발생한다.
    Problem_1_3()은 모든 변수를 활용하여 분석하기 때문에, 변수 간의 상관관계와 영향력을 포괄적으로 파악할 수 있다.
    반면 Problem_1_4()은 중요한 변수들을 선정하고 분석하기 때문에, 불필요한 변수를 제외하고 선형회귀분석을 진행할 수 있다.
    선택한 변수들에 대한 회귀계수를 통해 해당 변수들이 종속 변수에 어떠한 영향을 미치는지 파악할 수 있다.

    결과적으로, Problem_1_3() 함수와 Problem_1_4() 함수는 변수 선택의 차이를 가지고 있다.
    각각의 결과를 통해 모델의 성능과 변수들 간의 관계를 다른 관점에서 파악할 수 있다.
    ''']
    
    for text in answer[0]:
        if text == "\n":
            print()
        else:
            print(text, end="")
    
# --------------------------------- Problem 2 ---------------------------------
def Problem_2():
    import pandas as pd
    
    from sklearn.linear_model import LogisticRegression     # 로지스틱 회귀 모듈
    from sklearn.model_selection import train_test_split    # 데이터 분할 모듈
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    HAND_SEX = pd.read_table("handspan.txt", sep="\t")

    # ---------- ①번 결측치 여부 확인 ----------
    # isNULL = HAND_SEX.isnull().any().any()
    missing_values = HAND_SEX.isnull().sum()
    print(f'missing_values\n{missing_values}\n')
    
    # ---------- ②번 데이터 전처리 필요 여부 확인 ----------
    # 결측치를 보면 전부 0으로 확인 가능
    # 데이터 전처리가 필요하지 않은 상태
    
    # ---------- ③번 훈련용, 테스트용 데이터셋 분리 ----------
    X = HAND_SEX[['Height', 'HandSpan']]
    Y = HAND_SEX['Sex']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.3,
        random_state = 42)
    
    model = LogisticRegression()  # 모델 생성
    model.fit(X_train, Y_train)   # 모델 훈련
    
    # trainScore = model.score(X_train, Y_train) # 학습 데이터로 모델 평가
    # print(f'trainScore:\t{trainScore}')
    # testScore = model.score(X_test, Y_test)   # 평가 데이터로 모델 평가
    # print(f'testScore:\t{testScore}')
    
    # ---------- ④번 모형식의 절편과 회귀계수 구하기 ----------
    print(f'Y절편 model.intercept_\t{model.intercept_}')
    print(f'회귀계수값 model.coef_\t{model.coef_}\n')
    
    # ---------- ⑤번 평가데이터를 이용하여 예측값 구하기 ----------
    Y_predict = model.predict(X_test)
    #print(Y_predict)
    
    # ---------- ⑥번 오차행렬 ----------
    confusion = confusion_matrix(Y_test, Y_predict)                     # 혼동행렬
    accuracy = accuracy_score(Y_test, Y_predict)                        # 정확도
    precision = precision_score(Y_test, Y_predict, pos_label='Female')  # 정밀도
    recall = recall_score(Y_test, Y_predict, pos_label="Female")        # 재현율
    f1 = f1_score(Y_test, Y_predict, pos_label='Female')                # f1 score
    # roc_auc = roc_auc_score(pd.factorize(Y_test)[0], pd.factorize(Y_predict)[0])

    Y_test_rocauc = [1 if sex == "Female" else 0 for sex in Y_test]
    Y_predict_rocauc = [1 if sex == "Female" else 0 for sex in Y_predict]
    roc_auc = roc_auc_score(Y_test_rocauc, Y_predict_rocauc) # ROC 기반 AUC score
    
    print(f'confusion matrix\n{confusion}')
    print(f'accuracy\t{accuracy}')
    print(f'precision\t{precision}')
    print(f'recall\t\t{recall}')
    print(f'f1 score\t{f1}')
    print(f'ROC AUC\t\t{roc_auc}')    
    
    answer = ["""
    주어진 데이터셋에서 "Height"(키)와 "HandSpan"(손 한뼘의 길이) 변수로 성별을 분류하는 로지스틱 회귀를 수행했다.

    먼저, 데이터셋을 로드하고 결측치 여부를 확인하고 이를 기반으로 데이터 전처리 여부를 판단했다.
    전처리가 필요한 경우, 변수 변환, 이상치 처리, 범주형 변수 인코딩 등의 작업을 수행한다.
    하지만 주어진 데이터셋에는 결측치가 없는 것으로 나타났기에, 전처리가 필요없어 다른 작업은 생략했다.

    훈련을 위해 데이터셋을 훈련용과 테스트용으로 분리했다.
    일반적으로 데이터를 훈련용과 테스트용으로 나누는 이유는 모델의 일반화 성능을 평가하기 위해서이다.
    이번 분석에서는 데이터의 70%를 훈련용 데이터셋으로, 나머지 30%를 테스트용 데이터셋으로 나누었다.

    다음으로 로지스틱 회귀분석을 수행하여 모형식의 절편과 회귀계수를 구했다.
    회귀계수는 각 변수의 영향력을 나타낸다.
    양수일 경우 변수가 성별을 예측하는데 긍정적인 영향을, 음수일 경우 부정적인 영향을 준다.

    로지스틱 회귀모델을 통해 성별을 예측하고,
    이를 기반으로 평가 지표인 정확도, 정밀도, 재현율, F1 스코어, ROC 기반 AUC 스코어를 계산했다.
    이러한 평가 지표를 통해 모델의 예측 성능을 평가했다.

    결과적으로 주어진 "Height"와 "HandSpan" 기반으로 일정 수준의 성별 예측 성능을 보여주었다.
    예측 결과에 따라 정확도, 정밀도, 재현율, F1 스코어, AUC 스코어 등의 지표를 확인할 수 있으며,
    이러한 지표를 통해 모델의 성능을 평가할 수 있다.
    """]
    
    for text in answer[0]:
        if text == "\n":
            print()
        else:
            print(text, end="")
    
    # ---------- ⑦번 분석내용에 대하여 설명 ----------
    # 주어진 데이터셋에서 "Height"(키)와 "HandSpan"(손 한뼘의 길이) 변수로 성별을 분류하는 로지스틱 회귀를 수행했다.

    # 먼저, 데이터셋을 로드하고 결측치 여부를 확인하고 이를 기반으로 데이터 전처리 여부를 판단했다.
    # 전처리가 필요한 경우, 변수 변환, 이상치 처리, 범주형 변수 인코딩 등의 작업을 수행한다.
    # 하지만 주어진 데이터셋에는 결측치가 없는 것으로 나타났기에, 전처리가 필요없어 다른 작업은 생략했다.

    # 훈련을 위해 데이터셋을 훈련용과 테스트용으로 분리했다.
    # 일반적으로 데이터를 훈련용과 테스트용으로 나누는 이유는 모델의 일반화 성능을 평가하기 위해서이다.
    # 이번 분석에서는 데이터의 70%를 훈련용 데이터셋으로, 나머지 30%를 테스트용 데이터셋으로 나누었다.

    # 다음으로 로지스틱 회귀분석을 수행하여 모형식의 절편과 회귀계수를 구했다.
    # 회귀계수는 각 변수의 영향력을 나타낸다.
    # 양수일 경우 변수가 성별을 예측하는데 긍정적인 영향을, 음수일 경우 부정적인 영향을 준다.

    # 로지스틱 회귀모델을 통해 성별을 예측하고,
    # 이를 기반으로 평가 지표인 정확도, 정밀도, 재현율, F1 스코어, ROC 기반 AUC 스코어를 계산했다.
    # 이러한 평가 지표를 통해 모델의 예측 성능을 평가했다.

    # 결과적으로 주어진 "Height"와 "HandSpan" 기반으로 일정 수준의 성별 예측 성능을 보여주었다.
    # 예측 결과에 따라 정확도, 정밀도, 재현율, F1 스코어, AUC 스코어 등의 지표를 확인할 수 있으며,
    # 이러한 지표를 통해 모델의 성능을 평가할 수 있다.
    

# --------------------------------- Problem 3 ---------------------------------
def Problem_3():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.tree import DecisionTreeClassifier         # 결정 트리 모듈
    from sklearn.model_selection import train_test_split    # 데이터 분할 모듈
    from sklearn.metrics import accuracy_score              # 정확도 측정 모듈
    from sklearn.model_selection import GridSearchCV        # 하이퍼 매개변수 모듈
    from sklearn.tree import export_graphviz                # 트리구조 시각화 모듈
    #import graphviz                                         # 트리구조 시각화 모듈
    
    
    MPG = sns.load_dataset('mpg')
    
    # ---------- ⓪ 구조 확인용 출력 테스트(breakpoint() 함수로도 확인 가능) ----------
    # usa, europe, japan = 0, 0, 0
    # for country in MPG['origin']:
    #     if country == "usa": usa += 1
    #     if country == "europe": europe += 1
    #     if country == "japan": japan += 1
    # print(usa, europe, japan) # 249 70 79
    
    # print(MPG.head())    # 구조 파악용 상위 5개 데이터 출력
    # print(MPG.shape)     # 구조 파악용 (행, 열) 개수 출력
    # print(MPG.info())    # mpg 데이터 프레임 정보 검증용
    
    # ---------- ① 결측치 여부 확인 ----------
    # isNULL = MPG.isnull().any().any()
    missing_values = MPG.isnull().sum()
    print(f'missing_values\n{missing_values}\n')
    
    # ---------- ② 데이터 전처리 필요 여부 확인 ----------
    # 그외 데이터 전처리 함수들: dropna(), interpolate(), ffill(), bfill()
    # fillna()는 mean(), median(), mode() 함수로 평균, 중앙값, 최빈값으로 대체
    MPG['horsepower'].fillna(MPG['horsepower'].mean(), inplace=True)
    
    # ---------- ③ 훈련용, 테스트용 데이터셋 분리 ----------
    Y = MPG['origin']
    X = MPG[['mpg',
             'cylinders',
             'displacement',
             'horsepower',
             'weight',
             'acceleration',
             'model_year']]     # 예측값(origin), 상관없는 값(name)은 제외
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.3,
        random_state = 68)
    
    # ---------- ④ 결정트리 분류분석 모델 생성, 트리 모형 적합(훈련), 예측 수행 ----------
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    
    # ---------- ⑤ 생성된 결정 트리 모델의 분류 정확도 성능을 확인 ----------
    accuracy = accuracy_score(Y_test, Y_predict)
    print(f'Accuracy:\t{accuracy}')
    
    # ---------- ⑥ GridSearchCV 모듈로 정확도 검사, 최적의 하이퍼 매개변수 탐색 ----------
    params = {'max_depth': [3, 4, 5, 6, 7]}
    grid_cv = GridSearchCV(model,                       # 검사에 사용할 데이터
                           param_grid = params,         # 탐색할 매개변수 그리드
                           scoring = 'accuracy',        # 모델 성능 평가 지표
                           cv = 5,                      # 교차 검증 Fold 수
                           return_train_score = True)   # 훈련 데이터 점수 반환 여부
    grid_cv.fit(X_train, Y_train)
    
    cv_results = pd.DataFrame(grid_cv.cv_results_)
    # print(cv_results[['param_max_depth', 'mean_test_score', 'mean_train_score']])
    
    best_params = grid_cv.best_params_
    print(f'Best Parameter:\t{best_params}')
    
    # ---------- ⑦ Decision Tree Classifier의 주요매개변수들로 최고의 평균 정확도 탐색 ----------
    best_score = grid_cv.best_score_
    print(f'Best Score:\t{best_score}')
    
    # ---------- ⑧ 최적 모델 grid_cv.best_estimator_로 테스트 데이터 예측 수행 ----------
    # best_estimator_: GridSearchCV에서 최적의 매개변수 조합으로 학습한 모델
    best_model = grid_cv.best_estimator_
    best_Y_predict = best_model.predict(X_test)
    best_accuracy = accuracy_score(Y_test, best_Y_predict)
    print(f'Best Accuracy:\t{best_accuracy}')
    
    # ---------- ⑨ feature_importances_ 속성으로 각 피처의 중요도, 중요도가 높은 5개 피처 그래프 ----------
    feature_importances = best_model.feature_importances_       #
    
    # feature_importances_series = pd.Series(feature_importances, index = X_train.columns)
    # feature_top5 = feature_importances_series.sort_values(ascending = False)[:5]
    feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_top5 = feature_importances_df.sort_values(by='Importance', ascending=False).head(5)
    
    print(feature_top5)
        
    plt.figure(figsize = (10, 5))
    plt.title('Feature Top 5')
    # sns.barplot(x = feature_top5, y = feature_top5.index)
    sns.barplot(x = 'Importance', y='Feature', data=feature_top5)
    plt.tight_layout()
    plt.show()

    # ---------- ⑩ Graphviz 패키지 : 결정트리 모델의 트리구조를 그림으로 시각화 ----------
    export_graphviz(best_model,                         # 시각화 할 모델 선택
                    out_file = "tree.dot",              # 생성 파일 경로 및 이름 지정
                    class_names = best_model.classes_,  # 각 클래스 이름 지정
                    feature_names = X.columns,          # 각 특징 이름 지정
                    impurity = True,                    # 불순도 표시 여부
                    filled = True)                      # 노드 색상 채우기 여부
    
    with open("tree.dot") as f:
        dot_graph = f.read()
    #src = graphviz.Source(dot_graph)
    #src.view()
    
    # ---------- ⑪ 분석 내용에 대한 설명 ----------
    # seaborn의 'mpg' 데이터를 받아온 뒤, 데이터에서 결측치를 확인한다.
    # 'horsepower' feature의 결측치가 있음을 확인하고 평균값으로 대체한다.
    # 전처리한 데이터를 훈련용 데이터와 테스트용 데이터로 분할한다.
    # 이때 독립변수로는 종속변수의 origin과 관계없는 name은 제외하여 데이터를 구분한다.
    # DecisionTreeClassifier를 사용하여 결정 트리 분류 모델을 생성한다.
    # 훈련용 데이터로 모델을 학습시킨 후 테스트용 데이터를 예측한다.
    # 예측 결과와 실제값을 정확도(accuracy) 성능 지표로 비교하여 분류 정확도를 계산한다.
    # GridSearchCV를 사용하여 결정 트리 모델의 최적의 매개변수를 탐색하고, 교차 검증을 수행한다.
    # 이때 max_depth는 임의의 값 3, 4, 5, 6, 7의 다섯 개로 선언해주었다.
    # 위 매개변수들 중 최적의 매개변수를 가진 모델의 최고 평균 정확도와 최적 하이퍼 매개변수를 출력한다.
    # best_estimator_를 사용하여 최적 모델을 얻고, 이를 테스트 데이터로 예측하여 분류 정확도를 계산한다.
    # 여기서 최적의 매개변수로 조합한 모델의 정확도(accuracy) 성능 지표로 정확도를 계산하는 것이다.
    # 최적 모델에서 추출한 feature 중요도를 확인하고, 상위 5개 feature를 시각화하여 출력한다.
    # export_graphviz() 함수로 결정 트리 모델의 트리 구조를 dot 파일로 저장한다.
    # 이를 graphviz 패키지를 사용하여 시각화하되, view() 함수를 통하여 웹 페이지에서 pdf로도 볼 수 있게 한다.

    # 위 분석 내용으로 'mpg' 데이터셋을 활용하여 결정 트리 분류 모델을 생성하고,
    # 최적의 매개변수를 탐색하여 모델 성능을 개선하는 과정을 이해할 수 있었다.
    # 또한, feature 중요도를 확인하여 데이터셋의 특징을 시각화하여 효율적으로 흐름을 파악할 수 있었다.

# --------------------------------- Problem 4 ---------------------------------
def Problem_4():
    # ch06 데이터 전처리_수정2.pdf 참고
    
    answer = ['''
    주어진 zi = (xi - min(xi)) / (max(x) - min(x)) 식은 Min-Max 스케일링 식이다.
    이는 다른 말로는 정규화(Normalization)이라고도 한다.
    정규화 식을 사용해 데이터 스케일링을 진행하면 모든 값이 0과 1 사이 값으로 조정된다.
    
    ㆍ 분모의 xi는 스케일링 하려는 개별 데이터의 값이다.
    ㆍ min(x)는 해당 열의 최솟값이다.
    ㆍ max(x)는 해당 열의 최댓값이다.
    ㆍ zi는 스케일링된 이후의 결괏값이다.
    
    Min-Max 스케일링은 데이터의 상대적인 크기와 분포를 유지하며 값을 조정하는 장점이 있다.
    하지만 단점으로는 이상치(Outlier)에 민감할 수 있다는 것과,
    최소&최대가 지속적으로 바뀌는 동적 데이터에 적용이 어렵다는 단점이 있다.
    
    이를 데이터에 적용한다면 A열과 B열의 값들을 아래와 같이 변한다.
    변경 전 A열: [14.00  90.20  90.95   96.27   91.21]
    변경 후 A열: [ /* 이거는 계산해서 넣어라 */ ]
    
    변경 전 B열: [103.02    107.26  110.35  114.23  114.68]
    변경 후 B열: [ /* 이거는 계산해서 넣어라 */ ]
    
    C열은 문자열 값이므로 스케일링되지 않는다.
    Min-Max 스케일링은 수치형에 적용되며 범주형에는 적용되지 않는다.
    따라서 C열의 값은 스케일링되지 않고 원래 값을 그대로 유지한다.
    ''']
    
    for text in answer[0]:
        if text == "\n":
            print()
        else:
            print(text, end="")
            
# --------------------------------- Problem 5 ---------------------------------
def Problem_5():
    # ch06 데이터 전처리_수정2.pdf 참고
    
    answer = ['''
    데이터를 정규화하는 2가지 방법이 z-스코어와 최솟값-최댓값 방법이다.
    최솟값-최대값 정규화(min-max normalization)는 0에서 1(또는 지정값)까지 변화하는 방법이고,
    z-스코어 정규화(z-score normalization)는 표준 정규 분포를 사용하여 변화하는 방법이다.
    이 둘은 다음과 같은 차이점이 존재한다.
    
    1. 정규화 범위
    최솟값-최대값 정규화는 최솟값과 최댓값을 사용하여 데이터를 정규화한다.
    데이터는 최솟값과 최댓값 사이의 범위에 매핑된다. 즉, 결과 값은 0과 1(또는 지정값) 사이의 범위로 제한된다.
    z-스코어 정규화는 평균과 표준편차를 사용하여 데이터를 정규화한다.
    정규화된 값은 평균을 기준으로 얼마나 떨어져 있는지를 나타내는 표준화된 z-스코어이다.
    결과 값은 평균을 기준으로 정규분포를 따르며, 음수와 양수의 범위로 확장힌다.
    
    2. 이상치에 대한 영향
    최솟값-최대값 정규화는 최솟값과 최댓값에만 의존하므로 이상치에 영향을 덜 받는다.
    이상치가 최솟값 또는 최댓값보다 크거나 작더라도 정규화 범위는 유지된다.
    z-스코어 정규화는 평균과 표준편차를 기반으로 하므로 이상치에 민감할 수 있다.
    이상치는 평균과 표준편차에 큰 영향을 주어 정규화된 값에 큰 변화를 초래할 수 있다.
    
    3. 분포의 유지
    최솟값-최대값 정규화는 데이터를 0과 1 사이로 변환하므로 분포의 형태는 유지되지 않는다.
    z-스코어 정규화는 정규분포를 따르기에 데이터의 분포를 유지하면서 정규화한다.
    
    4. 데이터의 해석 가능성
    최솟값-최대값 정규화로 정규화된 값은 원래 데이터의 비율을 유지, 해석이 원래 데이터와 일치한다.
    z-스코어 정규화는 평균과 표준편차를 사용하므로 해석이 원래 데이터와 다소 다를 수 있다.
    ''']
    
    for text in answer[0]:
        if text == "\n":
            print()
        else:
            print(text, end="")

# --------------------------------- Problem 6 ---------------------------------
def Problem_6():    
    import pandas as pd
    
    data = {
        'source': [0, 1, 2],
        'target': [2, 2, 3],
        'weight': [3, 4, 5],
        'color': ['red', 'blue', 'blue']
    }

    df = pd.DataFrame(data)
    
    # ---------- ①번 pd.get_dummies(edges).iloc[:,3:] ----------
    df_encoded_1 = pd.get_dummies(df).iloc[:, 3:]
    print(df_encoded_1, "\n")
        
    # ---------- ②번 pd.get_dummies(edges["color"]) ----------
    df_encoded_2 = pd.get_dummies(df["color"])
    print(df_encoded_2, "\n")

    # ---------- ③번 pd.get_dummies(edges[["color"]]) ----------
    df_encoded_3 = pd.get_dummies(df[["color"]])
    print(df_encoded_3, "\n")

    # ---------- ④번 pd.get_dummies(edges["color"], prefix="color") ----------
    df_encoded_4 = pd.get_dummies(df["color"], prefix="color")
    print(df_encoded_4)

# --------------------------------- Problem 7 ---------------------------------
def Problem_7():
    # ch06 데이터 전처리_수정2.pdf 참고
    import numpy as np
    import pandas as pd
    
    raw_data = {
        'first_name': ['Jason', np.nan, 'Tine', 'Jake', 'Amy'],
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'],
        'age': [42, np.nan, 36, 24, 73],
        'sex': ['m', np.nan, 'f', 'm', 'f'],
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]
    }
    
    df = pd.DataFrame(raw_data, columns = [
        'first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'
    ])
    
    result = df.isnull().sum()
    print(result)

# --------------------------------- Problem 8 ---------------------------------
def Problem_8():
    import math
    import pandas as pd

    data = {
        'OUTLOOK': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 
                    'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'],
        'TEMPERATURE': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 
                        'Mild', 'Mild', 'Hot', 'Mild'],
        'HUMIDITY': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 
                     'Normal', 'High', 'Normal', 'High'],
        'WINDY': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'PLAY_GOLF': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }

    df = pd.DataFrame(data)
    
    value_counts = df['PLAY_GOLF'].value_counts()   # PLAY_GOLF value 빈도 계산
    
    entropy = 0                 # PLAY_GOLF 엔트로피 변수 선언
    total_count = df.shape[0]   # (14, 5) 크기의 데이터에서 행 개수 추출
    
    # Entropy equation : Entropy(A) = - Σ P(x) log2 P(x)
    for count in value_counts:
        probability = count / total_count
        entropy -= probability * math.log2(probability)
        
    print(f'Entropy: {entropy}')

# --------------------------------- Main ---------------------------------
Problem_1_1()
Problem_1_2()
Problem_1_3()
Problem_1_4()
Problem_1_5()
Problem_2()
Problem_3()
Problem_4()
Problem_5()
Problem_6()
Problem_7()
Problem_8()