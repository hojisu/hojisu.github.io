<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 다중공선성


### Summary
- 다중공선성은 독립변수들이 독립이 아니라 상관관계가 강할 경우 발생한다. 다중공선성이 있으면 조건수가 증가한다. 
- 과최적화 방지 방법으로는 변수 선택법에 의한 변수 제거, PCA 방법으로 의존적인 성분 제거, 정규화 방법이 있다.
- VIF는 독립변수를 다른 독립변수로 선형회귀한 성능을 나타낸 것이다. 다른 변수에 의존적일 수록 VIF가 커진다
_______________________

# 다중공선성과 변수 선택

다중공선성(multicollinearity)은 독립 변수의 일부가 다른 독립 변수의 조합으로 표현될 수 있는 경우이다.

독립 변수들이 서로 독립이 아니라 상호상관관계가 강한 경우 발생하고 독립 변수의 공분산 행렬이 full rank 이어야 한다는 조건을 침해한다.

다중 공선성이 있으면 독립변수의 공분산 행렬의 조건수(conditional number)가 증가한다.

학습용 데이터/ 검증용 데이터로 나누어 회귀분석 성능을 비교하면 과최적화를 확인할 수 있다.

~~~python
# 상관계수 행렬로 확인
dfX.corr()
~~~

~~~python
# heatmap으로 상관계수 그래프 
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(dfX.corr(), annot=True, cmap=cmap)
plt.show()
~~~

~~~python
def calc_r2(df_test, result):
    target = df.loc[df_test.index].TOTEMP
    predict_test = result.predict(df_test)
    RSS = ((predict_test - target)**2).sum()
    TSS = ((target - target.mean())**2).sum()
    return 1 - RSS / TSS


test1 = []
for i in range(10):
    df_train, df_test, result = get_model1(i)
    test1.append(calc_r2(df_test, result))

test1

#결과
[0.9815050656837723,
 0.9738497543069347,
 0.9879366369871746,
 0.7588861967897188,
 0.980720608930437,
 0.8937889315168234,
 0.8798563810651999,
 0.9314665778963799,
 0.8608525682180641,
 0.9677198735170137]
~~~

### 과최적화 방지 방법
- 변수 선택법으로 의존적인 변수 삭제
- PCA(principal component analysis) 방법으로 의존적인 성분 삭제
- 정규화(regularized) 방법 사용

### VIF(Variance Inflation Factor)

VIF는 독립변수를 다른 독립변수로 선형회귀한 성능을 나타낸 것이다. 다른 변수에 의존적일 수록 VIF가 커진다

StatsModels에서는 `variance_inflation_factor` 명령으로 VIF를 계산한다

~~~
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    dfX.values, i) for i in range(dfX.shape[1])]
vif["features"] = dfX.columns
vif
~~~

