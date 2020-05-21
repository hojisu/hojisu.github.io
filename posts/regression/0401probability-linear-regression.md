<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 확률론적 선형 회귀모형

### Summary

- 부트스트래핑은 회귀분석에 사용한 표본 데이터가 달라질 때 결과가 어느정도 영향을 받는지 알기 위한 방법이다.
- 선형회귀모형의 가정은 가우시안 정규분포를 따르고 잡음의 기댓값은 항상 0이고 표본의 잡음들의 공분산 값도 항상 0이고 표본의 잡음들의 분산은 항상 같다. 
- 단일 계수 t-검정의 귀무가설은 ''가중치가 0이다'' 로 유의확율이 0에 가깝다면 귀무가설은 기각이므로 가중치가 0일 확률이 적다. 두 독립변수의 계수를 비교할 때 쓸 수 있다. 범주형 독립변수의 범주값이 가지는 유의미함을 판단하는데 유용하다. 
- 회귀분석 F-검정의 귀무가설은 ''가중치가 모두 0이다'' 로 유의 확률이 작으면 작을수록 더 강력하게 기각된 것이므로 더 의미가 있는 모형이라고 할 수 있다. 
- 확률론적 선형회귀모형을 사용하면 부트스트래핑처럼 많은 계산을 하지 않아도 빠르고 안정적으로 가중치 추정값의 오차(std err)를 구할 수 있다. 이 값은 부트스트래핑을 사용하여 계산한 것이 아니라 확률론적 선형회귀모형을 사용하여 계산한 값이다.

______

### 부트스트래핑

부트스트래핑(bootstrapping)은 회귀분석에 사용한 표본 데이터가 달라질 때 회귀분석의 결과는 어느 정도 영향을 받는지 알기 위한 방법이다. 

기본의 데이터를 재표본화(re-sampling)하여 여러가지 다양한 표본 데이터 집합을 만드는 방법을 사용한다. 재표본화는 기존의 N개의 데이터에서 다시 N개의 데이터를 선택하되 중복 선택도 가능하게 한다. 

### 확률론적 선형 회귀모형

statsmodels 의 summary 메서드로 출력한 보고서에서 `ste err`이라는 이름으로 표시

#### (1) 선형 정규 분포 가정

선형 회귀분석의 기본 가정은 종속 변수 $$y$$ 가 독립 변수 $$x$$ 의 선형 조합으로 결정되는 기댓값과 고정된 분산 $$\sigma^2$$ 을 가지는 가우시안 정규 분포라는 것이다. 

$$
 y ~ N(w^Tx, \sigma^2)
$$

$$y$$ 의 확률 밀도 함수는 다음처럼 쓸 수 있다. 이 식에서 모수 벡터 $$\theta=(w, \sigma^2)$$ 이다.

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid w^Tx, \sigma^2 )
$$

이 관계식을 잡은(disturbance) $$\epsilon$$ 개념으로 변환하면 더 간단하게 표현 할 수 있다.

$$
\epsilon = y - w^Tx\\
p(\epsilon \mid \theta) = \mathcal{N}(0, \sigma^2 )
$$

**x, y 중 어느 것도 그 자체로 정규 분포일 필요는 없다**

y도 x에 대해 조건부로 정규 분포를 이루는 것이다.

#### (2) 외생성(Exogenity) 가정

잡음 $$ϵ$$의 기댓값은 독립 변수 $$x$$의 크기에 상관없이 항상 0이라고 가정한다. 이를 외생성(Exogeneity) 가정이라고 한다.

$$
\text{E}[\epsilon \mid x] = 0
$$

#### (3) 조건부 독립 가정

 $$i$$번째 표본의 잡음 $$ϵ_i$$와 $$j$$번째 표본의 잡음 $$ϵ_j$$의 공분산 값이 $$x$$와 상관없이 항상 0이라고 가정한다.

$$
\text{Cov}[\epsilon_i, \epsilon_j \mid x] = 0 \;\; (i,j=1,2,\ldots,N)
$$

#### (4) 등분산성 가정

- $$i$$번째 표본의 잡음 $$ϵ_i$$와 $$j$$번째 표본의 잡음 $$ϵ_j$$의 분산 값이 표본과 상관없이 항상 같다고 가정한다.

$$
\text{Cov}[\epsilon] = \text{E}[\epsilon^{} \epsilon^T] = \sigma^2I
$$

### 잔차의 분포

확률론적 선형 회귀모형에 따르면 회귀분석에서 생기는 잔차 $$e = y - w^Tx$$도 정규분포를 따른다. 

확률룬적 선형회귀모형의 잡음과 잔차는 다음과 같은 관계를 가진다.

$$
\hat y = X \hat w = X(X^TX)^{-1}X^Ty = Hy
$$

행렬 H은 Hat 행렬 혹은 projection 행렬 또는 influence(영향도)행렬이라고 부르는 대칭행렬이다. 

Hat 행렬을 이용하면 잔차는 다음처럼 표현된다.
$$
e = y - \hat y = y - Hy = (I-H)y = My
$$

행렬 M은 잔차(residual)행렬이라고 부른다. 
확률론적 선형회귀 모형의 가정을 적용하면,

$$
e = My = M(Xw + \epsilon) = MXw + M \epsilon
$$

그런데 MX = 0 에서 $$e = M \epsilon$$ 이다.

**잔차 $$e$$는 잡음 $$ϵ$$의 선형 변환(linear transform)**이다.
정규분포의 선형변환은 마차간지로 정규분포이므로 잔차도 정규분포를 따른다.

~~~python
# 잔차 정규성 검정
test = sm.stats.omni_normtest(result.resid)
for xi in zip(['Chi^2', 'P-value'], test):
    print("%-12s: %6.3f" % xi)
~~~

오차의 기대값이 x와 상관없이 0이므로 **잔차의 기댓값도 x와 상관없이 0이어야 한다.**

$$
E[e|x] = 0
$$

### 회귀 계수의 표준 오차

가중치의 에측치 $$\hat{w}$$는 정규분포 확률변수인 $$\epsilon$$의 선형변환이므로 정규분포를 따른다.

$$
\hat = (X^TX)^{-1}X^Ty
     = (X^TX)^{-1}X^T(Xw + \epsilon)
     = w + (X^TX)^{-1}X^TE[ \epsilon ]
     = w
$$

$$\hat w$$의 기댓값은 

$$
E[ \hat{w} ] = E[ w + (X^TX)^{-1}X^T \epsilon ]
            = w + (X^TX)^{-1}X^TE[ \epsilon ]
            = w
$$

따라서 $$\hat w$$의 w의 비편향 추정값(unbiased estimate)이다. 

$$\hat w$$의 공분산은 

$$
Cov[ \hat{w} ]  \\
E[(\hat{w} - w)(\hat{w} - w)^T]  \\
E[((X^TX)^{-1} X^T \epsilon)((X^TX)^{-1} X^T \epsilon)^T] \\ 
E[(X^TX)^{-1} X^T \epsilon \epsilon^T X(X^TX)^{−1} ] \\
(X^TX)^{-1} X^T E[\epsilon \epsilon^T] X(X^TX)^{−1} \\
(X^TX)^{-1} X^T (\sigma^2 I) X(X^TX)^{−1} \\
\sigma^2  (X^TX)^{-1} \\
$$

그런데 잡음의 분산 $$\text{E}[ \epsilon^2 ] = \sigma^2$$의 값은 알지 못하므로 다음과 같이 잔차의 분산 $$E[ \epsilon^2 ]$$으로부터 추정한다.

$$
\text{E}[ e^2 ] 
\text{E}[ (M\epsilon)^2 ] \\
\text{E}[(\epsilon^T M^T)(M\epsilon)] \\
\text{E}[ \epsilon^T M \epsilon] \\
\text{E}[ \text{tr}(\epsilon^T M \epsilon) ] \\
\text{tr}( \text{E}[ M \epsilon \epsilon^T ]) \\
\text{tr}( M \text{E}[\epsilon \epsilon^T ]) \\
\text{tr}( M \sigma^2 I ) \\
\sigma^2 \text{tr}(M) \\
\sigma^2 \text{tr}(I - X(X^TX)^{-1}X^T) \\
\sigma^2 \left( \text{tr}(I) - \text{tr}((X^TX)^{-1}(X^TX))  \right) \\
\sigma^2 (N-K) \\
$$

여기에서 N은 표본 데이터의 수, K는 X 행렬의 열의 수 즉 모수의 갯수이다. 상수항을 포함한 선형 모형이라면 모수의 갯수는 입력데이터 차원의 수 D에 1을 더한 값이 된다. 

$$K = D + 1$$

잡음에 대한 비편향 표본분산은 다음과 같다.

$$
s^2 = \dfrac{e^Te}{N-K} = \dfrac{RSS}{N-K}
$$

$$\hat w$$의 (공)분산의 추정값은 다음과 같다.

$$
\text{Cov}[ \hat{w}] \approx s^2(X^TX)^{-1}
$$

이 공분산 행렬에서 관심을 가져야하는 값은 $$w_i$$의 분산을 뜻하는 대각성분이다.

$$
\text{Var}[\hat{w}_i]  = \left( \text{Cov}[ \hat{w} ] \right)_{ii} \;\; (i=0, \ldots, K-1)
$$

이 값에서 구한 표준 편차를 **회귀 계수의 표준 오차(Standard Error of Regression Coefficient)**라고 한다.

$$\sqrt{\text{Var}[\hat{w}_i]} \approx {se_i} = \sqrt{s^2 \big((X^TX)^{-1}\big)_{ii}} \;\; (i=0, \ldots, K-1)
$$

실제 가중치 계수 $$w_i$$와 우리가 추정한 가중치 계수 $$\hat{w}_i$$의 차이를 표준오차로 나눈 값, 즉 정규화된 모수 오차는 자유도 N - K인 표준 스튜던트 t분포를 따른다. 

$$
\dfrac{\hat{w}_i - w_i}{se_i} \sim t_{N-K} \;\; (i=0, \ldots, K-1)
$$

### 단일 계수 t-검정(Single Coefficient t-test)

정규화된 모수 오차를 검정 통계량으로 사용하면 $$w_i$$가 0인지 아닌지에 대한 검정을 실시할 수 있다.

$$
H_0 : w_i = 0 (i = 0, ... , K - 1)
$$

이 검정에 대한 유의 확률이 0에 가깝게 나온다면 위의 귀무가설은 기각이므로 $$w_i$$ 값이 0일 가능성은 적다.

StatsModels `summary` 메서드가 출력하는 회귀분석 보고서에서 `std err`로 표시된 열이 모형계수의 표준오차, `t`로 표시된 열이 단일 계수 t-검정의 검정 통계량, 그리고 `P>|t|`로 표시된 열이 유의확률을 뜻한다.

`RegressionResults` 클래스 객체는 t test 를 위한 `t_test` 메서드를 제공한다. 이 메서드를 사용하면 계수 값이 0이 아닌 경우도 테스트할 수 있다.

단일 계수 t 검정은 두 독립변수의 계수값을 비교할 때도 쓸 수 있다. 그리고 범주형 독립변수의 범주값이 가지는 유의성을 판단하는데 유용하다.

~~~python
print(result_nottem.t_test("C(month)[01] = C(month)[02]"))
~~~

### 회귀분석 F-검정

전체 회귀 계수가 모두 의미가 있는지 확인하는 경우에 사용할 수 있다.

$$
H_0 : w_0 = w_1 = ... = w_{k-1} = 0
$$

이는 전체 독립 변수 중 어느 것도 의미를 가진 것이 없다는 뜻으로 유의 확률이 작으면 작을수록 더 강력하게 기각된 것이므로 더 의미가 있는 모형이라고 할 수 있다. 

여러 모형의 유의 확률을 비교하여 어느 모형이 더 성능이 좋은가를 비교할 때도 이 유의 확률을 사용한다.

보고서에서 `F-statistic`라고 표시된 `400.3`이라는 값이 회귀분석 F-검정의 검정통계량이고 `Prob (F-statistic)`로 표시된 `2.21e-36`라는 값이 유의확률이다.

