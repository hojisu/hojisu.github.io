<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 로지스틱(Logistic) 회귀분석

### Summary

- 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0에서 1 사이의 값으로 예측하고 그 확률에 따라 가능성이 더 높은 범주에 속하는 것으로 분류해주는 지도 학습 알고리즘이다.
- 각 속성(feature)들의 계수 log-odds를 구한 후 Sigmoid 함수를 적용하여 실제로 데이터가 해당 클래스에 속할 확률을 0과 1사이의 값으로 나타낸다.
- 손실함수(Loss Function)는 머신러닝 모델이 얼마나 잘 예측하는지 확인하는 방법이다. 로지스틱 회귀의 손실함수는 Log Loss이다.
- 로지스틱 함수를 구성하는 계수와 절편에 대해 Log Loss(로그 손실)을 최소화하는 값을 찾는 것
- 종속변수가 범주형 데이터를 대상으로 하여 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나눠지기 때문에 일종의 분류 기법으로 볼 수 있다.
- 선형 회귀식의 우변의 범위를 0~1로 제한하고 우변의 결과 값이 범주 1에 속하는 확률을 출력하는 식이다. 
_______________


종속변수가 이항 분포(binomial distribution)를 따르고 그 모수 $$\mu$$ 가 독립변수 $$x$$ 에 의존한다고 가정한다.

$$
p(y \mid x) = \text{Bin} (y; \mu(x), N)
$$

y의 값은 특정한 구간내의 값(0~N)만 가질 수 있다. 이항 분포의 특별한 경우(N=1)로 y가 베르누이 확률분포인 경우도 있다. 

$$
p(y \mid x) = \text{Bern} (y; \mu(x) )
$$

종속변수가 y가 0 또는 1인 분류 예측 문제를 풀 때는 x 값에 따라 $$\mu(x)$$ 를 예측한 후 다음 기준에 따라 y를 예측한다.

$$
\hat{y} = \begin{cases} 1 & \text{ if } \mu(x) \geq 0.5 \\ 0 & \text{ if } \mu(x) < 0.5 \end{cases}
$$

또는 $$\hat y$$ 로 $$y=1$$ 이 될 확률값 $$\mu(x)$$ 를 직접 출력할 수도 있다.

$$
\hat y = \mu(x)
$$

### 시그모이드 함수(sigmoid function)

로지스틱 회귀모형에서는 베르누이 확률분포 모수 $$\mu$$ 가 x의 함수라고 가정한다. $$\mu(x)$$ 는 x에 대한 선형함수를 0부터 1사이의 값만 나올수 있도록 시그모이드 함수(sigmoid function)라는 함수를 사용하여 변형한 것을 사용한다.

$$
\mu = f(w^Tx)
$$

모수 $$\mu$$ 는 0부터 1까지의 실수값만 가질 수 있기 때문에 시그모이드 함수를 사용한다. 
- 시그모이드 함수는 종속변수의 모든 실수 값에 대해 유한한 구간 (a, b) 사이의 한정된(bounded) 값과 항상 양의 기울기를 가지는(단조증가) 함수의 집합을 말한다. 

시그모이드 함수 종류
- 로지스틱(Logistic)함수

$$
\text{logitstic}(z) = \sigma(z) = \dfrac{1}{1+\exp{(-z)}}
$$

- 하이퍼볼릭탄젠트(Hyperbolic tangent) 함수 (로지스틱함수 기울기의 4배)
  
$$
\tanh(z) = \frac{\sinh z}{\cosh z} = \frac{(e^z - e^{-z})/2}{(e^z + e^{-z})/2} = 2 \sigma(2z) - 1
$$

- 오차(Error) 함수

$$
\text{erf}(z) = \frac{2}{\sqrt\pi}\int_0^z e^{-t^2}\,dt
$$

### 로지스틱 함수

무한대의 실수값을 0부터 1사이의 실수값으로 1대1 대응시키는 시그모이드함수이다. 

베르누이 시도에서 1이 나올 확률 $$\mu$$ 와 0이 나올 확률 $$1 -\mu$$ 의 비(ratio)의 승산비(odds ratio)는 다음과 같다

$$
\text{odds ratio} = \dfrac{\mu}{1-\mu}
$$

0부터 1사이의 값만 가지는 $$\mu$$ 를 승산비로 변환하면 0부터 $$\infty$$ 의 값을 가질수 있다.

승산비를 로그 변환한 것이 로지트 함수(Logit function)이다. 

$$
z = \text{logit}(\text{odds ratio}) = \log \left(\dfrac{\mu}{1-\mu}\right)
$$

로지트함수의 값은 로그 변환에 의해 $$-\infty$$ 부터 $$\infty$$ 까지의 값을 가질 수 있다.

로지스틱함수(Logistic function)는 로지트함수의 역함수이다.  

$$-\infty$$ 부터 $$\infty$$ 까지의 값을 가지는 변수를 0부터 1사이의 값으로 변환한 결과이다.

$$
\text{logitstic}(z) = \mu(z) = \dfrac{1}{1+\exp{(-z)}}
$$

### 선형 판별함수

로지스틱함수 $$\sigma(z)$$ 를 사용하는 경우에는 $$z$$ 값과 $$\mu$$ 값은 다음과 같은 관계가 있다.
- $$z = 0$$ 일 때 $$\mu=0.5$$ 
- $$z > 0$$ 일 때 $$\mu > 0.5 \; \rightarrow \hat{y} = 1$$
- $$z < 0$$ 일 때 $$\mu < 0.5 \; \rightarrow \hat{y} = 0$$

$$z$$ 가 분류 모형의 판별함수(decision function) 역할을 한다. 

$$
z = w^Tx
$$

로지스틱 회귀모형의 영역 경계면은 선형이다. 

### 로지스틱 회귀분석 모형의 모수 추정

로지스틱 회귀분석 모형(비선형 회귀모형)의 모수 $$w$$를 최대가능도(Maximum Likelihood Estimation, MLE) 방법으로 추정하면 선형모형과 같이 간단하게 그레디언트 0이 되는 모수 $$w$$ 값에 대한 수식을 구할 수 없으며 수치적 최적화 방법(numerical optimization)을 통해 구해야 한다.

예시로 베르누아분포의 확률밀도함수는 다음과 같다

$$
p(y \mid x) = \text{Bern} (y;\mu(x;w) ) = \mu(x;w)^y ( 1 - \mu(x;w) )^{1-y}
$$

$$\mu$$ 는 $$w^Tx$$ 에 로지스틱함수를 적용한 값이다.

$$
\mu(x;w) = \dfrac{1}{1 + \exp{(-w^Tx)}}
$$

이 식을 대입하면 조건부확률은 다음과 같다.

$$
\begin{eqnarray}
p(y \mid x) 
&=& \left(  \dfrac{1}{1 + \exp{(-w^Tx)}} \right) ^y \left(  1 - \dfrac{1}{1 + \exp{(-w^Tx)}} \right) ^{1-y} \\
&=& \left(  \dfrac{1}{1 + \exp{(-w^Tx)}} \right) ^y \left( \dfrac{\exp{(-w^Tx)}}{1 + \exp{(-w^Tx)}} \right) ^{1-y} \\
\end{eqnarray}
$$

로그가능도 $$LL$$ 
$$
\begin{eqnarray}
{LL} 
&=& \log \prod_{i=1}^N \mu(x_i;w)^{y_i} (1-\mu(x_i;w))^{1-y_i} \\
&=& \sum_{i=1}^N \left( y_i \log\mu(x_i;w) +  (1-y_i)\log(1-\mu(x_i;w)) \right) \\
&=& \sum_{i=1}^N \left( y_i \log\left(\dfrac{1}{1 + \exp{(-w^Tx_i)}}\right) + (1-y_i)\log\left(\dfrac{\exp{(-w^Tx_i)}}{1 + \exp{(-w^Tx_i)}}\right) \right) \\
\end{eqnarray}
$$

로그가능도를 최대화하는 $$w$$ 값을 구하기 위해 모수로 미분한다. 

$$
\dfrac{\partial{LL}}{\partial w}  = \sum_{i=1}^N \dfrac{\partial{LL}}{\partial \mu(x_i;w)} \dfrac{\partial\mu(x_i;w)}{\partial w}
$$

$$LL$$ 을 $$\mu$$ 로 미분하면

$$
\dfrac{\partial{LL}}{\partial \mu(x_i;w)} =  \left( y_i \dfrac{1}{\mu(x_i;w)} - (1-y_i)\dfrac{1}{1-\mu(x_i;w)} \right)
$$

$$\mu$$ 를 $$w$$ 로 미분하면

$$
\dfrac{\partial \mu(x_i;w)}{\partial w} 
= \dfrac{\partial}{\partial w} \dfrac{1}{1 + \exp{(-w^Tx_i)}} \ 
= \dfrac{\exp{(-w^Tx_i)}}{(1 + \exp{(-w^Tx_i)})^2} x_i \ 
= \mu(x_i;w)(1-\mu(x_i;w)) x_i
$$

두 식을 곱하면 그레디언트 벡터의 수식을 구할 수 있다.

$$
\begin{eqnarray}
\dfrac{\partial {LL}}{\partial w} 
&=& \sum_{i=1}^N \left( y_i \dfrac{1}{\mu(x_i;w)} - (1-y_i)\dfrac{1}{1-\mu(x_i;w)} \right) \mu(x_i;w)(1-\mu(x_i;w)) x_i   \\
&=& \sum_{i=1}^N \big( y_i (1-\mu(x_i;w)) - (1-y_i)\mu(x_i;w)  \big)  x_i \\
&=& \sum_{i=1}^N \big( y_i  - \mu(x_i;w) \big) x_i \\
\end{eqnarray}
$$

그레디언트 벡터가 영벡터가 되는 모수의 값이 로그가능도를 최대화하는 값이다. 하지만 그레디언트 벡터 수식이 $$w$$ 에 대한 비선형 함수이므로 선형 모형과 같이 간단하게 그레디언트가 0이 되는 모수 $$w$$ 값에 대한 수식을 구할 수 없으면 수치적 최적화방법(numerical optimization )을 통해 반복적으로 최적 모수 $$w$$ 값을 구해야 한다.  

###  수치적 최적화(numerical optimization)

로그 가능도 함수 $$LL$$ 을 최대화하는 것은 다음 목적함수를 최소화하는 것과 같다.

$$
J = -LL
$$

최대경사도(Steepest Gradient Descent) 방법을 사용한다.

그레디언트 벡터는 $$g_k = \dfrac{d}{dw}(-LL)$$ 이고 이 방향으로 스텝사이즈 $$\eta_k$$ 만큼 이동한다.

$$
\begin{eqnarray}
w_{k+1} 
&=& w_{k} - \eta_k g_k \\
&=& w_{k} + \eta_k \sum_{i=1}^N \big( y_i  - \mu(x_i; w_k) \big) x_i\\
\end{eqnarray}
$$

### StatsModels 패키지의 로지스틱 회귀

~~~python
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit(disp=0)
print(logit_res.summary())
~~~

#### 판별함수

`Logit` 모형의 결과 객체에는 `fittedvalues`라는 속성으로 판별함수 $$z=w^Tx$$ 값이 들어가 있다.

~~~python
plt.scatter(X0, y, c=y, s=100, edgecolor="k", lw=2, label="데이터")
plt.plot(X0, logit_res.fittedvalues * 0.1, label="판별함수값")
plt.legend()
plt.show()
~~~

#### 성능 측정

McFadden pseudo R square 값으로 측정

$$
R^2_{\text{pseudo}} = 1 - \dfrac{G^2}{G^2_0}
$$

$$G^2$$ = deviance or log loss(로그 손실) 

$$
G^2 = 2\sum_{i=1}^N \left( y_i\log\dfrac{y_i}{\hat{y}_i} + (1-y_i)\log\dfrac{1-y_i}{1-\hat{y}_i} \right)
$$

$$\hat y$$ 은 $$y = 1$$ 일 확률을 뜻한다. $$\hat y = \mu(x_i)$$

deviance는 모형이 100% 정확한 경우에는 0이 되고 모형의 성능이 나빠질수록 값이 커진다.

이 값은 로그 가능도의 음수값과 같다. $$G^2 = -LL$$

$$G_{0}^2$$ 는 귀무모형(null model)으로 측정한 deviance이다. 

귀무모형이란 모든 $$x$$ 가 $$y$$ 를 예측하는데 전혀 영향을 미치지 않는 모형이다. 즉 무조건부 확률 $$p(y)$$ 에 따라 $$x$$ 에 상관없이 동일하게 $$y$$ 를 예측하는 모형을 말한다. 

scikit-learn 패키지의 metric 서브패키지에는 로그 손실을 계산하는 `log_loss` 함수가 있다. `normalize=False`로 놓으면 위와 같은 값을 구한다. `normalize` 인수의 디폴트 값은 `True`이고 이 때는 로그 손실의 평균값을 출력한다.

~~~python
from sklearn.metrics import log_loss

y_hat = logit_res.predict(X)
log_loss(y, y_hat, normalize=False)
~~~

- 귀무 모형의 모수값을 구하기

~~~python
mu_null = np.sum(y) / len(y)
~~~

- 귀무 모형으로 로그 손실 계산

~~~python
y_null = np.ones_like(y) * mu_null
log_loss(y, y_null, normalize=False)
~~~

- McFadden pseudo R square 계산

~~~python
1 - (log_loss(y, y_hat) / log_loss(y, y_null))
~~~

### Scikit-Learn 패키지의 로지스틱 회귀

- Scikit-Learn 패키지는 로지스틱 회귀 모형 `LogisticRegression` 를 제공한다.

~~~python
from sklearn.linear_model import LogisticRegression

model_sk = LogisticRegression().fit(X0, y)

xx = np.linspace(-3, 3, 100)
mu = 1.0/(1 + np.exp(-model_sk.coef_[0][0]*xx - model_sk.intercept_[0]))
plt.plot(xx, mu)
plt.scatter(X0, y, c=y, s=100, edgecolor="k", lw=2)
plt.scatter(X0, model_sk.predict(X0), label=r"$$\hat{y}$$", marker='s', c=y,
            s=100, edgecolor="k", lw=1, alpha=0.5)
plt.xlim(-3, 3)
plt.xlabel("x")
plt.ylabel(r"$$\mu$$")
plt.title(r"$$\hat{y}$$ = sign $$\mu(x)$$")
plt.legend()
plt.show()
~~~

Reference
- 김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 
