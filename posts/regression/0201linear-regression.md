<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 선형 회귀분석의 기초

### Summary

- 선형회귀분석은 독립변수 x에 대응하는 종속변수 y와 가장 비슷한 값을 출력하는 선형함수를 찾는 과정이다. 
- OLS(Ordinary Least Squares) 는 가장 기본적인 결정론적 선형 회귀 방법으로 잔차제곱합(RSS: Residual Sum of Squares)를 최소화하는 가중치 벡터를 행렬 미분으로 구하는 방법이다.

_____________

### 회귀분석 regression analysis

회귀분석(regression analysis)은 D차원 벡터 독립 변수 $$x$$와 이에 대응하는 스칼라 종속변수 $$y$$간의 관계를 정량적으로 찾아내는 작업이다.

#### 결정론적 모형(deterministic Model)

독립 변수 $$x$$에 대해 대응하는 종속 변수 $$y$$와 가장 비슷한 값 $$\hat{y}$$ 을 출력하는 함수 $$f(x)$$ 를 찾는 과정이다. 

$$
\hat{y} = f \left( x \right) \approx y
$$

선형 회귀분석(linear regression analysis)은 독립 변수 $$x$$와 이에 대응하는 종속 변수 y간의 관계가 다음과 같은 선형 함수 $$f(x)$$ 를 찾는 과정이다.  
$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_D x_D = w_0 + w^Tx
$$

$$w_0, \cdots, w_D$$ 를 함수 $$f(x) $$ 의 계수(coefficient) 이자 이 선형 회귀모형의 모수(parameter) 라고 한다.


### 상수항 결합

상수항 결합(bias augmentation)은 상수항이 0이 아닌 회귀분석모형인 경우에는 수식을 간단하게 만들기 위해 다음과 같이 상수항을 독립변수에 추가한다.

$$
x_i =
\begin{bmatrix}
x_{i1} \\ x_{i2} \\ \vdots \\ x_{iD}
\end{bmatrix}
\rightarrow 
x_{i,a} =
\begin{bmatrix}
1 \\ x_{i1} \\ x_{i2} \\ \vdots \\ x_{iD}
\end{bmatrix}
$$

상수항 결합을 하게 되면 모든 원소가 1인 벡터가 입력 데이터 행렬에 추가된다.

$$
X =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1D} \\
x_{21} & x_{22} & \cdots & x_{2D} \\
\vdots & \vdots & \vdots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{ND} \\
\end{bmatrix}
\rightarrow 
X_a =
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1D} \\
1 & x_{21} & x_{22} & \cdots & x_{2D} \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
1 & x_{N1} & x_{N2} & \cdots & x_{ND} \\
\end{bmatrix}
$$

이렇게 되면 전체 수식이 다음과 같이 상수항이 추가된 가중치 벡터 𝑤와 상수항이 추가된 입력 데이터 벡터 𝑥의 내적으로 간단히 표시된다.

$$
f(x) = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_D x_D
= 
\begin{bmatrix}
1 & x_1 & x_2 & \cdots & x_D
\end{bmatrix}
\begin{bmatrix}
w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_D
\end{bmatrix}
= x_a^T w_a = w_a^T x_a
$$

~~~python
#상수항 결합
X = np.hstack([np.ones((X0.shape[0], 1)), X0])
X[:5]
~~~

~~~python
# statsmodels에는상수항 결합을 위한 `add_constant` 함수가 제공된다.
import statsmodels.api as sm

X = sm.add_constant(X0)
X[:5]
~~~

### OLS(Ordinary Least Squares)

OLS(Ordinary Least Squares)는 가장 기본적인 결정론적 선형 회귀 방법으로 잔차제곱합(RSS: Residual Sum of Squares)를 최소화하는 가중치 벡터를 행렬 미분으로 구하는 방법이다.

예측 모형은 상수항이 결합된 선형 모형이다.

$$
\hat{y} = Xw
$$

잔차 벡터(residual vector) $$e$$

$$
e = {y} - \hat{y} = y - Xw
$$

잔차 제곱합(RSS:residual sum of squares)

$$
\begin{eqnarray}
\text{RSS}
&=&  e^Te \\
&=& (y - Xw)^T(y - Xw) \\
&=& y^Ty - 2y^T X w + w^TX^TXw  
\end{eqnarray}
$$

잔차의 크기 즉, 잔차 제곱합을 가장 작게 하는 가중치 벡터를 구하기 위해 잔차 제곱합의 그레디언트(gradient) 벡터를 구하면 다음과 같다.

$$
\dfrac{d \text{RSS}}{d w} = -2 X^T y + 2 X^TX w
$$

잔차가 최소가 되는 최적화 조건은 그레디언트 벡터가 0벡터이어야 하므로 다음 식이 성립한다.

$$
\dfrac{d \text{RSS}}{d w}  = 0 \\
X^TX w^{\ast} = X^T y
$$

만약 $$X^TX$$ 행렬의 역행렬이 존재한다면 다음처럼 최적 가중치 벡터 $$w^∗$$를 구할 수 있다.

$$
w^{\ast} = (X^TX)^{-1} X^T y
$$

$$X^TX$$ 행렬의 역행렬이 존재하고 위에서 구한 값이 최저값이 되려면  잔차제곱합의 헤시안 행렬인 $$X^TX$$가 양의 정부호(positive definite) 행렬이어야 한다. 

$$
\frac{d^2 \text{RSS}}{dw^2} = 2X^TX > 0
$$

만약  X가 풀랭크가 아니면 즉, X의 각 행렬이 서로 독립이 아니면 $$X^TX$$가 양의 정부호가 아니고 역행렬이 존재하지 않으므로 위와 같은 해를 구할 수 없다.

### 직교 방정식

직교 방정식(normal equation)은 그레디언트가 0벡터가 되는 관계를 나타낸 것이다. 

$$
X^T y - X^TX w = 0
$$

직교 방정식을 인수 분해하면

$$
X^T (y - X w ) = 0 \\
X^T e = 0
$$

$$c_d$$가 모든 데이터의 $$d$$번째 차원의 원소로 이루어진 데이터 벡터(특징 행렬의 열벡터)라고 할 때 모든 차원 $$d(d=0,…,D)$$에 대해 $$c_d$$는 잔차 벡터 $$e$$와 수직을 이룬다.

$$
c_d^T e = 0 \;\;\; (d=0, \ldots, D) \\
c_d \perp e \;\;\; (d=0, \ldots, D)
$$

직교 방정식으로부터 다음과 같은 성질을 알 수 있다.

1. 모형에 상수항이 있는 경우에 잔차 벡터의 원소의 합은 0이다. 즉, 잔차의 평균은 0이다.

$$
  \sum_{i=0}^N e_i = 0
$$

상수항 결합이 되어 있으면 $$X$$의 첫번째 열이 1-벡터라는 것을 이용하여 증명할 수 있다.

$$
  c_0^T e = \mathbf{1}^T e = \sum_{i=0}^N e_i = 0
$$

2. $$x$$ 데이터의 평균값 $$\bar x$$에 대한 예측값은 $$y$$ 데이터의 평균값 $$\bar y$$이다.

$$
\bar{y} = w^T \bar{x}
$$

증명

$$
\begin{eqnarray}
  \bar{y} 
  &=& \dfrac{1}{N}\mathbf{1}^T y \\
  &=& \dfrac{1}{N}\mathbf{1}^T (Xw + e) \\
  &=& \dfrac{1}{N}\mathbf{1}^TXw + \dfrac{1}{N}\mathbf{1}^Te \\
  &=& \dfrac{1}{N}\mathbf{1}^TXw \\
  &=& \dfrac{1}{N}\mathbf{1}^T \begin{bmatrix}c_1 & \cdots & c_M \end{bmatrix} w \\
  &=& \begin{bmatrix}\dfrac{1}{N}\mathbf{1}^Tc_1 & \cdots & \dfrac{1}{N}\mathbf{1}^Tc_D \end{bmatrix} w \\
  &=& \begin{bmatrix}\bar{c}_1 & \cdots & \bar{c}_D \end{bmatrix} w \\
  &=& \bar{x}^T w \\
  \end{eqnarray}
$$

   

### NumPy를 이용한 선형 회귀분석

~~~python
from sklearn.datasets import make_regression

bias = 100
X0, y, w = make_regression(
    n_samples=200, n_features=1, bias=bias, noise=10, coef=True, random_state=1
)
X = sm.add_constant(X0)
y = y.reshape(len(y), 1)
w
# array(86.44794301)
~~~

$$
y = 100 + 86.44794301 x + \epsilon
$$

~~~python
# OLS 해를 직접 이용하는 방법
w = np.linalg.inv(X.T @ X) @ X.T @ y
w
# array([[99.79150869],
#       [86.96171201]])
~~~

$$
\hat{y} = 99.79150869 + 86.96171201 x
$$

이 결과에서 알 수 있는 것은 선형 회귀를 통해 구한 결과는 실제(자연 법칙)와 **비슷하지만 정확하지는 않다**는 점이다.