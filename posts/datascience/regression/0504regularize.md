<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 정규화 

### Summary
- 정규화는 오버피팅을 막는 방법이다. 
- 정규화 모형에는 Ridge 모형과 Lasso 모형이 있다. Ridge모형은 가중치들의 제곱합을 최소화 하는 것이고 Lasso는 가중치들의 절대값의 합을 최소화 하는 것이다.  Ridge모형과 Lasso 모형의 차이점은 Ridge는 계수가 한꺼번에 축소시키는 반면에 Lasso는 일부의 가중치 계수가 먼저 0으로 수렴하는 특징이 있다.
- 정규화 모형의 장점은 데이터가 달라져도 계수가 크게 달라지지 않는다. 
___________________

### 정규화(regularized) 선형회귀

선형회귀 계수(weight)에 대한 제약 조건을 추가함으로써 모형이 과도하게 최적화되는 현상(과최적화)를 막는 방법이다. (Regularized Method, Penalized Method, Contrained Least Squares)

모형이 과도하게 최적화되면 모형 계수의 크기도 과도하게 증가하는 경향이 나타난다. 정규화 방법에서 추가하는 제약 조건은 일반적으로 계수의 크기를 제한하는 방법이다.

#### Ridge 회귀모형

가중치들의 제곱합(squared sum of weights)을 최소화하는 것을 추가적인 제약조건으로 한다. 

$$
w = \text{arg}\min_w \left( \sum_{i=1}^N e_i^2 + \lambda \sum_{j=1}^M w_j^2 \right)
$$

$\lambda$ 는 기존의 잔차 제곱합과 추가적 제약 조건의 비중을 조절하기 위한 하이퍼모수(hyperparameter)이다. $\lambda$ 가 크면 정규화 정도가 커지고 가중치의 값들이 작아진다. $\lambda$ 가 작아지면 정규화 정도가 작아지며 $\lambda$ 가 0이 되면 일반적인 선형회귀모형이 된다. 

#### Lasso(Least Absolute Shrinkage and Selection Operator) 회귀모형

회귀모형은 가중치의 절대값의 합을 최소화하는 것을 추가적인 제약 조건으로 한다

$$
w = \text{arg}\min_w \left( \sum_{i=1}^N e_i^2 + \lambda \sum_{j=1}^M | w_j | \right)
$$

#### Elastic Net 회귀모형

가중치의 절대값의 합과 제곱합을 동시에 제약 조건으로 가지는 모형이다. $\lambda_1$, $\lambda_2$ 두개의 하이퍼 모수를 가진다. 

$$
w = \text{arg}\min_w \left( \sum_{i=1}^N e_i^2 + \lambda_1 \sum_{j=1}^M | w_j | + \lambda_2 \sum_{j=1}^M w_j^2 \right)
$$

### statsmodels의 정규화 회귀모형

statsmodels 패키지는 OLS 선형 회귀모형 클래스의 `fit_regularized` 메서드를 사용하여 Elastic Net 모형 계수를 구할 수 있다.

하이퍼 모수는 다음과 같이 모수 alphaalpha 와 L1_wtL1_wt 로 정의된다.

~~~python
# L1_wt=0 : 순수 Ridge 모형
result2 = model.fit_regularized(alpha=0.01, L1_wt=0)
print(result2.params)
plot_statsmodels(result2)
~~~

~~~python
# L1_wt=1 : 순수 Lasso 모형
result3 = model.fit_regularized(alpha=0.01, L1_wt=1)
print(result3.params)
plot_statsmodels(result3)

~~~

~~~python
# L1_wt=0과 1사이 : 순수 Elastic Net 모형
result4 = model.fit_regularized(alpha=0.01, L1_wt=0.5)
print(result4.params)
plot_statsmodels(result4)
~~~

### Scikit-Learn의 정규화 회귀모형

Scikit-Learn 패키지에서는 정규화 회귀모형을 위한 `Ridge`, `Lasso`, `ElasticNet` 이라는 별도의 클래스를 제공한다. 각 모형에 대한 최적화 목적 함수는 다음과 같다.

~~~python
# Ridge
model = make_pipeline(poly, Ridge(alpha=0.01)).fit(X, y)
print(model.steps[1][1].coef_)
plot_sklearn(model)
~~~

~~~python
# Lasso
model = make_pipeline(poly, Lasso(alpha=0.01)).fit(X, y)
print(model.steps[1][1].coef_)
plot_sklearn(model)
~~~

~~~python
# ElasticNet
model = make_pipeline(poly, ElasticNet(alpha=0.01, l1_ratio=0.5)).fit(X, y)
print(model.steps[1][1].coef_)
plot_sklearn(model)
~~~

### 정규화 모형의 장점

회귀분석에 사용된 데이터가 달라져도 계수가 크게 달라지지 않는다

### 정규화의 의미

정규화 제한조건은 정규화가 없는 최적화 문제에 부등식 제한조건을 추가한 것과 마찬가지이다. 라그랑지 방법을 사용하면 부등식 제한조건이 있는 최적화 문제는 부등식 제한조건이 없는 최적화 문제가 된다.

### Ridge 모형과 Lasso 모형의 차이

- Ridge 모형은 가중치 계수를 한꺼번에 축소시키는데 반해 Lasso 모형은 일부 가중치 계수가 먼저 0으로 수렴하는 특성이 있다.

### 최적 정규화

정규화에 사용되는 하이퍼 모수(hyper parameter)등을 바꾸면 모형의 검증 성능이 달라진다. 

최적 정규화(optimal regularization)는 최적의 성능을 가져올 수 있는 정규화 하이퍼 모수를 선택하는 과정이다.

scikit-learn의 Lasso 클래스를 사용하여 정규화를 예로 들면
- 학습용 데이터를 사용한 성능은 정규화 가중치 $\alpha$ 가 작으면 작을 수록 좋아진다.(과최적화)
- 검증용 데이터를 사용한 성능은 정규화 가중치 $\alpha$ 가 특정한 범위에 있을 때 가장 좋아진다. 

### 검증성능 곡선(validation curve)

특정한 하나의 하이퍼 모수를 변화시켰을 때 학습 성능과 검증 성능을 변화를 나타낸 곡선이다.

scikit-learn 패키지의 model_selection 서브패키지에서 제공하는 `validation_curve` 함수를 사용

~~~python
from sklearn.model_selection import validation_curve

train_scores, test_scores = validation_curve(
    Lasso(), X, y, "alpha", alphas, cv=5,
    scoring="neg_mean_squared_error")

plt.plot(alphas, test_scores.mean(axis=1), "-", label="검증성능 평균")
plt.plot(alphas, train_scores.mean(axis=1), "--", label="학습성능 평균")
plt.ylabel('성능')
plt.xlabel('정규화 가중치')
plt.legend()
plt.title("최적 정규화")
plt.show()
~~~

### 다항회귀의 차수 결정

다항회귀에서 차수가 감소하면 모형의 제약조건이 더 강화되므로 정규화 가중치가 커지는 것과 같다. 다항회귀에서 최적의 차수를 결정하는 문제는 최적 정규화에 해당한다.

~~~python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

n_samples = 100
np.random.seed(0)
X = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.5
X = X[:, np.newaxis]


model = Pipeline([("poly", PolynomialFeatures()),
                  ("lreg", LinearRegression())])

degrees = np.arange(1, 15)
train_scores, test_scores = validation_curve(
    model, X, y, "poly__degree", degrees, cv=100,
    scoring="neg_mean_squared_error")

plt.plot(degrees, test_scores.mean(axis=1), "o-", label="검증성능 평균")
plt.plot(degrees, train_scores.mean(axis=1), "o--", label="학습성능 평균")
plt.ylabel('성능')
plt.xlabel('다항 차수')
plt.legend()
plt.title("최적 정규화")
plt.show()
~~~

