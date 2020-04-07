<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 레버리지와 아웃라이어

### Summary
- 레버리지는 실제 종속변수값이 예측값에 미치는 영향을 나타낸 값이다. 즉 예측점을 자기 자신의 위치로 끌어당기는 정도를 나타낸 것이다.
- outlier(아웃라이어)는 잔차가 큰 데이터이다. 즉 모형과 동떨어진 데이터를 나타낸 것이다.
- Cook's Distance는 레버리지와 잔차의 크기가 모두 큰 데이터들을 살펴보기 위한 방법이다. 레버리지가 커지거나 잔차의 크기가 커지면 Cook's Distance 값이 커진다.
_________________________________

### 레버리지

**laverage(레버리지)**는 실제 종속변수값 $$y$$ 가 예측치(predicted target) $$\hat y$$ 에 미치는 영향을 나타는 값이다. 

레버리지는 수학적으로 **영향도 행렬의 대각성분 $$h_{ii}$$**으로 정의된다. 즉, 레버리지는 실제의 결과값 $$y_i$$이 예측값 $$\hat y_i$$에 미치는 영향, 즉 예측점을 자기 자신의 위치로 끌어 당기는 정도를 나타낸 것이다.

- 1보다 같거나 작은 양수 혹은 0이다.$$0 \leq h_{ii} \leq 1$$ 
- 레버리지의 합은 모형에 사용된 모수의 갯수 K와 같다. 모수에는 상수항도 포함되므로 상수항이 있는 1차원 모형에서는 K = 2가 된다. 

$$
\text{tr}(H) = \sum_i^N h_{ii} = K
$$

#### statsmodels를 이용한 레버리지 계산

레버리지 값은 `RegressionResults` 클래스의 `get_influence` 메서드로 다음과 같이 구할 수 있다. 우선 다음과 같은 가상의 1차 데이터로 회귀분석 예제를 풀어보자.

~~~python
influence = result.get_influence()
hat = influence.hat_matrix_diag

plt.figure(figsize=(10, 2))
plt.stem(hat)
plt.axhline(0.02, c="g", ls="--")
plt.title("각 데이터의 레버리지 값")
plt.show()

~~~

위 결과를 이용하여 레버리지가 큰 데이터만 아래의 코드로 표시할 수 있다. 

~~~python
ax = plt.subplot()
plt.scatter(X0, y)
sm.graphics.abline_plot(model_results=result, ax=ax)

idx = hat > 0.05
plt.scatter(X0[idx], y[idx], s=300, c="r", alpha=0.5)
plt.title("회귀분석 결과와 레버리지 포인트")
plt.show()
~~~

#### 레버리지의 영향

레버리지가 큰 데이터는 포함되거나 포함되지 않는가에 따라 모형에 주는 영향이 큰 것을 알 수 있다. 반대로 레버리지가 작은 데이터는 포함되거나 포함되지 않거나 모형이 별로 달라지지 않는 것을 알 수 있다. 혹은 레버리지가 크더라도 오차가 작은 데이터는 포함되거나 포함되지 않거나 모형이 별로 달라지지 않는다.

### 아웃라이어

**outlier(아웃라이어)**는 모형에서 설명하고 있는 데이터와 동떨어진 값을 가지는 데이터, 즉 잔차가 큰 데이터이다. 잔차의 크기는 독립 변수의 영향을 받으므로 아웃라이어를 찾으러면 이 영향을 제거한 표준화된 잔차를 계산해야 한다.

#### 표준화 잔차

잔차를 레버리지와 잔차의 표준 편차로 나누어 동일한 표준 편차를 가지도록 스케일링한 것을 **표준화 잔차**(standardized residual 또는 normalized residual 또는 studentized residual)라고 한다.

$$
r_i = \dfrac{e_i}{s\sqrt{1-h_{ii}}}
$$

#### statsmodels를 이용한 표준화 잔차 계산

잔차는 `RegressionResult` 객체의 `resid`속성에 있다.

~~~python
plt.figure(figsize=(10, 2))
plt.stem(result.resid)
plt.title("각 데이터의 잔차")
plt.show()
~~~

표준화 잔차는 `resid_pearson` 속성에 있다. 보통 표준화 잔차가 2~4보다 크면 아웃라이어로 본다.

~~~python
plt.figure(figsize=(10, 2))
plt.stem(result.resid_pearson)
plt.axhline(3, c="g", ls="--")
plt.axhline(-3, c="g", ls="--")
plt.title("각 데이터의 표준화 잔차")
plt.show()
~~~

### Cook's Distance

회귀 분석에는 잔차의 크기가 큰 데이터가 아웃라이어가 되는데 이 중에서도 주로 관심을 가지는 것은 레버리지와 잔차의 크기가 모두 큰 데이터들이다. 잔차와 레버리지를 동시에 보기위한 기준으로는 Cook's Distance가 있다. 다음과 같이 정의되는 값으로 레버리지가 커지거나 잔차의 크기가 커지면 Cook's Distance 값이 커진다.

Fox' Outlier Recommendation 은 Cook's Distance가 다음과 같은 기준값보다 클 때 아웃라이어로 판단하자는 것이다.

$$
D_i > \dfrac{4}{N − K - 1}
$$

~~~python
  from statsmodels.graphics import utils
  
  cooks_d2, pvals = influence.cooks_distance
  K = influence.k_vars
  fox_cr = 4 / (len(y) - K - 1)
  idx = np.where(cooks_d2 > fox_cr)[0]
  
  ax = plt.subplot()
  plt.scatter(X0, y)
  plt.scatter(X0[idx], y[idx], s=300, c="r", alpha=0.5)
  utils.annotate_axes(range(len(idx)), idx,
                      list(zip(X0[idx], y[idx])), [(-20, 15)] * len(idx), size="small", ax=ax)
  plt.title("Fox Recommendaion으로 선택한 아웃라이어")
  plt.show()
~~~

모든 데이터의 레버리지와 잔차를 동시에 보려면 `plot_leverage_resid2` 명령을 사용한다. 이 명령은 x축으로 표준화 잔차의 제곱을 표시하고 y축으로 레버리지값을 표시한다. 데이터 아이디가 표시된 데이터들이 레버리지가 큰 아웃라이어이다.

~~~python
sm.graphics.plot_leverage_resid2(result)
plt.show()
~~~

`influence_plot` 명령을 사용하면 Cook's distance를 버블 크기로 표시한다.

~~~python
sm.graphics.influence_plot(result, plot_alpha=0.3)
plt.show()
~~~

