<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# Variance & Standard Deviation(분산과 표준편차)

### Summary

- 기댓값은 확률변수에서 어떤 값이 나올지를 예측한다.
- 분산은 그 예측의 정확도(신뢰성) 표현한다.
- 표본평균을 계산한 표본의 갯수가 커지면 표본평균의 값의 변동은 작아진다. 표본의 갯수 $$N$$ 가 크면 표본평균 $$\bar x$$ 은 원래 확률변수 $$X$$ 의 기댓값 $$E[X]$$ 의 근삿값 이라고 할 수있다.
- 표본분산의 값이 이론적인 분산의 값보다 더 작아진다. 표본분산을 계산한 때 사용하는 표본평균이 데이터가 몰린 쪽으로 편향되게 나옵니다. 이렇게 데이터가 몰려있는 위치에 있는 표본평균을 기준으로 각 데이터까지의 거리를 계산하면 원래의 기댓값으로부터의 거리보다 작게 나올 수 있다. 그래서 N-1로 나눠준다. 
- 두 개의 확률분포 $$X, Y$$ 가 있고 1차부터 무한대 차수까지 두 확률분포의 모든 모멘트 값이 서로 같다면 두 확률분포는 같은 확률분포이다.  ( = pdf 모양이 같다는 것)

_______________

### 확률분포의 분산

확률밀도함수 $$p(x)$$ 의 수식을 알고 있다면 다음처럼 이론적인 분산을 구할 수 있다.

$$
\sigma^2 = \text{Var}[X] = \text{E}[(X - \mu)^2]
$$

이산확률변수의 분산은 평균으로부터 표본까지 거리의 제곱을 확률질량함수 $$p(x)$$ 로 가중평균한 값이다.

$$
\sigma^2 = \text{Var}[X] = \text{E}[(X - \mu)^2] =  \sum_{x_i \in \Omega} (x_i - \mu)^2 p(x_i)
$$

**연속확률변수의 분산은 평균으로부터 표본까지 거리의 제곱을 확률밀도함수 $$p(x)$$ 로 가중하여 적분한 값이다.**

$$
\sigma^2 = \text{Var}[X] = \text{E}[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 p(x)dx
$$

### 분산의 성질

항상 0 또는 양수 

$$
\text{Var}[X] \geq 0
$$

확률변수가 아닌 상수 값 c에 대해 

$$
Var[c] = 0 \\
Var[cX] = c^2Var[X]
$$

기댓값의 성질을 이용하여 다음 성질을 증명할 수 있다.

$$
\text{Var}[X] = \text{E}[X^2] - (\text{E}[X])^2  = \text{E}[X^2] - \mu^2
$$

$$
\text{E}[X^2] = \mu^2 + \text{Var}[X]
$$

$$
\begin{eqnarray}
\text{Var}[X] 
&=& \text{E}[(X - \mu)^2] \\
&=& \text{E}[X^2 - 2\mu X + \mu^2] \\
&=& \text{E}[X^2] - 2\mu\text{E}[X] + \mu^2 \\
&=& \text{E}[X^2] - 2\mu^2 + \mu^2 \\
&=& \text{E}[X^2] - \mu^2\\
\end{eqnarray}
$$

### 두 확률변수의 합의 분산

두 확률변수 $$X, Y$$ 의 합의 분산은 각 확률변수의 분산의 합과 다음과 같은 관계가 있다.

$$
\text{Var}\left[ X + Y \right] =
\text{Var}\left[ X \right] + \text{Var}\left[ Y \right]+ 2\text{E}\left[ (X-\mu_X)(Y-\mu_Y) \right]
$$

마지막 항은 양수 or 음수 될 수 있다. 확률변수 $$𝑋+𝑌$$의 기댓값은 기댓값의 성질로부터 각 확률변수의 기댓값의 합과 같다.

$$
\text{E}[X + Y] = \mu_X + \mu_Y
$$

분산의 정의와 기댓값의 성질로부터 다음이 성립한다.

$$
\begin{eqnarray}
\text{Var}\left[ X + Y \right] 
&=& \text{E}\left[ (X + Y - (\mu_X + \mu_Y))^2 \right] \\
&=& \text{E}\left[ ((X -\mu_X) + (Y - \mu_Y))^2 \right] \\
&=& \text{E}\left[ (X -\mu_X)^2 + (Y - \mu_Y)^2 + 2(X-\mu_X)(Y-\mu_Y) \right] \\
&=& \text{E}\left[ (X -\mu_X)^2 \right] + \text{E}\left[ (Y - \mu_Y)^2 \right] + 2\text{E}\left[ (X-\mu_X)(Y-\mu_Y) \right] 
\end{eqnarray}
$$

### 확률변수의 독립

두 확률변수가 서로 독립(independent) 이라는 것은 결합사건의 확률이 각 사건의 확률의 곱과 같다는 뜻이다. 확률변수가 서로에게 영향을 미치지 않는다 라는 의미이다. 

두 확률변수에서 하나의 확률변수의 값이 특정한 값이면 다른 확률변수의 확률분포가 영향을 받아 변하게 되면 종속(dependent) 라고 한다. 두 확률변수가 서로에게 영향을 미치는 경우이다.

두 확률변수 $$X, Y$$ 가 서로 독립이면 다음 식이 성립한다.

$$
\text{E}\left[ (X-\mu_X)(Y-\mu_Y) \right] = 0
$$

이 등식을 이용하면 서로 독립인 두 확률변수의 합의 분산은 각 확률변수의 분산의 합과 같다는 것을 보일 수 있다.

$$
\text{Var}\left[ X + Y \right] =  \text{Var}\left[ X \right] + \text{Var}\left[ Y \right]
$$

### 표본평균의 분산

$$
\text{Var}[\bar{X}] = \dfrac{1}{N} \text{Var}[{X}]
$$

**표본평균을 계산한 표본의 갯수가 커지면 표본평균의 값의 변동은 작아진다**. 표본의 갯수 $$N$$ 가 크면 표본평균 $$\bar x$$ 은 원래 확률변수 $$X$$ 의 기댓값 $$E[X]$$ 의 근삿값 이라고 할 수있다.

### 표본분산의 기댓값

$$
\text{E}[S^2] = \dfrac{N-1}{N}\sigma^2
$$

표본분산의 값이 이론적인 분산의 값보다 더 작아진다. 표본분산을 계산한 때 사용하는 표본평균이 데이터가 몰린 쪽으로 편향되게 나옵니다. 이렇게 데이터가 몰려있는 위치에 있는 표본평균을 기준으로 각 데이터까지의 거리를 계산하면 원래의 기댓값으로부터의 거리보다 작게 나올 수 있다. 그래서 N-1로 나눠준다. 

$$
\begin{align}
\begin{aligned}
\sigma^2 
&= \dfrac{N}{N-1} \text{E}[S^2] \\
&= \dfrac{N}{N-1} \text{E} \left[ \dfrac{1}{N} \sum (X_i-\bar{X})^2 \right] \\
&= \text{E} \left[ \dfrac{1}{N-1} \sum (X_i-\bar{X})^2 \right]
\end{aligned}
\end{align}
$$

$$
\begin{align}
S^2_{\text{unbiased}} \equiv \dfrac{1}{N-1} \sum (X_i-\bar{X})^2
\end{align}
$$

###  모멘트

기댓값이나 분산은 확률분포의 **모멘트(moment)** 의 하나이다.
$$
\mu_n = \operatorname{E}[(X-\mu)^n] = \int (x - \mu)^n p(x)dx
$$

두 개의 확률분포 $$X, Y$$ 가 있고 1차부터 무한대 차수까지 두 확률분포의 모든 모멘트 값이 서로 같다면 두 확률분포는 같은 확률분포이다.  ( = pdf 모양이 같다는 것)
$$
X \stackrel d= Y
$$

$$\stackrel d=$$ 는 두 확률변수가 같은 분포에서 나왔다는 것을 표시하는 기호이다.

### 비대칭도와 첨도

비대칭도(skew)는 3차 모멘트 값에서 계산하고 확률밀도함수의 비대칭 정도를 가리킨다. 비대칭도가 0이면 확률분포가 대칭이다.
$$
\operatorname{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3 \right] = \frac{\mu_3}{\sigma^3}
$$

첨도(kurtosis)는 4차 모멘트 값에서 계산하며 확률이 정규분포와 대비하여 중심에 모여있는지 바깥으로 퍼져있는지를 나타낸다.
$$
\operatorname{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4 \right] = \frac{\mu_4}{\sigma^4}
$$

