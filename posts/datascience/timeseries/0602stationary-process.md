<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 정상 확률 과정과 비정상 확률 과정

### Summary
- 정상확률과정은 확률과정의 모든 모멘트가 시간 차이에만 의존한다는 것이다.
- 에르고딕 성질은 정상 확률 과정에서는 각각의 시간에 해당하는 확률 변수의 무조건부 분포가 모두 같다는 것이다. 시계열 데이터를 이루는 각 숫자가 하나의 분포에서 나온 표본 데이터 집합이라고 생각할 수 있다. 
___________________________________

### 정상 확률 과정

협의의 정상 확률 과정(strictly stationary process, strong stationary process)은 확률 과정의 모든 모멘트(moment)가 시간 차이(time lag)에만 의존하고 절대 시간에 의존하지 않는 것이다.

$$
\text{E}[Y_{t} Y_{t+k_1} Y_{t+k_2} \cdots Y_{t+k_i} \cdots ] = \text{E}[Y_s Y_{s+k_1} Y_{s+k_2} \cdots Y_{s+k_i} \cdots]
$$

아래의 두가지 조건만 성립하는 경우가 광의의 정상 확률 과정(wide-sense stationary process, weak stationary process)이다.
- 기댓값 $$\text{E}[Y_{t}] = \text{E}[Y_{s}] = \mu$$
- 자기공분산 $$\text{E}[Y_{t}Y_{t+k}] = \text{E}[Y_{s}Y_{s+k}] = f(k)$$

자기공분산함수(auto covariance function)은 자기공분산 시차에 대한 함수이다. 정상 확률과정에서는 자기공분산의 값이 시간 변수의 차이 즉 시차(lag) k에만 의존한다. 

$$
\gamma_{t,t+k} =  \gamma_{0,k} \triangleq  \gamma_k
$$

**자기상관계수함수(auto correlation function) = ACF**은 정상 확률 과정의 자기상관계수도 마찬가지로 시차 k에만 의존한다. 

$$
\rho_{t,t+k} = \rho_{0,k} \triangleq \rho_k = \dfrac{\gamma_k}{\gamma_0}
$$

정상 확률 과정 성질은 아래와 같다.

$$
\begin{eqnarray}
\gamma_0 &=& \text{Var}[Y_t] \\
\gamma_{k} &=& \gamma_{-k} \\
\left| \gamma_{k} \right| &\leq& \gamma_{0}  \\
\rho_{0} &=& 1 \\
\rho_{k} &=& \rho_{-k} \\
\left| \rho_{k} \right| &\leq& 1 \\
\end{eqnarray}
$$

### 에르고딕 성질(ergodicity)

에르고딕 성질은 정상 확률 과정에서는 각각의 시간에 해당하는 확률 변수의 무조건부 분포가 모두 같다. 

시계열 데이터를 이루는 각 숫자가 하나의 분포에서 나온 표본 데이터 집합이라고 생각할 수 있다. 

기댓값

$$
\lim_{N\rightarrow \infty} \dfrac{1}{N} \sum^N Y_t = \text{E}[Y_t]
$$

자기공분산

$$
\lim_{N\rightarrow \infty} \dfrac{1}{N} \sum^N Y_t Y_{t+k} =  \text{E}[Y_tY_{t+k}]
$$

### 비정상 확률 과정(non-stationary process)

비정상 확률 과정은 정상 확률 과정이 아닌 확률 과정이다.

추세를 가지는 경우가 해당한다. 일차 모멘트($$E[y_t]$$) 가 0이 아니며 시간에 따라 변화한다.

추세가 없고 $$E[y_t] = 0$$ 이지만 분산 $$Var[y_t]$$ 이 시간에 따라 변하는 경우

시계열 자료들은 동일한 확률과정의 샘플들이다. 하나 하나의 샘플(시계열 자료)만 보면 마치 추세가 있는 것처럼 보이지만 사실은 확률과정의 분산이 시간에 따라 커지는 것이다. 이런 경우를  **확률적 추세(stochastic trend)**를 가진다고 말하기도 한다. 

