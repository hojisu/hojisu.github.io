<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 시계열 자료와 확률 과정

### 확률 과정(Stochastic process, Random process)

확률과정은 상관 관계를 가지는 무한개의 확률 변수의 순서열(sequence of infinite random variables)이다.

확률 과정에 포함된 확률 변수는 시간 변수 t를 기준으로 정렬한다. 

**시계열 자료(time series data)**는 확률 과정의 표본이다.

**이산 시간 확률 과정(discrete time stochastic)**는 시간 변수 t를 정수만 사용한 것이다.
- 서로 상관관계를 가지는 복수개의 확률변수이며 우리가 살고 있는 이 세계 자체가 확률 과정의 하나의 표본이다. 

### 앙상블 평균(ensemble average)

앙상블 평균은 확률 과정 Y의 특정 시간에 대한 기닷값 $$E[Y_t]$$ 은 복수의 시계열 자료 표본에서 특정 시간 t의 값만을 평균한 것이다.

앙상블 평균에 대한 추정값을 얻기 위해서는 확률 과정이 **정상 과정(stationary process)**이며 **에르고딕(ergodic process)**이라는 가정이 있어야 한다.

### 확률 과정의 기댓값, 자기공분산, 자기상관계수

확률 과정의 특성은 개별 시간 변수에 대한 확률 변수들의 결합 확률 밀도 함수를 사용하여 정의한다. 

**확률 과정의 기댓값**은 시간 변수 t에 대한 확률 변수 $$Y_t$$ 의 기댓값이다.

$$
\mu_t = \text{E}[Y_t]
$$

**확률 과정의 자기공분산(auto-covariance)**은  시간 변수 t에 대한 확률 변수 $$Y_t$$ 와 시간 변수 s에 대한 확률 변수$$Y_s$$ 의 공분산이다. 

$$
\gamma_{t,s} = \text{Cov}[Y_t, Y_s]  = \text{E}\left[(Y_t-\text{E}[Y_t])(Y_s-\text{E}[Y_s])\right]
$$

**확률 과정의 자기상관계수(auto-correlation)**은 시간 변수 t에 대한 확률 변수 $$Y_t$$ 와 시간 변수 s에 대한 확률 변수 $$Y_s$$ 의 상관계수이다.

$$
\rho_{t,s} = \text{Corr}[Y_t, Y_s] = \dfrac{ \text{Cov}[Y_t, Y_s] }{\sqrt{\text{Var}[Y_t]\text{Var}[Y_s]}} = \dfrac{\gamma_{t,s}}{\sqrt{\gamma_t\gamma_s}}
$$

성질

  $$
  \begin{eqnarray} \gamma_{t,t} &=& \text{Var}[Y_t] \\ \gamma_{t,s} &=& \gamma_{s,t} \\ \left| \gamma_{t,s} \right| &\leq& \sqrt{\gamma_{t,t} \gamma_{s,s} } \\ \rho_{t,t} &=& 1 \\ \rho_{t,s} &=& \rho_{s,t} \\ \left| \rho_{t,s} \right| &\leq& 1 \\ \end{eqnarray}
  $$

