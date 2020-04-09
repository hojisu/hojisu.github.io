<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 일반 선형확률과정 모형(general linear precess model)

일반 선형확률과정 모형은 정상 확률 과정(stationary process)에서 가장 일반적으로 사용되는 모형이다. 시계열이 가우시안 백색 잡음의 현재 값과 과거 값들의 선형 조합으로 이루어져 있다고 가정한다.

$$\epsilon_t$$ : 가우시안 백색 잡음

$$\psi$$ : 백색 잡음에 곱해지는 가중 계수(weight coefficient)

$$
Y_t = \epsilon_t + \psi_1 \epsilon_{t-1}  + \psi_2 \epsilon_{t-2}  + \psi_3 \epsilon_{t-3}  + \cdots
$$

$$
\sum_{i=1}^{\infty} \psi_i^2 < \infty
$$

Lag 연산자 수식 의미

$$
Y_{t-1} = LY_t \\
Y_{t-2} = L^2Y_t \\
Y_{t-k} = L^{k}Y_t \\
$$

일반 선형확률과정 모형은 계수의 특성에 따라 아래와 같은 하위 모형으로 분류할 수 있다

### MA(Moving Average) 모형 

MA 모형은 일반 선형 확률 모형의 차수가 유한(finite)한 경우이다. q차수의 MA 모형은 MA(q)로 표기

$$
Y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}
$$

MA 수식을 Lag 연산자(opertor)를 사용하여 표기하면 아래와 같다.

$$
Y_t = \epsilon_t + \theta_1 L \epsilon_{t} + \theta_2 L^2 \epsilon_{t} + \cdots + \theta_q L^q \epsilon_{t}
$$

$$
Y_t = (1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q) \epsilon_{t}
$$

$$
Y_t = \theta(L) \epsilon_t
$$

위의 식에서 $$\theta(L)$$ 은 다음 다항식을 뜻한다.

$$
\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots \theta_q L^q
$$

### AR(Auto-Regressive) 모형

AR모형은 자기 자신의 과거값에 의존하는 모형이다. p차수의 AR 모형은 AR(p)로 표기

$$
Y_t = -\phi_1 Y_{t-1} - \phi_2 Y_{t-2} - \cdots - \phi_p Y_{t-p}  + \epsilon_t
$$

AR 수식을 Lag 연산자(opertor)를 사용하여 표기하면 다음과 같다.

$$
(1 + \phi_1 L + \phi_2 L^2 + \cdots + \phi_p L^p) Y_t = \epsilon_{t}
$$

$$
\phi(L) Y_t = \epsilon_t
$$

$$
\phi(L) = 1 + \phi_1 L + \phi_2 L^2 + \cdots \phi_p L^p
$$

### ARMA(Auto-Regressive Moving Average) 모형

ARMA모형은 AR 모형과 MA 모형을 합친 모형이다.

$$
Y_t = -\phi_1 Y_{t-1} - \phi_2 Y_{t-2} - \cdots - \phi_p Y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}
$$

$$
\phi(L) Y_t = \theta(L) \epsilon_t
$$

  