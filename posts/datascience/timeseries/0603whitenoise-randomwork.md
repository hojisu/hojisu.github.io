<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 백색 잡음과 랜덤 워크

### Summary
- 백색잡음은 확률과정을 구성하는 개별 확률변수들이 서로 독립이고 동일한 확률분포를 따르는 확률과정을 말한다. 
- 랜덤워크는 백색잡음을 누적한 확률 과정이다.
__________________________________

### 백색 잡음(white noise)

백색잡음은 확률 과정을 구성하는 모든 개별 확률 변수 $$\epsilon_t$$ 들이 서로 독립(independent)이고 동일한 확률 분포를(idenically distributed)를 따르는 확률 과정을 말한다. **i.i.d. 가정** 이라고도 한다.

특성
- 정상 과정(stictly stationary process) 이다.
- 시차(lag)가 0일 경우, 자기 공분산은 확률 분포의 분산이 되고 시차가 0이 아닌 경우 자기공분산은 0이다.
 
$$
\gamma_l = \begin{cases} \text{Var}[\epsilon_t] & \;\; \text{ for } l = 0 \\  0 & \;\; \text{ for }  l \neq 0 \end{cases}
$$

- 시차(lag)가 0일 경우 자기상관계수는 1이 되고, 시차가 0이 아닌 경우 자기상관계수는 0이다.

$$
\rho_l = \begin{cases} 1 & \;\; \text{ for } l = 0 \\  0 & \;\; \text{ for }  l \neq 0 \end{cases}
$$

#### 가우시안 백색 잡음(Gaussina White noise)

가우시안 백색 잡음은 확률 분포가 표준 가우시안 정규 분포인 백색 잡음이다.

~~~python
e = sp.stats.norm.rvs(size=300)
plt.plot(e)
plt.show()
~~~

#### 비-가우시안 백색 잡음

비-가우시안 백색 잡음은 백색 잡음을 이루는 확률 분포가 반드시 정규 분포일 필요는 없다. 예를 들어 가장 단순한 경우로서 {1,−1}로 구성되고 1이 나올 확률 𝑝=0.5인 베르누이 확률 과정도 백색 잡음이 된다.

~~~python
e = sp.stats.bernoulli.rvs(0.5, size=100) * 2 - 1
plt.step(np.arange(len(e)), e)
plt.ylim(-1.1, 1.1)
plt.show()
~~~

### 이산 시간 랜덤 워크(discrete-time random walk)

이산 시간 랜덤워크는 백색 잡음(white noise)을 누적한 확률 과정이다.

$$
W_t = W_{t-1} + \epsilon_t
$$

이산 시간 랜덤 워크 특성
-  기댓값은 0
-  분산은 시간에 비례
-  자기공분산은 두 시간 중 빠른 시간에 비례
-  자기상관계수는 두 시간의 비율의 제곱근에 비례

