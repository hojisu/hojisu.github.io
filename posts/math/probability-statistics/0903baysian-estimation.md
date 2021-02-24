<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 베이지안 추정

### Summary

- 베이지안 모수 추정(Bayseian parameter estimation)은 모수값이 가질 수 있는 모든 가능성을 분포로 계산하는 작업니다. 모수도 하나의 확률변수라고 생각한다. 장점은 순차적인 계산이 가능합니다.
- 모수적방법은 모수의 분포를 다른 확률분포모형으로 사용하는 방법이다. 비모수적 방법은 모수의 분포와 동일한 분포를 가지는 실제 숫자 집합을 생성하여 히스토그램이나 표본기술통계값으로 분포를 표현하는 방법이다. 
_____________

### 베이지안 모수 추정의 기본 원리

베이지안 모수 추정 방법에서는 모수도 확률적인 값으로 표현한다. 

**베이지안 모수 추정 방법** 은 주어진 데이터 $${x_1, ... , x_N}$$ 를 기반으로 모수 $$\mu$$ 의 조건부 확률분포 $$p(\mu \vert x_{1},\ldots,x_{N})$$ 를 계산하는 작업이다. 조건부 확률분포를 구하므로 베이즈 정리를 사용한다.

$$
p(\mu \mid x_{1},\ldots,x_{N}) = \dfrac{p(x_{1},\ldots,x_{N} \mid \mu) \cdot p(\mu)}{p(x_{1},\ldots,x_{N})} \propto p(x_{1},\ldots,x_{N}  \mid \mu )  \cdot p(\mu)
$$

- $$p(\mu)$$ 는 **모수의 사전(Prior)분포** 이다. 모수의 분포에 대해 아무런 지식이 없는 경우에는 균일(uniform)분포 Beta(1,1)나 0을 중심으로 가지는 정규분포 $$\mathcal{N}(0,\sigma_0^2)$$ 등을 사용할 수 있다. 
- $$p(\mu \mid x_{1},\ldots,x_{N})$$ 는 **모수의 사후(Posterior)분포** 이다. 수학적으로는 데이터 $$x_{1},\ldots,x_{N}$$ 가 주어진 상태에서의 $$\mu$$ 에 대한 조건부 확률분포이다. 베이지안 모수 추정 작업을 통해 구하고자 하는 것
- $$p(x_{1},\ldots,x_{N} \mid \mu)$$ 는 **가능도(likelihood) 함수** 이다. 

##### 모수의 분포 방법

(1) 모수적(parametric) 방법

- 모수의 분포를 다른 확률 분포 모형을 사용하여 나타낸다. 이렇게 하려면 모수를 나타내는 확률분포함수의 모수를 다시 계산해야 하는데 이를 **하이퍼모수(hyper-parameter)**라고 부른다. 모수적 방법을 사용한 베이지안 모수 추정은 결국 하이퍼모수의 값을 계산하는 작업이다.

(2) 비모수적(non-parametric) 방법

- 모수의 분포와 동일한 분포를 가지는 실제 숫자의 집합을 생성하여 히스토그램이나 표본 기술통계값으로 분포를 표현한다. MCMC(Markov chain Monte Carlo)와 같은 몬테카를로(Monte Carlo) 방법이 비모수적 방법이다.

#### 베르누이 확률변수의 베이지안 모수추정

베르누이 확률별수의 모수 $$\mu$$ 를 베이지안 추정법으로 추정해보자.

베르누이 분포의 모수는 0부터 1사이의 값을 가지므로 사전 분포는 하이퍼 모수를 a=b=1인 베타 분포라고 가정하자.

$$
p(\mu) \propto \mu^{a−1}(1−\mu)^{b−1} \;\;\; (a=1, b=1)
$$

데이터는 모두 독립적인 베르누이 분포의 곱이므로 가능도 함수는 다음과 같다.

$$
p(x_{1},\ldots,x_{N} \mid \mu) = \prod_{i=1}^N  \mu^{x_i} (1 - \mu)^{1-x_i}
$$

베이즈 정리를 사용하면 사후분포가 다음처럼 갱신된 하이퍼모수 $$a'$$, $$b'$$를 가지는 또다른 베타 분포가 된다.

$$
\begin{aligned}
p(\mu \mid x_{1},\ldots,x_{N})
&\propto p(x_{1},\ldots,x_{N} \mid \mu)  p(\mu) \\
&= \prod_{i=1}^N  \mu^{x_i} (1 - \mu)^{1-x_i} \cdot \mu^{a−1}(1−\mu)^{b−1}  \\
&= \mu^{\sum_{i=1}^N x_i + a−1} (1 - \mu)^{\sum_{i=1}^N (1-x_i) + b−1 }   \\
&= \mu^{N_1 + a−1} (1 - \mu)^{N_0 + b−1 }   \\
&= \mu^{a'−1} (1 - \mu)^{b'−1 }   \\
\end{aligned}
$$

켤레 사전확률분포(conjugate prior)는 사전분포와 사후분포가 모수값만 다르고 함수 형태가 같은 확률밀도함수로 표현될 수 있도록 해주는 사전분포이다. 

갱신된 하이퍼 모수의 값은 다음과 같다.

$$
\begin{aligned}
a' &= N_1 + a \\
b' &= N_0 + b 
\end{aligned}
$$

베이지안 모수 추정의 장점은 **순차적(sequential)** 계산이 가능하다는 점이다. 불확실성을 표현할 수 있다. 

#### 카테고리 확률변수의 베이지안 모수추정

클래스 개수가 $$K$$ 인 카테고리 확률변수의 모수 $$\mu$$ 벡터를 베이지안 추정법으로 추정

카테고리 확률변수의 모수의 각 원소는 모두 0부터 1사이의 값을 가지므로 사전 분포는 하이퍼 모수 $$𝛼_𝑘=1$$인 디리클리 분포라고 가정한다.

$$
p(\mu) \propto \prod_{k=1}^K \mu_k^{\alpha_k - 1} \;\;\; (\alpha_k = 1, \; \text{ for all } k)
$$

데이터는 모두 독립적인 카테고리 분포의 곱이므로 가능도 함수는 다음처럼 다항 분포이다.

$$
p(x_{1},\ldots,x_{N} \mid \mu) = \prod_{i=1}^N  \prod_{k=1}^K \mu_k^{x_{i,k}}
$$

베이즈 정리로 사후 분포를 구하면 다음과 같이 갱신된 하이퍼 모수. $$\alpha'_i$$를 가지는 디리클리 분포가 된다.

$$
\begin{aligned}
p(\mu \mid x_{1},\ldots,x_{N})
&\propto p(x_{1},\ldots,x_{N} \mid \mu)  p(\mu) \\
&= \prod_{i=1}^N  \prod_{k=1}^K \mu_k^{x_{i,k}} \cdot \prod_{k=1}^K \mu_k^{\alpha_k - 1}  \\
&= \prod_{k=1}^K  \mu^{\sum_{i=1}^N x_{i,k} + \alpha_k − 1}   \\
&= \prod_{k=1}^K  \mu^{N_k + \alpha_k −1}   \\
&= \prod_{k=1}^K  \mu^{\alpha'_k −1}   \\
\end{aligned}
$$

이 경우에도 마찬가지로 디리클리분포는 켤레 분포임을 알 수 있다. 갱신된 하이퍼 모수의 값은 다음과 같다.

$$
\alpha'_k = N_k + \alpha_k
$$

#### 정규 분포의 기댓값 베이지안 모수추정

이번에는 정규 분포의 기댓값 모수를 베이지안 방법으로 추정한다. 분산 모수 $$𝜎^2$$은 알고 있다고 가정한다.

기댓값은 $$−∞$$부터 $$∞$$까지의 모든 수가 가능하기 때문에 모수의 사전 분포로는 정규 분포를 사용한다.

$$
p(\mu) = N(\mu_0, \sigma^2_0) = \dfrac{1}{\sqrt{2\pi\sigma_0^2}} \exp \left(-\dfrac{(\mu-\mu_0)^2}{2\sigma_0^2}\right)
$$

데이터는 모두 독립적인 정규 분포의 곱이므로 가능도 함수는 다음과 같다.

$$
p(x_{1},\ldots,x_{N} \mid \mu) = \prod_{i=1}^N N(x_i \mid \mu )  = \prod_{i=1}^N  \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp \left(-\dfrac{(x_i-\mu)^2}{2\sigma^2}\right) \\
\begin{aligned}
p(\mu \mid x_{1},\ldots,x_{N})  
&\propto p(x_{1},\ldots,x_{N} \mid \mu) p(\mu) \\
&\propto \exp \left(-\dfrac{(\mu-\mu'_0)^2}{2\sigma_0^{'2}}\right) \\
\end{aligned}
$$

베이즈 정리를 이용하여 사후 분포를 구하면 다음과 같이 갱신된 하이퍼 모수를 가지는 정규 분포가 된다.

$$
\begin{aligned}
\mu'_0 &= \dfrac{\sigma^2}{N\sigma_0^2 + \sigma^2}\mu_0 + \dfrac{N\sigma_0^2}{N\sigma_0^2 + \sigma^2} \dfrac{\sum x_i}{N} \\
\dfrac{1}{\sigma_0^{'2}} &= \dfrac{1}{\sigma_0^{2}} + \dfrac{N}{\sigma^{'2}}
\end{aligned}
$$

$$\mu'_0$$ 에서 N은 새로운 데이터값이다.  $$\dfrac{\sigma^2}{N\sigma_0^2 + \sigma^2}$$  와 $$\dfrac{N\sigma_0^2}{N\sigma_0^2 + \sigma^2}$$ 은 가중치 $$w_1, w_2$$ 로 볼 수 있다. 새로 들어온 데이터의 N의 갯수가 처음 데이터보다 갯수가 적을 경우 $$w_1$$ 가중치에 신뢰도를 더 준다.

분산은 작아지면서 N(데이터 갯수)에 비례하면서 정밀도가 증가한다. $${\sigma_0^{'2}}$$ 은 옛날 정보의 믿음 정보이다. 

___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다.