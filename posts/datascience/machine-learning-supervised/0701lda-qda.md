<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# LDA(선형판별분석법) QDA(이차판별분석법)

가능도 즉 $$y$$ 의 클래스에 따른 $$x$$ 의 분포에 대한 정보를 먼저 알아낸 후, 베이즈 정리를 이용하여 주어진 $$x$$ 에 대한 $$y$$의 확률분포를 구한다. 

### 생성모형

베이즈 정리를 이용하여 $$p = (y=k|x)$$ 를 구한다.

$$
P(y = k \mid x) = \dfrac{P(x \mid y = k)\; P(y = k)}{P(x)}
$$

<!-- 분류문제를 풀기 위해서는 각 클래스 $$k$$ 에 대한 확률을 비교하여 가장 큰 값을 선택한다. 따라서 모든 클래스에 대해 값이 같은 분모 $$P(x)$$ 은 굳이 계산하지 않아도 괜찮다.

$$
P(y = k \mid x) \;\; \propto \;\; P(x \mid y = k) \; P(y = k)
$$

여기에서 사전확률 $$P(y=k)$$ 는 특별한 정보가 없는 경우, 다음처럼 계산한다.

$$
P(y = k) \approx \frac{\;\;\;\; y = k \text{인 데이터의 수 } \text{ }\;\;\;\;\;\; }{{\text{ }}\text{    모든 데이터의 수    }\;\;\;\;\;\;\;\;}
$$

만약 다른 지식이나 정보를 알고 있는 사전확률값이 있다면 그 값을 사용하면 된다. 

$$y$$ 에 대한 $$x$$ 의 조건부확률인 가능도는 다음과 같이 계산한다.
- $$P(x|y=k)$$ 가 특정한 확률분포 모형을 따른다고 가정한다. 즉, 확률밀도함수의 형태를 가정한다. 
- $$k$$ 번째 클래스에 속하는 학습 데이터 $${x_1, \dots, x_N }$$ 을 사용하여 이 모형의 모수값을 구한다. 
- 모수값을 알고 있으므로 $$P(x|y=k)$$의 확률밀도함수를 구한 것이다. 즉, 새로운 독립변수 $$x$$가 어떤 값이 되더라도 $$P(x|y=k)$$의 값을 계산할 수 있다. 

### 이차판별분석법(QDA)

이차판별분석법의 가정은 독립변수 $$x$$ 가 실수이고 확률분포가 다변수 정규분포라고 가정한다. 단, $$x$$ 분포의 위치와 형태는 클래스에 따라 달라질 수 있다.  
- y=k 클래스마다 다른 다변수 정규분포이다.

$$
p(x \mid y = k) = \dfrac{1}{(2\pi)^{D/2} |\Sigma_k|^{1/2}} \exp \left( -\dfrac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k) \right)
$$

독립변수 $$x$$ 에 대한 $$y$$ 클래스의 조건부확률분포는 베이즈 정리와 전체 확률의 법칙으로 구할 수 있다. 

$$
P(y=k \mid x) = \dfrac{p(x \mid y = k)P(y=k)}{p(x)} = \dfrac{p(x \mid y = k)P(y=k)}{\sum_l p(x \mid y = l)P(y=l) }
$$

### 선형판별분석법(LDA)

선형판별분석법의 가정은 각 Y 클래스에 대한 독립변수 X의 조건부확률분포가 공통된 공분산 행렬을 가지는 다변수 정규분포라고 가정한다. 

$$
\Sigma_k = \Sigma \ for \ all \ k
$$

조건부확률분포 정리

$$
\begin{eqnarray}
\log p(x \mid y = k) 
&=& \log \dfrac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} -  \dfrac{1}{2} (x-\mu_k)^T \Sigma^{-1} (x-\mu_k) \\
&=& C_0 - \dfrac{1}{2} (x-\mu_k)^T \Sigma^{-1} (x-\mu_k) \\
&=& C_0 - \dfrac{1}{2} \left( x^T\Sigma^{-1}x - 2\mu_k^T \Sigma^{-1}x + \mu_k^T \Sigma^{-1}\mu_k \right) \\
&=& C(x)  + \mu_k^T \Sigma^{-1}x - \dfrac{1}{2} \mu_k^T \Sigma^{-1}\mu_k \\
\end{eqnarray}
$$

$$
\begin{eqnarray}
p(x \mid y = k) 
&=& C'(x)\exp(w_k^Tx + w_{k0}) \\
\end{eqnarray}
$$

위의 식에서 $$C'(x) = \exp C(x)$$ 이다.

$$
\begin{eqnarray}
P(y=k \mid x) 
&=& \dfrac{p(x \mid y = k)P(y=k)}{\sum_l p(x \mid y = l)P(y=l) } \\
&=& \dfrac{C'(x)\exp(w_k^Tx + w_{k0}) P(y=k)}{\sum_l C'(x)\exp(w_l^Tx + w_{l0})P(y=l) } \\
&=& \dfrac{C'(x)\exp(w_k^Tx + w_{k0}) P(y=k)}{C'(x)\sum_l \exp(w_l^Tx + w_{l0})P(y=l) } \\
&=& \dfrac{P(y=k) \exp(w_k^Tx + w_{k0}) }{\sum_l P(y=l) \exp(w_l^Tx + w_{k0})} \\
&=& \dfrac{P(y=k) \exp(w_k^Tx + w_{k0}) }{P(x)} \\
\end{eqnarray}
$$

$$P(x)$$ 는 $$y$$ 클래스값에 영향을 받지 않는다. 

$$
\log P(y=k \mid x) = \log P(y=k) + w_k^Tx + w_{k0} - \log{P(x)} = w_k^Tx + C''_k
$$

모든 클래스 $$k$$ 에 대해 위와 같은 식이 성립하므로 클래스 $$k_1$$ 과 클래스 $$k_2$$ 의 경계선, 즉 두 클래스에 대한 확률값이 같아지는 $$x$$ 위치를 찾으면 다음과 같다.

$$
w_{k_1}^Tx + C''_{k_1} = w_{k_2}^Tx + C''_{k_2} \\
(w_{k_1} - w_{k_2})^Tx + (C''_{k_1} - C''_{k_2}) = 0 \\
w^Tx + C = 0
$$

판별함수가 $$x$$ 에 대한 선형방정식이 되고 경계선의 모양이 직선이 된다. 
 -->
