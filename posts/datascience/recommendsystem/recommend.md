<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# Recommender System

### Summary

- 베이스라인 모형 사용자 아이디, 상품 아이디, 두 개의 카테고리 값 입력에서 평점의 예측치을 예측하는 가장 단순한 회귀분석모형으로 사용자와 상품 특성에 의한 평균 평점의 합으로 나타난다.
- collaborative filter 모형은 평점 행렬이 가진 특정한 패턴을 찾아서 이를 평점 예측에 사용하는 방법이다. CF 방법도 사용자나 상품 기준으로 평점의 유사성을 살피는 Neighborhood 모형과 행렬의 수치적 특징을 이용하는 Latent Factor 모형이 있다. 
- Latent Factor Model은 관찰된 데이터와 잠재되어 있는 데이터를 연결시키는 기법이다.
- Matrix Factorization 커다란 평점 행렬을 사용자와 아이템 Latent Factor로 분해하고, 이를 각각 학습시킨다.
- Implicit Dataset이 주어질 경우, Latent Factor model 가운데 Matrix Factorization이 적합하다.
- Contents Based Filtering에서는 각각의 사용자와 아이템에 대하여 프로필을 작성하고, 이를 기반으로 추천을 한다.

__________

____________

### 추천시스템

추천시스템이란 사용자가 선호하는 상품을 예측하는 시스템이다. 사용자 아이디와 상품 아이디라는 두 개의 카테고리 입력과 평점 출력을 가지는 예측 시스템이다. 

### 추천시스템 알고리즘

두 개의 카테고리 값 입력에서 하나의 실수 값 출력을 예측하는 회귀모형이지만 여러가지 방법으로 예측 성능을 향상시키고 있다.
- 베이스 라인 모형
- Collaborative Filtering
    - Neighborhook Models : User-based CF, Item-based CF
    - Latent Factor Models : Matrix Factorization, SVD 
- Content-Based Recommendation

### 베이스라인 모형(baseline model)

사용자 아이디 $$u$$ , 상품 아이디 $$i$$, 두 개의 카테고리 값 입력에서 평점 $$r(u, i)$$ 의 예측치 $$\hat r(u,i)$$ 을 예측하는 가장 단순한 회귀분석모형으로 다음과 같이 사용자와 상품 특성에 의한 평균 평점의 합으로 나타난다. 

$$\mu$$ 는 전체 평점의 평균이고, $$b(u)$$ 는 동일한 사용자에 의한 평점의 조정값, $$b(i)$$ 는 동일한 상품에 대한 평점 조정값이다.

$$
\hat{r}(u,i) = \mu + b(u) + b(i)
$$

베이스라인 모형은 다음 오차 함수를 최소화하도록 구해진다. 

$$R_{train}$$ 은 실제 평점이 존재하는 학습용 데이터 집합니다. 

$$
\sum_{u,i \in R_{train}} \left(r(u,i) - \hat{r}(u,i))\right)^2
$$

과최적화를 피하기 위해 다음과 같이 정규화 항을 추가할 수 있다. 

$$
\sum_{u,i \in R_{train}} \left(r(u,i) - \hat{r}(u,i))\right)^2 + \lambda \left(b(u)^2 + b(i)^2 \right)
$$

#### 추천성능 평가기준

- RMSE(Root Mean Squared Error)

$$
\text{RMSE} = \sqrt{\frac{1}{|\hat{R}|} \sum_{\hat{r}(u,i) \in \hat{R}}(r(u,i) - \hat{r}(u,i))^2}
$$

- MAE(Mean Absolute Error)

$$
\text{MAE} = \dfrac{1}{|\hat{R}|} \sum_{\hat{r}(u,i) \in \hat{R}}|r(u,i) - \hat{r}(u,i)|
$$

- FCP(Fraction of Concordant Pairs)
  - 회귀 분석 $$i$$ 번쨰 데이터와 $$j$$ 번째 데이터에 대해 실제 데이터 $$y_i, y_j$$ 와 예측 데이터 $$\hat y_i, \hat y_j$$ 사이의 증가 방향이 같으면 condordant pair라고 한다. $$sign(y_i-y_j) = sign(\hat y_i - \hat y_j)$$

$$
\text{FCP} = \dfrac{\text{number of concordant pairs}}{\text{number of discordant pairs}}
$$


### Collaborative Filter (CF)

![image-20200311221845516](../../../ressource/img/image-20200311221845516.png)

모든 사용자의 데이터를 균일하게 사용하는 것이 아니라 평점 행렬이 가진 특정한 패턴을 찾아서 이를 평점 예측에 사용하는 방법이다. CF 방법도 사용자나 상품 기준으로 평점의 유사성을 살피는 Neighborhood 모형과 행렬의 수치적 특징을 이용하는 Latent Factor 모형이 있다. 

- 내가 남긴 평점 데이터를 가지고 나와 취향이 비슷한 사람이 선호하는 아이템을 추천한다.
- 프로필 데이터 없이 사용자와 과거 행동 데이터만 가지고 추천을 진행한다. 
- 도메인에 제약없이 데이터 셋을 쌓기가 쉽다. 
- 신규사용자의 경우 관찰된 행동 데이터가 없거나 적기 때문에 추천의 정확도가 급격히 떨어지는 Cold start 문제가 발생한다.

### Explicit, Implicit

Explicit(선호와 비선호를 명확하게 구분해준 데이터 셋)는 A score, such as a rating or a like가 있다.
  - Explicit Dataset에서 주어진 평점만으로 사용자의 선호/비선호를 알 수 있습니다. 이 경우 빈 matrix 에 해당하는 데이터들을 제외한 평접 데이터만으로 사용자의 선호도를 학습한 다음 추축은 한다.
Implicit(선호와 비선호의 구분 없이 행동의 빈도수만 기록한 데이터 셋) : Not as obvious in terms of preference, such as a click, view, or purchase, 호불호 구분 없이 사용자가 아이템을 얼마나 소비하였는지를 기록한 데이터 셋이다. **Implicit Dataset**에서는 우리는 사용자가 어떤 아이템을 선호하는지는 알지만, 어떤 아이템을 비선호 하는지는 알지 못한다. 빈 matrix 안에는 사용자가 좋아하는 아이템이 있을수도 없을 수도 있다. 때문에 **빈 matirx에 해당하는 영역 역시 데이터로 포함하여 사용자의 선호도를 학습한다.**

#### Neighborhood 모형, Memory-based CF

특정 사용자의 평점을 예측하기 위해 사용하는 것이 아니라 **해당 사용자와 유사한(similar)사용자에 대해 가중치를 준다.** 

평점데이터를 가지고 양의 선형 관계 혹은 음의 선형관계를 계산한다. 이는 평점 자체에 선호와 비선호가 구분되어 있는 Explicit Dataset에 적합하다. 

- 사용자 기반(User-based) CF : 해당 사용자와 유사한 사용자를 찾는 방법 즉, 평점 행렬에서 유사한 사용자 행 벡터를 찾아서 이를 기반으로 빈 데이터를 계산하는 방법. 유사도는 주로 Pearson 상관계수를 통해 구한다. 피어슨 상관계수는 양으로 변할 때 함께 양으로 변하고, 음으로 변할때 함께 음으로 변하는 정도를 나타낸다.
  - 예를 들어 평점 데이터를 기반으로 유저 간의 상관도를 계산한다고 해보겠다. 1에 가까울 수록 서로 유사한 아이템에 높은 선호도를 매긴다고 볼 수 있다. 이를 통해서 유사한 사용자가 좋아한 아이템을 추천해 줄 수 있다.
- 상품 기반(Item-based) CF : 특정한 상품에 대해 사용자가 준 점수 즉, 평점 행렬의 상품 열 벡터의 유사성을 찾고 특정 상품과 유사한 평점 정보를 가지는 상품들로 해당 상품의 빈 데이터를 예측하는 방법. 특정 사용자가 아직 평점을 남기지 않은 아이템을 추천해야할 경우, 이 K개의 가까운 이웃들에게 사용자가 남긴 평점을 가지고 해당 아이템에 내릴 평점을 예측할 수 있다.


### Latent Factor 모형

Latent Factor Model은 관찰된 데이터와 잠재되어 있는 데이터를 연결시키는 기법이다. 유저의 비선호도가 반영되지 않은 Implicit Dataset은 Latent Factor model이 더 적합하다.
- 예를 들어 우리의 상황에 적용하면 주어진 평점 데이터로 아직 알지 못하는 사용자와 아이템의 특성을 찾아내는 것이다.

긴 사용자 특성이나 상품 특성을 몇개의 요인 벡터로 간략화(approximate)할 수 있다는 가정에서 출발한 모형이다. PCA를 사용하면 긴 특성벡터를 소수의 차원으로 축소할 수 있듯이 사용자의 특성도 차원 축소 할 수 있다. 

우리에게 주어진 데이터는 사용자와 평점을 나타내는 커다란 행렬이다. 사용자와 아이템, 두 Latent Factor를 알아내고자 한다. 이 경우 커다란 평점 행렬을 사용자와 아이템 Latent Factor로 분해하고, 이를 각각 학습시키는 Matrix Factorization 기법이 적합하다.

### Matrix Factorization

Implicit Dataset이 주어질 경우, Latent Factor model 가운데 Matrix Factorization이 적합하다.

![image-20200311220637018](../../../resource/img/image-20200311220637018.png)

$$R$$ : original rating data matrix 
$$N_u$$ : number of users 
$$N_i$$ : number of items

$$R$$은 사용자가 아이템에 남긴 평점을 행렬로 나타낸 것
$$R$$ = $$N_u$$ x $$N_i$$

$$N_f$$ : dimension of latent factor
$$X$$ : user lantent factor matrix ($$N_f$$ x $$N_u$$)
$$Y$$ : item latent factor matrix ($$N_f$$ x $$N_i$$)

$$N_f$$는 Matrix Factorization 학습 시에 정하는 임의의 차원 수이며 보통 50~200 사이로 설정
$$X$$, $$Y$$는 사용자와 아이템의 Latent Factor 행렬을 나타내며 우리가 학습시키고자 하는 대상
이 행렬들의 값은 아주 작은 랜덤한 값들로 초기화된다.(R에 있는 값을 쪼개어 생성하는 것이 아니다)

$$x_u$$ : user latent vector of specific index $$u$$ 
$$y_u$$ : item latent vaetor of specific index $$i$$

$$x_u$$와 $$y_i$$는 각각 특정 사용자와 특정 아이템의 특징 벡터를 나타낸다.
여기서 벡터는 1차원 array를 뜻하며 $$x_u$$, $$y_i$$는 각각 $$N_f$$ x 1 크기의 열벡터(column vector)로 볼 수 있다.
$$x_u$$를 특정 사용자의 Latent Vector, $$y_i$$를 특정 아이템의 Latent Vector라고 하겠다.

![image-20200311220911256](../../../resource/img/image-20200311220911256.png)

$$R'$$ : predicted rating data matrix

$$X$$ 행렬의 전치 행렬(Transpose)를 구하면 $$N_u$$ x $$N_f$$ 크기의 행렬이 된다. 이를 $$Y$$행렬과 곱해주어 $$N_u$$ x $$N_i$$ 크기의 평점 예측 행렬을 구한다. 예측된 하나의 평점 행렬의 각각의 값들은 다음과 같은 곱셈을 통해서 계산된다.

$$
r_{ui} = x_u^Ty_i
$$

해석해보면 특정 사용자의 Latent Vector와 특정 아이템의 Latent Vector를 곱하여 평점을 계산함을 알 수 있다. 
Latent Factor Matrix가 적절하게 학습이 되었다면 예측된 $$R'$$은 $$R$$과 유사한 결과를 낼 것이라 예상할 수 있다.

모든 사용자와 상품에 대해 다음 오차함수를 최소화하는 요인 벡터를 찾아낸다. 

$$
R \approx PQ^T
$$

-  $$R \in \mathbf{R}^{m \times n}$$ : m 사용자와 n 상품의 평점 행렬
-  $$P \in \mathbf{R}^{m \times k}$$ : m 사용자와 k 요인의 관계 행렬
-  $$Q \in \mathbf{R}^{n \times k}$$ : n 상품과 k 요인의 관계 행렬

#### SVD (Singular Value Decomposition)

SVD는 MF 문제를 푸는 방법 중 하나이다. 

m x n 크기의 행렬 R은 다음과 같이 세 행렬의 곱으로 나타낼 수 있다. 이를 SVD라고 한다. 
  
$$
R=U\Sigma V^T
$$

-  $$U$$ 는 $$m$$×$$m$$ 크기의 행렬로 역행렬이 대칭 행렬
-  $$\Sigma$$ 는  $$m$$×$$m$$ 크기의 행렬로 비대각 성분이 0
-  $$V$$ 는 $$n$$×$$n$$ 크기의 행렬로 역행렬이 대칭 행렬

$$\Sigma$$의 대각 성분은 특이치라고 하며 전체 특이치 중에서 가장 값이 큰 k개의 특이치만을 사용하여 (Truncated SVD), 다음과 같은 행렬을 만들수 있다.
- $$\hat U$$ 는 $$U$$에서 가장 값이 큰 k개의 특이치에 대응하는 k개의 성분만을 남긴m×k 크기의 행렬
- $$\hat \Sigma$$는 가장 값이 큰 k개의 특이치에 대응하는 k개의 성분만을 남긴 k×k 크기의 대각 행렬
- $$\hat V$$는 $$V$$에서 가장 값이 큰 k개의 특이치에 대응하는 k개의 성분만을 남긴 k×n 크기의 행렬

이 행렬을 다시 조합하면 원래의 행렬과 같은 크기를 가지고 유사한 원소를 가지는 행렬을 만들 수 있다.

$$
\hat{U} \hat{\Sigma} \hat{V}^T =  \hat{R} \approx R
$$

하지만 실제로 평점 행렬은 빈 원소가 많은 sparse 행렬로서 SVD를 바로 적용하기 힘들기 때문에 행렬 $$P$$, $$Q$$는 다음과 같은 모형에 대해 오차 함수를 최소화하여 구한다.
  
$$
\hat{r}(u,i) = \mu + b(u) + b(i) + q(i)^Tp(u) \\
\sum_{u,i \in R_{train}} \left(r(u,i) - \hat{r}(u,i) \right)^2 +
\lambda\left(b(i)^2 + b(u)^2 + ||q(i)||^2 + ||p(u)||^2\right)
$$

### Loss Function

$$X$$와 $$Y$$를 곱하여 얻은 예측 평점 행렬의 오차가 최대한 작아지도록 수식을 구성하면 된다.

$$
\min_{x^*, y^*} \sum_{u, i} c_{ui}(p_{ui} - x_u^Ty_i)^2 
+\lambda (\sum_{u} \Vert x_u \Vert ^2 + \sum_{i} \Vert y_i \Vert ^2)
$$

수식의 앞 부분은 예측된 평점과 실제 평점 간의 차이를 나타낸다. 뒷부분 람다 표기가 붙은 부분은 모델이 오버피팅 하지 않도록 정규화 시켜주는 역할을 수행한다. 즉, 전체 모델에서 파라미터가 미치는 영향력을 감소시켜 줌으로써 모델의 일반화를 돕는다. 람다의 크기는 데이터 셋마다 달라질 수 있으므로, 여러 실험을 통해 조절해 나가는 것이 바람직 한다.


### Optimization

Loss Function을 최소화하도록 $$X$$와 $$Y$$를 학습시켜야 한다. 이러한 최적화를 수행하는 알고리즘으로 Gradient Descent와 Alternating Least Squares 알고리즘이 있다.
Gradient Descent 적용하는 방식은 간단하다. loss를 미분하고, 여기에 learnign rate를 곱한 다음 $$X$$와 $$Y$$ 행렬의 값들을 업데이트 해주는 방식이다.

여기서 우리가 학습시켜야 하는 것은 $$X$$와 $$Y$$ 행렬 두개이고, 이 둘은 곱셈으로 묶여있다.($$x_u^T$$ x $$y_i$$) 이 둘을 동시에 최적화 시키는 문제는 Non-convex problem으로 NP에 속한다. Gradient Descent로 이를 최적화 시키는 것은 너무 느리고 많은 반복이 필요하다는 단점이 있다.

**Alternatin Least Squares**는 이러한 문제를 해결 할 수 있다. $$X$$와 $$Y$$ 둘 중 하나를 고정시키고 다른 하나를 최적화 시킨다. 그리고 이 과정을 번갈아가며 반복하여 짧은 시간 내에 최적의 X와 Y를 찾아낸다.

### Alternating Least Squares

Matrix Factorization와 Latent Factor 그리고 이를 학습시키기 위한 Loss function을 최소화하는 사용자와 아이템의 Latent Factor Matrix를 찾아내는 것이 목표! 이를 학습 시키기 위한 방법으로는 Gradient Descent와 Alaternating Least Squares(ALS)이 있으며 ALS가 더 적합한 방법!

ALS는 **사용자와 아이템의 Latent Factor를 한번씩 번갈아가며 학습시킨다.** 두 행렬을 한꺼번에 최적화시키는 것은 어렵기 때문에 둘 중 하나를 고정시킨다.

아이템의 행렬을 상수로 놓고 사용자의 행렬을 학습시키고, 사용자 행렬을 상수로 놓고 아이템 행렬을 학습시키는 방식이다.

이 과정을 계속 반복하면서 최적의 사용자와 아이템 Latent Factor를 학습한다.

#### Loss Function

$$
\min_{x^*, y^*} \sum_{u, i} c_{ui}(p_{ui} - x_u^Ty_i)^2 
+\lambda (\sum_{u} \Vert x_u \Vert ^2 + \sum_{i} \Vert y_i \Vert ^2)
$$

기존의 Loss Function에서는 단순히 원래 rating에서 예측된 rating을 빼준 값의 제곱을 사용하였다.

$$r_{ui}$$ : original rating 
$$x_u^Ty_i$$ : predicted rating 
$$(r_{ui} - x_u^Ty_i)^2$$ : previous loss

새로운 Loss Function에서는 $$r_{ui}$$ 대신 $$c_{ui}$$와 $$p_{ui}$$가 새롭게 추가된 것이다. 이는 평점 예측을 선호도로 나타내는 $$p$$와 그 예측값의 신뢰도를 나타내는 $$c$$로 나누어 준것이다.

$$
p_{ui} = 
\begin{cases}
1 & if \ r_{ui} > 0 \\
0 & if \ r_{ui} = 0
\end{cases}
$$

먼저 선호를 나타내는 $$p$$ 이다. Implicit Dataset에서의 평점은 선호와 비선호를 구분하지 않는다. 그렇기 때문에 평점 데이터가 존재할 경우 선호함을 나타내는 1, 반대의 경우 0으로 바꾸어줍니다.

$$
c_{ui} = 1 + \alpha r_{ui}
$$

해당 데이터의 신뢰도를 나타내는 $$c$$입니다. 우리는 평점이 남아있지 않은 데이터는 모두 0으로 바꾸었다. 여기에는 실제 선호하지만, 평점이 남아있지 않은 데이터들도 포함되어 있다. 즉, **데이터의 신뢰도가 낮은 것**으로 볼 수 있다.

Colloaborative Filtering For Implicit Datasets 논문에서는 신뢰도 변수를 도입하여 평점이 남아 있지 않은 데이터에 대한 예측값도 전체 Loss function에 영향을 주도록 만들었다. 이는 평점이 남아있지 않은 데이터는 모두 학습에서 제외한 Explicit Dataset과 대조적이다. 또한 이러한 데이터는 낮은 $$c$$값을 갖게하여 loss 포함하되 영향력이 작게끔 조절하였습니다. 사용자가 높은 평점을 주어 명확하게 선호임을 파악 할 수 있는 경우엔 영향력이 높아지게끔 합니다.

여기서 '어차피 가중치가 낮으면 평점이 0인 데이터는 있으나 마나한거 아닌가?'라는 의문이 들 수 있다. 하지만 Implicit Dataset을 보면 평점이 낮은 아이템보다 0으로 남아있는 데이터가 훨씬 더 많다. 이러한 행렬 데이터를 Sparse Matrix라고 한다. 기존에 Explicit Dataset을 다루었던 기법들은 이렇게 0으로 남은 데이터들은 버리고 학습을 진행하였다. 하지만 앞서 제시한 신뢰도 지수 $$c$$를 통해서 0으로 남아있는 방대한 아이템들에 대한 예측 값도 Latent Factor 학습에 포함시킬 수 있게 되었다. 또한 가중치가 낮다 하더라도 그 수가 평점이 남은 아이템들에 비해 월등히 많기 때문에 학습에 유의미한 영향을 미치게 된다. 

#### ALS 원리

ALS는 먼저 사용자 혹은 아이템의 Latent Factor 행렬을 아주 작은 랜덤 값으로 초기화합니다. 그 다음 둘 중 하나를 상수처럼 고정시켜 Loss Function을 Convex Function으로 만듭니다. 그리고 이를 미분한 다음 미분 값을 0으로 만드는 사용자 혹은 아이템의 Latent Factor 행렬을 계산합니다. 이 과정을 사용자 한번, 아이템 한번 반복하면서 최적의 X, Y를 찾아내는 것입니다.

먼저 아이템의 Latent Factor행렬을 고정하고 사용자의 Latent Factor를 최적화시키는 과정을 살펴보자. 아이템 행렬을 고정된 상수로 놓고 LossFuction을 미분하면 다음과 같다.

$$
{\triangle L(x_u) \over \triangle x_u} = -2\sum_{i} C_{ui}(p_{ui} - x_u^Ty_i) \times y_i + 2\lambda x_{u}
$$

이 값이 0이 되는 $$X_u$$를 찾기 위해서 미분한 식을 전개하면 다음과 같다.

$$
-2\sum_{i} C_{ui}(p_{ui} - x_u^Ty_i) \times y_i + 2\lambda x_{u} = 0
$$

이를 다시 전개한 다음 좌변 우변을 나누어 정리해주면 다음과 같다.

$$
\sum_{i} C_{ui}(x_u^Ty_i) \times + \lambda x_{u} = \sum C_{ui}(p_{ui}) \times y_i
$$

여기서 $$x_u^Ty_i$$를 계산하면 1 X 1 크기의 스칼라 값이 나온다. 즉 $$x_u^Ty_i$$의 전치 행렬을 구해도 동일한 스칼라 값이 나오게 된다.

$$
\sum_{i} C_{ui}(y_i^Tx_u) \times y_i + \lambda x_{u} = \sum C_{ui}(p_{ui}) \times y_i
$$

$$C_{ui}$$ 역시 스칼라 값이기 때문에 뒤에 오는 $$y_i^Tx_u$$와 순서를 바꾸어도 결과값을 동일하다

$$
\sum_i C_{ui} \times y_i \times (y_i^Tx_u) + \lambda x_u = \sum_i C_{ui}(p_{ui})  \times y_i
$$

여기서 $$x_u$$를 밖으로 빼주고 항을 정리하면 다음과 같은 형태가 나온다. I는 단위행렬(Identitiy Matrix)이다.

$$
(\sum_{i} C_{ui} \times y_i \times y_i^T + \lambda I)x_u = \sum_i C_{ui}(p_{ui}) \times y_i
$$

먼저 좌변부터 정리해보겠다. 모든 아이템 갯수만큼 벡터와 스칼라 곱을 수행하는 이 식은 하나의 행렬 곱으로 나타낼 수 있다.

먼저 원본 Y행렬과 그 전치 행렬을 곱해보았다.

$$
Y \times Y^T = [y_1y_1^T + y_2y_2^T + \dots + y_iy_i^T]
$$

그 다음 각 $$i$$에 맞는 신뢰도를 곱해주기 위해서 $$C_{ui}$$를 diagonal matrix로 만든 다음 그 사이에 곱해주면 된다.

$$
Y = [y_1y_2\dots y_i]
$$

$$
\begin{align}
C = 
\begin{bmatrix}
c_{u1} & 0 & 0 & 0 \\
0 & c_{u1} & 0 & 0 \\
0 & 0 & \dots & 0 \\
0 & 0 & 0 & c_{u1} \\
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
Y^T =
\begin{bmatrix}
y_1^T \\
y_2^T \\
\dots \\
y_i^T
\end{bmatrix}
\end{align}
$$

$$
Y \times C \times Y^T = [c_{u1}y_1y_1^T + c_{u2}y_2y_2^T + \dots + c_{ui}y_iy_i^T]
$$

우변 역시 비슷한 방식으로 정리하면 다음과 같은 결과를 얻을 수 있다.

$$
(YC_uY^T + \lambda I)x_u = Y^TC_up(u) 
$$

이제 좌변에 $$X_u$$만 남기기 위해서 양변에 같은 역행렬을 곱해주면 다음과 같은 식이 나온다.

$$
x_u = (YC^uY^T + \lambda I)^{-1}Y^TC^up(u) \\
    = (Y^TC^uY + \lambda I)^{-1}Y^TC^up(u)
$$

$$YC_uY^T$$와 $$Y^TC_uY$$는 결국 계산하면 1X1 크기의 스칼라 값이므로 이 둘은 같다. 우리는 Loss의 미분 값을 0으로 만드는 $$X_u$$ 행렬을 찾아낸 것이다.

이제 X를 위 식을 통해 계산한 다음 업데이트를 해준다. 다음으로 사용자 행렬 X를 고정시키고, 아이템에 대하여 같은 작업을 반복하여 같은 형태의 수식을 얻을 수 있다.

$$
y_i = (X^TCX + \lambda I)^{-1}X^TC^ip(i)
$$

이를 통해 Y행렬을 계산하여 업데이트 해준다. 이과정을 반복하면서 최적의 X와 Y를 찾아내는 것이 ALS 알고리즘의 핵심이다. 보통 반복하는 횟수를 10회에서 15회 정도로 설정합니다. 하지만 데이터의 양이나 Sparse 정도에 따라서 조절해나가는 것이 바람직하다.


### 유사도 계산

사용자 특성 벡터(평점 행렬의 행 벡터)이나 상품 특성 벡터(평점 행렬의 열 벡터)의 유사도(similarity)을 비교하기 위한 기준
- 평균제곱차이 유사도(Mean Squred Difference Similarity)
- 코사인 유사도(Cosine similarity)
- 피어슨 유사도(Pearson similarity)
- 피어슨-베이스라인 유사도(Pearson-Baseline Similarity)

#### 평균제곱차이 유사도 Mean Squared Difference, MSD

유클리드 공간에서의 거리제곱에 비례하는 값이다. msd값을 구하고 그 역수로 유사도를 정의한다. msd값이 0이 되는 경우를 대비하여 1을 더한다. 
$$
\begin{split}\text{msd_sim}(u, v) &= \frac{1}{\text{msd}(u, v) + 1}\\
\text{msd_sim}(i, j) &= \frac{1}{\text{msd}(i, j) + 1}\end{split}
$$

사용자 $$u$$ 와 사용자 $$v$$ 간의 msd 

- $$I_{uv}$$ 는 사용자 $$u$$ 와 사용자 $$v$$ 모두에 의해 평가된 상품의 집합이고 |$$I_{uv}$$| 는 사용자 $$u$$ 와 사용자 $$v$$ 모두에 의해 평가된 상품의 수이다.

$$
\text{msd}(u, v) = \frac{1}{|I_{uv}|} \cdot \sum\limits_{i \in I_{uv}} (r(u,i) - r_(v,i))^2
$$

상품 $$i$$  와 상품 $$j$$ 간의 msd

- $$U_{ij}$$ 는 상품 $$i$$ 와 상품 $$j$$ 모두를 평가한 사용자의 집합이고 |$$U_{ij}$$| 는 상품 $$i$$ 와 상품 $$j$$ 모두를 평가한 사용자의 수

$$
\text{msd}(i, j) = \frac{1}{|U_{ij}|} \cdot \sum\limits_{u \in U_{ij}} (r(u,i) - r(u,j))^2
$$

#### 코사인 유사도 Cosine Similarity

두 특성 벡터의 각도에 대한 코사인 값을 말한다. 벡터 $$x$$ 와 벡터 $$y$$ 사이의 각도 $$\theta$$ 는 두 벡터의 내적 $$x \cdot y$$ 와 다음과 같은 관계가 있다. 각도 $$\theta$$ 가 0도이면 코사인 유사도는 1이다. 반대로 각도 $$\theta$$ 가 90도이면 코사인 유사도는 0이다.

$$
x \cdot y = |x| |y| \cos\theta \\
\cos\theta = \dfrac{x \cdot y}{|x| |y|}
$$

사용자 $$u$$ 와 사용자 $$v$$ 간의 코사인 유사도

$$
\text{cosine_sim}(u, v) = \frac{
\sum\limits_{i \in I_{uv}} r(u,i) \cdot r(v,i)}
{\sqrt{\sum\limits_{i \in I_{uv}} r(u,i)^2} \cdot
\sqrt{\sum\limits_{i \in I_{uv}} r(v,i)^2}
}
$$

- 상품 $$i$$  , 상품$$j$$ 간의 코사인 유사도

$$
\text{cosine_sim}(i, j) = \frac{
\sum\limits_{u \in U_{ij}} r(u,i) \cdot r(u,j)}
{\sqrt{\sum\limits_{u \in U_{ij}} r(u,i)^2} \cdot
\sqrt{\sum\limits_{u \in U_{ij}} r(u,j)^2}
}
$$

#### 피어슨 유사도 Pearson Similarity

두 벡터의 상관계수(Pearson correlation coefficient)를 말하면 다음과 같이 정의한다. 

사용자 $$u$$ 와 사용자 $$v$$ 간의 피어슨 유사도
    
$$
\text{pearson_sim}(u, v) = \frac{
\sum\limits_{i \in I_{uv}} (r(u,i) -  \mu(u)) \cdot (r(v,i) - \mu(v))}
{\sqrt{\sum\limits_{i \in I_{uv}} (r(u,i) -  \mu(u))^2} \cdot
\sqrt{\sum\limits_{i \in I_{uv}} (r(v,i) -  \mu(v))^2}
}
$$

상품 $$i$$  , 상품$$j$$ 간의 피어슨 유사도

- $$\mu(i)$$ 는 상품 $$i$$ 의 평균 평점이다. 

$$
\text{pearson_sim}(i, j) = \frac{
\sum\limits_{u \in U_{ij}} (r(u,i) -  \mu(i)) \cdot (r(u,j) - \mu(j))}
{\sqrt{\sum\limits_{u \in U_{ij}} (r(u,i) -  \mu(i))^2} \cdot
\sqrt{\sum\limits_{u \in U_{ij}} (r(u,j) -  \mu(j))^2}
}
$$

상관계수는 가장 높은 경우의 값이 1이고 무상관인 경우는 0이다.

#### 피어슨-베이스라인 유사도 Pearson-Baseline Similarity

피어슨 유사도와 같이 상관계수를 구하지만 각 벡터의 기댓값을 단순 평균이 아니라 베이스라인 모형에서 예측한 값을 사용한다. 

사용자 $$u$$ 와 사용자 $$v$$ 간의 피어슨 베이스라인 유사도

$$
\text{pearson_baseline_sim}(u, v) = \hat{\rho}_{uv} = \frac{
\sum\limits_{i \in I_{uv}} (r(u,i) -  b(u,i)) \cdot (r(v,i) - b(v,i))}
{\sqrt{\sum\limits_{i \in I_{uv}} (r(u,i) -  b(u,i))^2} \cdot
\sqrt{\sum\limits_{i \in I_{uv}} (r(v,i) -  b(v,i))^2}}
$$

상품 $$i$$  , 상품$$j$$ 간의 피어슨 베이스라인 유사도
    
$$
\text{pearson_baseline_sim}(i, j) = \hat{\rho}_{ij} = \frac{
\sum\limits_{u \in U_{ij}} (r(u,i) -  b(u,i)) \cdot (r(u,j) - b(u,j))}
{\sqrt{\sum\limits_{u \in U_{ij}} (r(u,i) -  b(u,i))^2} \cdot
\sqrt{\sum\limits_{u \in U_{ij}} (r(u,j) -  b(u,j))^2}}
$$

피어슨-베이스라인 유사도는 벡터의 차원 즉, 두 사용자나 상품에 공통적으로 있는 평점 원소의 갯수를 이용하여 정규화를 하는 shrinkage를 추가하여 사용한다.

$$
\begin{split}\text{pearson_baseline_shrunk_sim}(u, v) &= \frac{|I_{uv}| - 1}
{|I_{uv}| - 1 + \text{shrinkage}} \cdot \hat{\rho}_{uv}\end{split} \\
\begin{split}\text{pearson_baseline_shrunk_sim}(i, j) &= \frac{|U_{ij}| - 1}
{|U_{ij}| - 1 + \text{shrinkage}} \cdot \hat{\rho}_{ij}\end{split}
$$

### KNN 가중치 예측 방법

일단 유사도가 구해지면 평점을 예측하고자 하는 사용자(또는 상품)과 유사도가 큰 k개의 사용자(또는 상품)벡터를 사용하여 가중 평균을 구해서 가중치를 예측한다. 이러한 방법을 KNN(K Nearest Neighbors) 기반 예측 방법이라고 한다.

- `KNNBasic` : 평점들을 단순히 가중평균한다. 다음 식에서 $$N^k$$ 는 $$k$$ 개의 가장 유사도가 큰 벡터의 집합니다. 
$$
\hat{r}(u,i) = \frac{
            \sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot r(v,i)}
            {\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)}
$$
    
또는
    
$$
\hat{r}(u,i) = \frac{
            \sum\limits_{j \in N^k_u(i)} \text{sim}(i, j) \cdot r(u,j)}
            {\sum\limits_{j \in N^k_u(j)} \text{sim}(i, j)}
$$

- `KNNWithMeans` : 평점들을 평균값 기준으로 가중 평균한다. 

$$
\hat{r}(u,i) = \mu(u) + 
\frac{\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot (r(v,i) - \mu(v))}{\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)}
$$

또는
  
$$
\hat{r}(u,i) = \mu(i) + 
\frac{\sum\limits_{j \in N^k_u(i)}\text{sim}(i, j) \cdot (r(u,j) - \mu(j))} {\sum\limits_{j \in N^k_u(i)} \text{sim}(i, j)}
$$

-  `KNNBaseline` : 평점들을 베이스라인 모형의 값 기준으로 가중 평균한다. 

$$
\hat{r}(u,i) = b(u,i) + \frac{ \sum\limits_{v \in N^k_i(u)}
\text{sim}(u, v) \cdot (r(v,i) - b(v,i))} {\sum\limits_{v \in
N^k_i(u)} \text{sim}(u, v)}
$$

또는

$$
\hat{r}(u,i) = b(u,i) + \frac{ \sum\limits_{j \in N^k_u(i)}
\text{sim}(i, j) \cdot (r(u,j) - b(u,j))} {\sum\limits_{j \in
N^k_u(j)} \text{sim}(i, j)}
$$

### NMF (Non-negative matrix factorization)

행렬 V가 보통 두 개의 행렬 W와 H로 분해되는 선형대수와 다변량 분석의 알고리즘 중에 하나로, 여기서 등장하는 세 개의 행렬들은 음수의 원소를 가지지 않는 특징이 있다. 이러한 음수가 아닌 성질은 결과 행렬들이 더욱 쉽게 조사될 수 있도록 하고, 또한 데이터 속에 이미 음수가 아닌 성질을 내재되어 있을 때, 이것을 쉽게 분해 또는 분석할 수 있다. 
  
$$
V = WH
$$

행렬의 곱은 V의 열벡터들은 W에서의 열벡터들의 선형결합으로 계산될 수 있고, 여기서 H의 열에 있는 값들을 선형 결합의 계수로 사용하면 된다. 즉, V의 각 열은 아래와 같이 계산된다.

$$
v_i = Wh_i
$$

$$v_i$$ 라는 것은 V행렬의 i번째 열벡터이고, $$h_i$$ 는 행렬 H의 i번째 열벡터이다. 행렬을 곱할 때에는, 요인 행렬의 차원은 곱하는 행렬의 차원 보다 상당히 적은 값을 가진다. 그리고 이것이 NMF의 기저를 형성하는 특성이 된다. NMF는 본래의 관심이 대상이 되는 행렬과 비교하여 훨씬 줄어든 차원과 함께 요인들을 생성할 수 있다. 예를 들어, V는 mxn 행렬이고, W는 mxp 행렬이고, H는 pxn 행렬이라고 하자. 그러면 p는 상당히 m과 n보다 작을 수 있는 것이다.

### Contents Based Filtering

사용자 혹은 아이템에 대한 프로필 데이터를 가지고 내가 좋아했던 아이템과 비슷한 유형의 아이템을 추천하거나 나와 비슷한 유형의 사람이 좋아하는 아이템을 추천한다. 각각의 사용자와 아이템에 대하여 프로필을 작성하고, 이를 기반으로 추천을 한다.

영화 추천을 예시로 들어보면, 먼저 사용자에 대해 성별, 연령대, 지역 등의 프로필을 작성할 수 있다. 나와 프로필이 비슷한 다른 사용자가 선호하는 영화를 추천해주는 방식(user-based recommendation)
  - 영화는 장르, 출연하는 배우들, 흥행 여부 등의 프로필을 작성할 수 있다. 이를 기반으로 특정 영화를 좋아했던 사용자에게 비슷한 영화를 추천해주는 방식(item-based recommendation)

하지만 이런 방식은 데이터 셋을 구성하기가 매우 어려운 한계가 있다.
  - 영화를 예시로 들어보면 수백만편의 영화에 대하여 일일이 프로필을 작성하는 것은 많은 시간과 비용이 들어간다. 또한 프로필 데이터 작성 시에 주관성이 개입하는 문제가 있다. 어떤 영화가 폭력적이거나, 로맨틱하다라는 정보는 개인의 주관에 영향을 받습니다. 따라서 이러한 정보를 사람이 일일이 입력하여 프로필 데이터를 구성하게 되면, 추천의 객관성이 떨어지게 된다.

![image-20200311221808191](../../../resource/img/image-20200311221808191.png)


### Hybrid Filtering (Contents based Filtering + Collaborative Filtering)

Contents Based Filtering과 Collaborative Filter을 모두 적용하고, 이의 가중평균(Weighted Average)를 구하는 Combining Filtering 기법이다. 평점데이터와 아이템 프로필을 조합하여 사용자 프로필을 만들어 추천하는 Collaboration via Content 기법이다.


### 추천을 위한 연관성 규칙(Association Rules)

####  지지도,  신뢰도, 향상도

어떤 사용자의 장바구니(구매 리스트)를 기반으로 분석하는 것을 장바구니 분석이라고 하며, 이 장바구니 분석에서 사용하는 대표적인 것이 지지도, 신뢰도, 향상도이다.

예로, 어떤 사람이 맥주를 고르고, 그 이후 오징어를 구입했다고 가정해보자. 그럼 우리는 사람들은 맥주를 마시는 사람은 오징어를 고를 수 있다고 판단을 하게 된다. 그래서 맥주 주변에 오징어 상품을 배열하여, 나도 모르게 오징어까지 구입하게 만드는 것이 추천이다.

##### 신뢰도(Confidence)

![image-20200311222614272](../../../resource/img/image-20200311222614272.png)

맥주를 구매할 때, 어떤 상품을 추천하는 것이 옳을까?  장바구니에서 맥주의 출현 개수 중, 같이 구입한 상품의 개수를 판단하면 된다. 

맥주를 기준으로 추천을 해야 되는 상품을 나열하자면 아래와 같을 것이다.

  오징어(2), 치즈(1), 소주(1), 라면(1), 사이다(1), 콜라(1)

다시 정렬하면

  2회 : 오징어

  1회 : 치즈, 소주, 라면, 사이다, 콜라

**신뢰도 = A상품에 대한 B상품의 신뢰도 = A상품과 B상품의 동시출현 횟수 / A상품의 출현 횟수**

- 2회(66%) : 오징어

- 1회(33%) : 치즈, 소주, 라면, 사이다, 콜라

신뢰도를 뽑는 이유는 특별한 기준을 잡기 위해서이다.  예를 들어, 50% 이상의 신뢰도만 의미가 있다라고 가정을 준다면, 위 데이터에서는 오징어를 제외한 치즈, 소주, 라면, 사이다, 콜라는 무의미한 데이터가 된다.

##### 지지도(Support)

지지도를 구하는 목적은, 우리가 구한 추천 데이터가 과연 올바른가? 라는 의문에서 시작을 한다.

예를 들어, 1000개의 물품이 있다. 총 1만번의 거래가 있었는데, 이때 A라는 상품은 단 한번밖에 거래가 안됐다고 가정을 해보자. 이때, A상품에 대한 데이터는 과연 효율성이 있을까?*라는 질문에서 탄생을 한다. 지지도를 구하는 목적은 위에 적은 것처럼, 추천할 만한 데이터 분석이다 라는 것을 판단하기 위해서 사용한다.

**A와 B상품의 지지도 = A상품과 B상품을 같이 구입한 횟수 / 전체 구매 횟수**
- 1만번의 거래가 있다고 가정할 때, A상품과 B상품이 같이 구입한 횟수는 2번이라면, 2 / 10000 => 0.0002 라는 수치로 환산이 된다. 여기서, 신뢰도는 0.0001 이상만 가져온다라고 한다면, 표본이 적은 위와 같은 사례는 의미가 없는 데이터가 된다. 이렇게 필터링을 위해서, 지지도를 구하게 된다.
- 맥주의 신뢰도는 위에서 구한 것과 같이 오징어가 66%의 신뢰도를 보여주었다. 그럼 지지도는 몇프로일까?
- 신뢰도는 "**동시 출현 개수 / 기준 상품 리스트**" 이었기 때문에 "**2 / 3**" 이라는 값이 나오게 되었다면,
- 지지도는 "**동시 출현 개수 / 전체 구매리스트**" 이기 때문에 "**2 / 5**"가 된다.

즉, 맥주와 오징어의 지지도는 40%가 된다. 신뢰도보다 뚝 떨어진 수치이다. 위 공식대로 보자면, 지지도는 절대 신뢰도보다 높을 수 없다. 같거나 낮을 수 밖에 없는 수치이다.

##### 리프트(Lift) = 향상도(Improvement)

향상도란, 품목간의 의미를 파악하기 위해서 사용이 된다.

예를 들어 A라는 상품에서 신뢰도가 동일한 상품 B와 C가 존재할 때, 우리는 어떤 상품을 더 추천해야 좋을지를 판단한다.
- A와 B = 40%, A와 C = 40% 라고 가정을 할 때, A가 총 5번 등장했다면, B와 C는 A와 동시에 2번 등장했음을 알 수 있을 것이다. 전체 구매리스트는 총 50개 이고, 여기서 B는 총 5번 등장했고, C는 총 10번 등장했다면, 어떤 상품을 추천해야 옳을 것인가?
- 좀 더 쉽게 설명하자면, 위 데이터만 봤을 때, C의 등장은 B보다 2배 더 높으므로, 2배 더 높은 인기(상대적적 인기) 상품이라는 것을 알 수 있다.
- 향상도의 공식은 $$L = \dfrac {P(Y|X)} {P(Y))} = \dfrac {P(X \cap Y)} {P(X)P(Y)} $$

위와 같은데, 위 공식대로 나온 값이 아래와 같이 뜻하고 있다.

- 1 : 두 품목이 서로 **독립**적인 관계
- \>1 : 두 품목이 서로 **양**의 상관 관계
- <1 : 두 품목이 서로 **음**의 상관 관계

위 공식인 독립적, 양, 음이라는 수치보다는 일반적인 현업에서는 Confidence가 동일할 경우, Lift 값을 2차 정렬로 사용한다던지, Lift 값을 정렬 후, Confidence를 정렬하는 경우가 있는데 즉, 위와 같은 케이스에서는 B를 추천하는 것이 더 옳을 것이다.

그럼 B가 C보다 Lift가 높은지 계산을 수행해보자. 
- A와 B 상품의 향상도 계산 : A와 B의 신뢰도 / (B상품 등장 횟수 / 전체 리스트) = 0.40 / (5/50) = 4
- A와 C 상품의 향상도 계산 : A와 C의 신뢰도 / (C상품 등장 횟수 / 전체 리스트) = 0.40 / (10/50) = 2 가 되어서, A를 구입할 때, C보다 B를 상품추천하는 것이 더 옳다라는 것을 알 수 있을 것이다.
- A상품을 대상으로 추천되는 상품이 C상품 하나밖에 없는데 이 상품이 음의 상관관계이면, 추천하지 말아야 하나?? 라는 질문을 할 수 있다. 사실 데이터 마이닝을 할 때, 수많은 삽질과 예상치 못한 고객들의 반응을 볼 수 있다. 이론적으로는 이건데 왜 이렇게 나오지? 등등.... 공식이 하나로만 정리되면 편하련만 사이트 및 데이터의 특성에 따라 너무나도 예상치 못한 값들이 나오기 마련이다. 어떤 가설을 세워서 사이트를 구축하였는데 늘 문제가 있을 수 있다라는 마인드로 대비책을 세워야 한다. 정말 Lift가 낮은 것이 추천률이 높아질 수도 있다. Case by Case이고, 이럴 경우를 대비하여 사용자의 특성까지도 고려해야 한다. 특정 사용자 집단은 어떤 구매 패턴등이 있다. 그렇기에 지지도 신뢰도 향상도는 하나의 추천 시스템 지수이며, 사이트, 그리고 데이터 별로 여러가지의 추천 방식을 사용해야 성공적인 프로젝트 수행을 할 수 있을 것이다.

Reference
- https://rfriend.tistory.com/191
- https://yeomko.tistory.com/3
- https://datascienceschool.net/ (데이터사이언티스트스쿨)