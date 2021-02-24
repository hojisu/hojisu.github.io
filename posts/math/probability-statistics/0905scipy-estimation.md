<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 기초적인 검정

### Summary

- 이항검정은 두 가지 값을 가지는 확률변수의 분포를 판단하는데 도움을 준다.
- 카이제곱 검정은 어떤 범주형 확률변수 𝑋가 다른 범주형 확률변수 𝑌와 독립인지 상관관계를 가지는가를 검증하는데도 사용할 수 있다. 독립변수들이 무수히 많을 때 종속변수와 상관 관계가 있는 변수들을 선택할 때 사용할 수 있다. 
- 단일 표본 z-검정(One-sample z-test)은 분산 $$\sigma^2$$ 의 값을 정확히 알고 있는 정규 분포의 표본에 대해 기댓값을 조사하는 검정방법이다.
- 귀무가설이 진실임에도 불구하고 거짓으로 나온 경우 유형 1오류라고 한다. 귀무가설이 거짓임에도 불구하고 진실로 나온 경우 유형 2오류라고 한다.
- 단일 표본 t-검정(One-sample t-test)은 정규 분포의 표본에 대해 기댓값을 조사하는 검정방법이다. 
- 독립 표본 t-검정(Independent-two-sample t-test)은 두 개의 독립적인 정규 분포에서 나온 $$N_1, N_2$$ 개의 데이터 셋을 사용하여 두 정규 분포의 기댓값이 동일한지를 검사한다
- 등분산 검정(Equal-variance test)은 두 정규 분포로부터 생성된 두 개의 데이터 집합으로부터 두 정규 분포의 분산 모수가 같은지 확인하기 위한 검정이다.
____________

SciPy 파이썬 패키지는 다음과 같은 다양한 검정 명령을 제공한다.

- 이항 검정 (Binomial test)
- 카이제곱 검정 (Chi-square test)
- 단일 표본 z-검정 (One-sample z-test)
- 단일 표본 t-검정 (One-sample t-test)
- 독립 표본 t-검정 (Independent-two-sample t-test)
- 대응 표본 t-검정 (Paired-two-sample t-test)
- 분산 검정 (Chi squared variance test)
- 등분산 검정 (Equal-variance test)
- 정규성 검정 (Normality test)

### 이항 검정

이항 검정은 이항 분포를 이용하여 베르누이 확률변수의 모수 𝜇에 대한 가설을 조사하는 검정 방법이다. 

SciPy stats 서브패키지의 `binom_test` 명령은 이항 검정의 유의확률을 계산한다. 디폴트 귀무 가설은 𝜇=0.5이다.

```
scipy.stats.binom_test(x, n=None, p=0.5, alternative='two-sided')
```

- `x`: 검정통계량. 1이 나온 횟수
- `n`: 총 시도 횟수
- `p`: 귀무가설의 𝜇μ 값
- `alternative`: 양측검정인 경우에는 `'two-sided'`, 단측검정인 경우에는 `'one-sided'`

이항 검정은 **두 가지 값을 가지는 확률변수의 분포를 판단**하는데 도움을 준다.

### 카이제곱 검정

카데고리 분포의 모수에 대한 검정에 사용된다. 통계량은 스칼라가 아닌 벡터값을 가지기 때문이다. 

카이제곱 검정은 범주형 확률 분포의 모수 $$𝜇=(𝜇_1,…,𝜇_𝐾)$$에 대한 가설을 조사하는 검정 방법으로 **적합도 검정(goodness of fit test)**이라고도 부른다.

원래 범주형 값 $$𝑘$$가 나와야 할 횟수의 기댓값 $$𝑚_𝑘$$와 실제 나온 횟수 $$𝑥_𝑘$$의 차이를 이용하여 다음처럼 검정 통계량을 구한다.
$$
\sum_{k=1}^K \dfrac{(x_k - m_k)^2}{m_k}
$$

SciPy stats 서브패키지의 `chisquare` 명령은 카이제곱 검정의 검정 통계량과 유의확률을 계산한다. 디폴트 귀무 가설은 $$\mu = \left(\frac{1}{K}, \ldots, \frac{1}{K} \right)$$이다.

**카이제곱 검정은 어떤 범주형 확률변수 𝑋가 다른 범주형 확률변수 𝑌와 독립인지 상관관계를 가지는가를 검증**하는데도 사용할 수 있다.

독립변수들이 무수히 많을 때 종속변수와 상관 관계가 있는 변수들을 선택할 때 사용할 수 있다. 

SciPy의 `chi2_contingency` 명령은 이러한 검정을 수행한다. X의 값에 따른 각각의 Y의 분포가 2차원 표(contingency table)의 형태로 주어지면 y분포의 평균 분포와 실제 y분포의 차이를 검정 통계량으로 계산한다. 이 값이 충분히 크다면 X와 Y는 상관관계가 있다. `chi2_contingency` 명령의 결과는 튜플로 반환되며 첫번째 값이 검정 통계량, **두번째 값이 유의확률이다.**

**귀무가설은 두 표본은 독립니다.**이다.

### 단일 표본 z-검정

**단일 표본 z-검정(One-sample z-test)**은 분산 $$\sigma^2$$ 의 값을 정확히 알고 있는 정규 분포의 표본에 대해 기댓값을 조사하는 검정방법이다.

귀무 가설이 진실임에도 불구하고 거짓으로 나온 경우로 **유형 1 오류(Type 1 Error)**라고 한다. **유의확률은 유형 1 오류가 나올 확률을 말한다.**

### 단일 표본 t-검정

**단일 표본 t-검정(One-sample t-test)**은 정규 분포의 표본에 대해 기댓값을 조사하는 검정방법이다. 

$$
\dfrac{\bar{x} - \mu_0}{\dfrac{s}{\sqrt{N}}}
$$

SciPy의 stats 서브 패키지의 `ttest_1samp` 명령을 사용한다. `ttest_1samp` 명령의 경우에는 디폴트 모수가 없으므로 기댓값을 나타내는 `popmean` 인수를 직접 지정해야 한다.

### 독립 표본 t-검정

**독립 표본 t-검정(Independent-two-sample t-test)** 은 두 개의 독립적인 정규 분포에서 나온 $$N_1, N_2$$ 개의 데이터 셋을 사용하여 두 정규 분포의 기댓값이 동일한지를 검사한다.

검정통계량으로는 두 정규 분포의 분산이 같은경우 ;

$$
t = \dfrac{\bar{x}_1 - \bar{x}_2}{s \cdot \sqrt{\dfrac{1}{N_1}+\dfrac{1}{N_2}}}
$$

$$\bar{x}_1, \bar{x}_2$$는 각각의 표본평균이고 표본표준편차 $$s$$는 각각의 표본분산 $$s_1^2, s_2^2$$로부터 다음처럼 구한다. 이 통계량은 자유도가 $$N_1+N_2−2$$인 스튜던트-t 분포를 따른다.

$$
s = \sqrt{\dfrac{\left(N_1-1\right)s_{1}^2+\left(N_2-1\right)s_{2}^2}{N_1+N_2-2}}
$$

두 정규 분포의 분산이 다른 경우 :

$$
t = \dfrac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{N_1} + \dfrac{s_2^2}{N_2}}}
$$

자유도가 다음과 같은 스튜던트-t 분포를 따른다.

$$
\dfrac{\left(\dfrac{s_1^2}{N_1} + \dfrac{s_2^2}{N_2}\right)^2}{\dfrac{\left(s_1^2/N_1\right)^2}{N_1-1} + \dfrac{\left(s_2^2/N_2\right)^2}{N_2-1}}
$$

독립 표본 t-검정은 SciPy stats 서브패키지의 `ttest_ind` 명령을 사용하여 계산한다. 독립 표본 t-검정은 두 정규 분포의 분산값이 같은 경우와 같지 않은 경우에 사용하는 검정 통계량이 다르기 때문에 `equal_var` 인수를 사용하여 이를 지정해 주어야 한다. 두 분포의 분산이 같은지 다른지는 다음에 나올 등분산 검정(equal-variance test)을 사용하면 된다. 만약 잘 모르겠으면 `equal_var=False`로 놓으면 된다.

귀무 가설이 거짓임에도 불구하고 진실로 나온 경우로 **유형 2 오류(Type 2 Error)**라고 한다. 데이터 수가 증가하면 이러한 오류가 발생할 가능성이 줄어든다.

### 대응 표본 t-검정

**대응 표본 t-검정(Paired-two-sample t-test)** 은 독립 표본 t-검정을 두 집단의 표본이 1대1 대응하는 경우에 대해 수정한 것이다. 즉, 독립 표본 t-검정과 마찬가지로 두 정규 분포의 기댓값이 같은지 확인하기 위한 검정이다.

통계량은 대응하는 표본 값의 차이 $$x_d=x_{i,i}−x_{i,2}$$에서 다음처럼 계산한다.
$$
t = \dfrac{\bar{x}_d - \mu_0}{\dfrac{s_d}{\sqrt{N}}}
$$

대응 표본 t-검정은 `ttest_rel` 명령을 사용한다.

### 카이제곱 분산 검정

**카이제곱 분산 검정(Chi-Squara Test for the Variance)** 은 정규 분포의 표본 분산 값은 정규화 하면 카이제곱 분포를 따른다는 점을 이용하는 검정 방법이다.

$$
\chi^2=(N-1)\dfrac{s^2}{\sigma^2_0}
$$

SciPy는 카이제곱 분산 검정에 대한 명령이 없으므로 `chi2` 클래스를 사용하여 직접 구현해야 한다.

### 등분산 검정

등분산 검정(Equal-variance test)은 두 정규 분포로부터 생성된 두 개의 데이터 집합으로부터 두 정규 분포의 분산 모수가 같은지 확인하기 위한 검정이다.

가장 기본적인 검정통계량은 F분포가 되는 표본분산의 비율을 사용하는 것이다.

$$
F=\dfrac{s_1^2}{s_2^2}
$$

실제로는 이보다 더 복잡한 통계량을 이용하는 bartlett, fligner, levene 방법을 주로 사용한다. SciPy의 stats 서브패키지는 이를 위한 `bartlett`, `fligner`, `levene` 명령을 제공한다.

### 정규성 검정

**정규성 검정(normality test)** 은 **확률 분포가 가우시안 정규 분포를 따르는지 아닌지를 확인**하는 것이다.

#### SciPy 에서 제공하는 정규성 검정 명령어

- Kolmogorov-Smirnov test

  - ```
    scipy.stats.ks_2samp
    ```

- Shapiro–Wilk test

  - ```
    scipy.stats.shapiro
    ```

- Anderson–Darling test

  - ```
    scipy.stats.anderson
    ```

- D'Agostino's K-squared test

  - `scipy.stats.mstats.normaltest`

#### StatsModels에서 제공하는 정규성 검정 명령어

- Omnibus Normality test

  - ```
    statsmodels.stats.stattools.omni_normtest
    ```

- Jarque–Bera test

  - ```
    statsmodels.stats.stattools.jarque_bera
    ```

- Kolmogorov-Smirnov test

  - ```
    statsmodels.stats.diagnostic.kstest_normal
    ```

- Lilliefors test

  - ```
    statsmodels.stats.diagnostic.lillifors
    ```

___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다.