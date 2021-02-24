<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# Categorical distribution (카테고리 분포) & Multinomial distribution (다항 분포)

### Summary

- 카테고리분포는 1부터 K까지의 K개의 정수값 중 하나가 나오는 확률변수의 분포이다. $$\text{Cat}(x_1, x_2, \dots, x_K;\mu_1, \dots, \mu_K), \text{Cat}(x;\mu) = \mu_1^{x_1} \mu_2^{x_2}  \cdots \mu_K^{x_K}  =  \prod_{k=1}^K \mu_k^{x_k}$$ 
- 다항 분포(Multinomial distribution)는 독립적인 카테고리 분포를 여러번 시도하여 얻은 각 원소의 성공횟수 값의 분포이다. $$\text{Mu}(x;N,\mu) = \binom N x  \prod_{k=1}^K \mu_k^{x_k} = \binom N {x_1, \cdots, x_K} \prod_{k=1}^K \mu_k^{x_k}$$

___________________

## Categorical distribution(카테고리 분포)

### 카테고리 분포

카테고리 분포(categorical distribution)는 1부터 K까지의 $$K$$ 개의 정수 값 중 하나가 나오는 확률변수의 분포이다. 

주의할 점은 카테고리 분포를 가진 확률변수는 원래 카테고리인 스칼라 값을 출력하는 확률변수지만 다음과 같이 1과 0으로만 이루어진 다차원 벡터로 변형하여 사용한다. 이러한 인코딩 방식을 **원-핫-인코딩(One-Hot-Encoding)** 이라고 한다

$$
\begin{eqnarray}
x = 1  \;\; & \rightarrow & \;\; x = (1, 0, 0, 0, 0, 0) \\
x = 2  \;\; & \rightarrow & \;\; x = (0, 1, 0, 0, 0, 0) \\
x = 3  \;\; & \rightarrow & \;\; x = (0, 0, 1, 0, 0, 0) \\
x = 4  \;\; & \rightarrow & \;\; x = (0, 0, 0, 1, 0, 0) \\
x = 5  \;\; & \rightarrow & \;\; x = (0, 0, 0, 0, 1, 0) \\
x = 6  \;\; & \rightarrow & \;\; x = (0, 0, 0, 0, 0, 1) \\
\end{eqnarray}
$$

확률변수의 값 $$𝑥$$가  다음과 같은 벡터이다. $$x = (x_1, x_2, x_3, x_4, x_5, x_6)$$ 

이 벡터를 구성하는 원소 $$x_1, x_2, x_3, x_4, x_5, x_6$$ 에는 다음과 같은 제한 조건이 붙는다.

$$
x_i = \begin{cases} 0 \\ 1 \end{cases} \\
\sum_{k=1}^K x_k = 1
$$
첫번째 제한 조건은 $$𝑥_𝑘$$의 값이 0 또는 1 만 가능하다는 것이고 

두번째 제한 조건은 여러개의 $$𝑥_𝑘$$ 중 단 하나만 1일 수 있다는 것이다. 이 각각의 원소 값 $$𝑥_𝑘$$는 일종의 베르누이 확률분포로 볼 수 있기 때문에 각각 1이 나올 확률을 나타내는 모수 $$𝜇_𝑘$$를 가진다. 그리고 전체 카테고리 분포의 모수는 다음과 같이 벡터로 나타낸다.
$$
\mu = ( \mu_1, \cdots , \mu_K )
$$
모수 벡터 제한 조건

$$\mu_k$$ 는 0부터 1사이의 어떤 실수값이든 가질 수 있다.

$$
0 \leq \mu_i \leq 1 \\
\sum_{k=1}^K \mu_k = 1
$$

카테고리 분포는 $$\text{Cat}(x_1, x_2, \dots, x_K;\mu_1, \dots, \mu_K)$$   또는 출력 벡터 $$𝑥=(𝑥_1,𝑥_2,…,𝑥_𝐾)$$, 모수 벡터 $$𝜇=(𝜇_1,…,𝜇_𝐾)$$를 사용하여 $$\text{Cat}(x;\mu)$$ 로 나타낼 수 있다.

카테고리 분포 확률질량함수 표기하면 아래와 같다. 

$$
\text{Cat}(x;\mu) = 
\begin{cases}
\mu_1 & \text{if } x = (1, 0, 0, \cdots, 0) \\
\mu_2 & \text{if } x = (0, 1, 0, \cdots, 0) \\
\mu_3 & \text{if } x = (0, 0, 1, \cdots, 0) \\
\vdots & \vdots \\
\mu_K & \text{if } x = (0, 0, 0, \cdots, 1) \\
\end{cases}
$$

원핫인코딩 사용 표현하면 표기가 간략해진다.

$$
\text{Cat}(x;\mu) = \mu_1^{x_1} \mu_2^{x_2}  \cdots \mu_K^{x_K}  =  \prod_{k=1}^K \mu_k^{x_k}
$$

### 카테고리 분포의 모멘트

카테고리 분포는 표본값이 벡터이므로 기댓값과 분산도 벡터이다

#### 기댓값

$$
\text{E}[x_k] = \mu_k
$$

#### 분산

$$
\text{Var}[x_k] = \mu_k(1-\mu_k)
$$

#### SciPy를 이용한 카테고리 분포의 시뮬레이션

SciPy는 카테고리 분포를 위한 별도의 클래스를 제공하지 않는다. 하지만 뒤에서 설명할 다항 분포를 위한 `multinomial` 클래스에서 시행 횟수를 1로 설정하면 카테고리 분포가 되므로 이 명령을 사용할 수 있다.

```python
mu = np.array([1/6]*6)
rv = sp.stats.multinomial(1, mu)
```

```python
# 카테고리 분포에서 나올 수 있는 값은 다음처럼 벡터값들
xx = np.arange(1, 7)
xx_ohe = pd.get_dummies(xx)
xx_ohe
```

```python
# pmf 메서드의 인수로도 벡터를 넣어야 한다.
plt.bar(xx, rv.pmf(xx_ohe.values))
plt.ylabel("P(x)")
plt.xlabel("표본값")
plt.title("카테고리 분포의 pmf")
plt.show()
```

시뮬레이션

```python
np.random.seed(1)
X = rv.rvs(100)
X[:5]
```

```python
# 시뮬레이션 결과
y = X.sum(axis=0) / float(len(X))
plt.bar(np.arange(1, 7), y)
plt.title("카테고리 분포의 시뮬레이션 결과")
plt.xlabel("표본값")
plt.ylabel("비율")
plt.show()
```

이론적인 확률분포와 시뮬레이션 결과를 비교

```python
df = pd.DataFrame({"이론": rv.pmf(xx_ohe.values), "시뮬레이션": y},
                  index=np.arange(1, 7)).stack()
df = df.reset_index()
df.columns = ["표본값", "유형", "비율"]
df.pivot("표본값", "유형", "비율")
sns.barplot(x="표본값", y="비율", hue="유형", data=df)
plt.title("카테고리 분포의 이론적 분포와 시뮬레이션 분포")
plt.show()
```

## Multinomial distribution (다항 분포)

### 다항 분포

다항 분포(Multinomial distribution)는 독립적인 카테고리 분포를 여러번 시도하여 얻은 각 원소의 성공횟수 값의 분포이다. 

카테고리 시도를 $$N$$ 번 반복하여 $$k(k = 1, ….., k)$$ 가 각각 $$x_k$$ 번 나올 확률분포 즉, 표본값이 벡터 $$x = (x_1, ….  x_k)$$ 가 되는 확률분포이다. 

다항 분포의 확률질량함수 수식
$$
\text{Mu}(x;N,\mu) = \binom N x  \prod_{k=1}^K \mu_k^{x_k} = \binom N {x_1, \cdots, x_K} \prod_{k=1}^K \mu_k^{x_k}
$$

$$
\binom N {x_1, \cdots, x_K} = \dfrac{N!}{x_1! \cdots x_K!}
$$

### 다항 분포의 모멘트

#### 기댓값

$$
\text{E}[x_k] = N\mu_k
$$

#### 분산

$$
\text{Var}[x_k] = N\mu_k(1-\mu_k)
$$

#### SciPy 를 이용한 다항 분포의 시뮬레이션

SciPy는 다항 분포를 위한 `multinomial` 클래스를 지원한다. 인수로는 시행 횟수 $$𝑁$$과 모수 벡터 $$𝜇$$를 받는다. 

```python
N = 30
mu = [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]
rv = sp.stats.multinomial(N, mu)

# 시뮬레이션 결과는  𝐾 개의 박스 플롯(box plot)으로 표시할 수 있다.
np.random.seed(0)
X = rv.rvs(100)
X[:5]
plt.boxplot(X)
plt.title("다항 분포의 시뮬레이션 결과")
plt.xlabel("class")
plt.ylabel("binomial")
plt.show()
```

seaborn 패키지를 사용하면 보다 다양하게 시각화할 수 있다.

```python
df = pd.DataFrame(X).stack().reset_index()
df.columns = ["trial", "class", "binomial"]

# boxplot 사용
sns.boxplot(x="class", y="binomial", data=df)
sns.stripplot(x="class", y="binomial", data=df, jitter=True, color=".3")
plt.title("다항 분포의 시뮬레이션 결과")
plt.show()
```

```python
# violinplot 사용
sns.violinplot(x="class", y="binomial", data=df, inner="quartile")
sns.swarmplot(x="class", y="binomial", data=df, color=".3")
plt.title("다항 분포의 시뮬레이션 결과")
plt.show()
```

다항분포의 표본 하나는 같은 모수를 가진 카테고리분포를 표본 여럿을 합쳐놓은 것이므로 다항분포의 표본이 이루는 분포의 모양은 카테고리분포와 비슷해진다. 


___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다.