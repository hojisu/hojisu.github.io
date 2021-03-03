<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 분류 모형

### Summary
- 분류모형에는 확률적 모형인 확률적 판별모형, 확률적 생성모형과 판별함수 모형이 있다. 확률적 판별모형은 직접 조건부확률 함수의 모양을 계산하는 것으로 로지스틱회귀, 의사결정나무가 있다. 확률적 판별모형에는 가능도를 구하고 베이즈정리를 사용하여 조건부확률을 구하는 모형으로 LDA, QDA, 나이브베이지안이 있다. 판별함수모형은 주어진 데이터를 카테고리에 따라 서로 다른 영역으로 나누는 경계면을 찾아낸 다음 이 경계면으로부터 주어진 데이터가 어느 위치에 속하는지 계산하는 판별함수 모형을 이용하는 모형으로 퍼셉트론, SVM이 있다.
- 다중클래스 분류 방법으로는 OvO와 OvR이 있다. OvO는 k개의 클래스가 존재할 경우, 이 중 2개의 클래스 조합을 선택하여 이진분류문제를 풀고 각 클래스가 얻은 조건부 확률값을 모두 더한 값을 비교하여 가장 큰 조건부 확률 총합을 가진 클래스를 선택한다. 단점은 클래스가 많을수록 이진 분류문제의 수가 너무 많아진다. OvR은 각각의 클래스에 대해 표본이 속하는지 안속하는지의 이진 분류문제를 푼다. 클래스 수만큼의 이진 분류문제를 풀면 된다. 각 클래스가 얻은 조건부 확률값을 더해서 이 값이 가장 큰 클래스를 선택한다.
__________________________________________________


### 분류 모형의 종류

분류(classification) 문제는 독립변수값이 주어졌을 때 그 값과 가장 연관성이 큰 종속변수값(클래스)을 예측하는 문제이다. 

분류 모형(분류문제를 푸는 방법)

**확률적 모형**은 주어진 데이터에 대해(conditionally) 각 카테고리 혹은 클래스가 정답일 조건부확률(conditional probability) 를 계산하는 모형이다.
- **확률적 판별(discriminative)모형**은 직접 조건부확률 함수의 모양을 계산하는 모형이다. 
    - 로지스틱 회귀, 의사결정나무
- **확률적 생성(generative) 모형** : likelihood $$p(x \mid y)$$를 구하고 베이즈 정리를 사용하여 간접적으로 조건부확률을 구하는 모형이다. 장점은 클래스가 3개 이상인 경우에도 바로 적용할 수 있다.
    - LDA/QDA, 나이브 베이지안
- **판별함수 모형**은 주어진 데이터를 카테고리에 따라 서로 다른 영역으로 나누는 경계면(dicision boundary)을 찾아낸 다음 이 경계면으로부터 주어진 데이터가 어느 위치에 있는지를 계산하는 판별함수(discriminant function)를 이용하는 모형이다.
    - 퍼셉트론, 서포트벡터머신, 인공신경망

### 확률적 모형

확률적 모형은 출력변수 $$y$$ 가 $$K$$ 개의 클래스 $$1, \cdots, K$$ 중의 하나의 값을 가진다고 가정하자. 확률적 모형은 다음과 같은 순서로 $$x$$ 에 대한 클래스를 예측한다.

입력 $$x$$ 가 주어졌을 때, $$y$$ 가 클래스 $$k$$ 가 될 확률 $$P(y=k \mid x )$$ 을  모두 계산하고 

$$
\begin{eqnarray}
P_1 &=& P(y=1 \mid x ) \\
\vdots & & \vdots \\
P_K &=& P(y=K \mid x )\\
\end{eqnarray}
$$

이 중에서 가장 확률이 큰 클래스를 선택하는 방법

$$
\hat{y} = \arg\max_{k} P(y=k \mid x )
$$

사이킷런 패키지에서 조건부확률 $$P(y=k \mid x )$$을 사용하는 분류 모형들은 모두 `predict_proba` 메서드와 `predict_log_proba` 메서드를 지원한다. 이 메서드들은 독립변수 $$x$$가 주어지면 종속변수 $$y$$의 모든 카테고리 값에 대해 조건부확률 또는 조건부확률의 로그값을 계산한다.

#### 확률적 생성모형

확률적 생성모형은 각 클래스 별 특징 데이터의 확률분포 $$P(x \mid y=k)$$ 을 추정한 다음 베이즈 정리를 사용하여 $$P(y=k \mid x )$$를 계산하는 방법이다.

$$
P(y=k \mid x) = \dfrac{P(x \mid y=k)P(y=k)}{P(x)}
$$

전체 확률의 법칙을 이용하여 특징 데이터의 $$x$$ 의 무조건부 확률분포 $$P(x)$$ 를 구할 수 있다. 

$$
P(x) = \sum_{k=1}^K P(x \mid y=k)P(y=k)
$$

새로운 가상의 특징 데이터를 생성해내거나 특징 데이터만으로도 아웃라이어를 판단 할 수 있다.

클래스가 많을 경우 모든 클래스에 대해 $$P(x \mid y=k)$$ 를 추정하는 것은 많은 데이터를 필요로 할 뿐더러 최종적으로 사용하지도 않을 확률분포를 계산하는데 계산량을 너무 많이 필요로 한다는 단점이 있다.

##### QDA(Quadratic Discriminant Analysis)

QDA는 조건부 확률 기반 생성(generative) 모형의 하나이다. 

##### 나이브 베이지안(Naive Bayesian) 모형

사이킷런의 `TfidfVectorizer` 전처리기는 텍스트 데이터를 정해진 크기의 실수 벡터로 변환한다. `MultinomialNB` 모형은 나이브 베이즈 방법으로 분류문제를 예측한다. 이 두 클래스 객체는 `Pipeline`을 사용하여 하나의 모형으로 합쳐놓을 수 있다. `Pipeline`으로 합쳐진 모형은 일반적인 모형처럼 `fit`, `predict` 등의 메서드를 제공하며 내부의 전처리기의 메소드를 상황에 따라 적절히 호출한다.

~~~python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

news = fetch_20newsgroups(subset="all")
model = Pipeline([
    ('vect', TfidfVectorizer(stop_words="english")),
    ('nb', MultinomialNB()),
])
model.fit(news.data, news.target)

n = 1
x = news.data[n:n + 1]
y = model.predict(x)[0]
print(x[0])
print("=" * 80)
print("실제 클래스:", news.target_names[news.target[n]])
~~~


#### 확률적 판별 모형(probabilistic discriminative model)

확률적 판별 모형은 조건부확률 $$p(y = 1 \mid x)$$이 $$x$$에 대한 함수 $$f(x)$$로 표시될 수 있다고 가정하고 그 함수를 직접 찾아내는 방법이다. 

$$
p(y = k \mid x) = f(x)
$$

이 함수 $$f(x)$$ 는 0보다 같거나 크고 1보다 같거나 작다는 조건을 만족해야 한다. 

##### 로지스틱 회귀모형

로지스틱 회귀모형은 확률론적 판별 모형에 속한다. 

#### 판별함수 모형

판별함수 모형은 동일한 클래스가 모여 있는 영역과 그 영역을 나누는 경계면(boundary plane)을 정의한다. 경계면(boundary plane)은 경계면으로부터의 거리를 계산하는 $$f(x)$$형태의 함수인 **판별함수(discriminant function)**로 정의된다. 판별함수(discriminant function)의 값은 부호에 따라 클래스가 나뉘어 진다.

$$
판별 경계선 : f(x) = 0 \\
class 1 : f(x) > 0 \\
class 2 : f(x) < 0
$$

사이킷런 에서 판별함수 모형은 판별함수 값을 출력하는 `decision_function` 메서드를 제공한다.

##### 퍼셉트론(Perceptron)

퍼셉트론은 가장 단순한 판별함수 모형이다. 직선이 경계선(boundary line) 으로 데이터 영역을 나눈다. 데이터의 차원이 3차원이라면 경계면(boundary surface)을 가지게 되는데 이러한 경계면이나 경계선을 의사결정 하이퍼 플레인(decision hyperplane)이라고 한다. 

~~~python
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
iris = load_iris()
idx = np.in1d(iris.target, [0, 2])
X = iris.data[idx, 0:2]
y = iris.target[idx]

model = Perceptron(max_iter=100, eta0=0.1, random_state=1).fit(X, y)
XX_min, XX_max = X[:, 0].min() - 1, X[:, 0].max() + 1
YY_min, YY_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000),
                     np.linspace(YY_min, YY_max, 1000))
ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
plt.contour(XX, YY, ZZ, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', linewidth=1)

idx = [22, 36, 70, 80]
plt.scatter(X[idx, 0], X[idx, 1], c='r', s=100, alpha=0.5)
for i in idx:
    plt.annotate(i, xy=(X[i, 0], X[i, 1] + 0.1))
plt.grid(False)
plt.title("퍼셉트론의 판별영역")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
~~~

~~~python
plt.bar(range(len(idx)), model.decision_function(X[idx]))
plt.xticks(range(len(idx)), idx)
plt.gca().xaxis.grid(False)
plt.title("각 데이터의 판별함수 값")
plt.xlabel("표본 번호")
plt.ylabel("판별함수값 f(x)")
plt.show()
~~~

##### 커널 SVM (Kernel Support Vector Machine)

커널 SVM은 복잡한 형태의 경계선을 생성할 수도 있다.

~~~python
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

model = svm.NuSVC().fit(X, Y)
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=3)
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
idx = [0, 20, 40, 60]
plt.scatter(X[idx, 0], X[idx, 1], c=Y[idx], s=200, alpha=0.5)
for i in idx:
    plt.annotate(i, xy=(X[i, 0], X[i, 1]+0.15), color='white')
plt.grid(False)
plt.axis([-3, 3, -3, 3])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("커널 SVM의 판별영역")
plt.show()
~~~

### 다중 클래스 분류

확률적 모형은 클래스가 3개 이상인 경우를 다중 클래스(Multi-Class) 분류문제도 풀 수 있지만 판별함수 모형은 종속변수의 클래스가 2개인 경우를 이진(Binary Class) 분류문제밖에는 풀지 못한다. 그래서 OvO(One-Vs-One) 방법이나 OvR(One-vs-the-Rest) 방법 등을 이용하여 여러개의 이젠 클래스 분류문제로 변환하여 푼다.

#### OvO(One-Vs-One)  방법

$$K$$ 개의 클래스가 존재하는 경우, 이 중 2개의 클래스 조합을 선택하여 $$K(K-1)/2$$ 개의 이진 분류문제를 풀어 가장 많은 결과가 나온 클래스를 선택하는 방법이다. 선택받은 횟수로 선택하면 횟수가 같은 경우도 나올 수 있기 때문에 각 클래스가 얻은 조건부 확률값을 모두 더한 값을 비교하여 가장 큰 조건부 확률 총합을 가진 클래스를 선택한다. 단점은 클래스의 수가 많아지면 실행해야 할 이진 분류문제의 수가 너무 많아진다.

#### OvR(One-vs-the-Rest) 방법

$$K$$ 개의 클래스가 존재하는 경우, 각각의 클래스에 대해 표본이 속하는지(y=1) 속하지 않는지(y=0) 의 이진 분류문제를 푼다. 클래스 수만큼의 이진 분류문제를 풀면 된다. 각 클래스가 얻은 조건부 확률값을 더해서 이 값이 가장 큰 클래스를 선택한다.


Reference
- 김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 