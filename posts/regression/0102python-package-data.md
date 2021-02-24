<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 파이썬 패키지와 데이터

statsmodels 패키지 (스탯츠모델즈 패키지라고 읽는다.)

scikit-learn 패키지 (싸이킷-런 패키지라고 읽는다.)

### statsmodels 패키지

검정 및 추정 (test and estimation)

회귀분석 (regression analysis)

시계열 분석 (time-seies analysis)

~~~python
import statsmodels.api as sm
~~~

### scikit-learn 패키지

~~~python
import sklearn as sk
~~~

### 가상데이터

`make_regression()` : 회귀분석 결과를 검증하기 위해 가상의 데이터가 필요한 경우 사용

~~~python
X, y = make_regression(n_samples, n_features, bias, noise, random_state)
~~~

~~~python
X, y, w = make_regression(... coef=True)
~~~

`n_samples`: 정수 (옵션, 디폴트 100)

표본 데이터의 갯수 𝑁N

`n_features` : 정수 (옵션, 디폴트 100)

독립 변수(feature)의 수(차원) 𝑀M

`bias`: 실수 (옵션, 디폴트 0.0)

y 절편

`noise`: 실수 (옵션, 디폴트 0.0)

출력 즉, 종속 변수에 더해지는 잡음 𝜖ϵ의 표준편차

`random_state`: 정수 (옵션, 디폴트 None)

난수 발생용 시드값

`coef`: 불리언 (옵션, 디폴트 False)

True 이면 선형 모형의 계수도 출력

출력은 다음과 같다

`X`: [`n_samples`, `n_features`] 형상의 2차원 배열

독립 변수의 표본 데이터 행렬 X

`y`: [`n_samples`] 형상의 1차원 배열

종속 변수의 표본 데이터 벡터 𝑦y

`coef`: [`n_features`] 형상의 1차원 배열 또는 [`n_features`, `n_targets`] 형상의 2차원 배열 (옵션)

선형 모형의 계수 벡터 𝑤, 입력 인수 `coef`가 True 인 경우에만 출력됨

#### `make_regression( )` 명령은 내부적으로 다음 과정을 거쳐 가상의 데이터를 만든다.

1. 독립변수 데이터 행렬 `X`를 무작위로 만든다.
2. 종속변수와 독립변수를 연결하는 가중치 벡터 `w`를 무작위로 만든다.
3. `X`와 `w`를 내적하고 y절편 `b` 값을 더하여 독립변수와 완전선형인 종속변수 벡터 `y_0`를 만든다.
4. 기댓값이 0이고 표준편차가 `noise`인 정규분포를 이용하여 잡음 `epsilon`를 만든다.
5. 독립변수와 완전선형인 종속변수 벡터 `y_0`에 잡음 `epsilon`을 더해서 종속변수 데이터 𝑦를 만든다.

$$
y = w^Tx + b + \epsilon
$$

#### `make_regression( )` 함수 구현 (coef=True, n_features=1)

~~~python
coef=True, n_features=1
def make_regression(n_sample, bias, noise, random_state=None):
	np.random.seed(random_state)
	n_sample = np.random.rand(n_sample)
	w = np.random.rand()*100
	epsilon = np.random.randn(n_sample)*noise
	y_0 = (n_sample * w) + bias + epsilon
	return n_sample, y_0, w
~~~

`n_features` 즉, 독립 변수가 2개인 표본 데이터를 생성하여 스캐터 플롯을 그리면 다음과 같다. 종속 변수 값은 점의 명암으로 표시하였다.

`make_regression` 명령은 위에서 설명한 인수 이외에도 다음과 같은 인수를 가질 수 있다.

`n_informative`: 정수 (옵션, 디폴트 10)

독립 변수(feature) 중 실제로 종속 변수와 상관 관계가 있는 독립 변수의 수(차원)

`effective_rank`: 정수 또는 None (옵션, 디폴트 None)

독립 변수(feature) 중 서로 독립인 독립 변수의 수. 만약 None이면 모두 독립

`tail_strength`: 0부터 1사이의 실수 (옵션, 디폴트 0.5)

`effective_rank`가 None이 아닌 경우 독립 변수간의 상관관계를 결정하는 변수. 0.5면 독립 변수간의 상관관계가 없다.



________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 