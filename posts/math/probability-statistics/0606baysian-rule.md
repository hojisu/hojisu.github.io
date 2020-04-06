<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 베이즈 정리 Baysian Rule

### Summary

- 베이즈 정리는 사건 B가 발생함으로써 사건 A의 확률이 어떻게 변화하는지를 표현한 정리다. 베이즈 정리는 새로운 정보가 기존의 추론에 어떻게 영향을 미치는지를 나타내고 있다. 새로운 정보가 기존의 추론에 어떻게 영향을 미치는지를 나타내고 있다.
- 베이즈 추론(baysian inference)은 관측되지 않은 데이터에 대해 알고 있는 정보로 추론 대상을 밝히는 것입니다.
- MAP(maximum a posterior)는 확률을 확률모형으로(사후확률) 모델을 한 후 이 확률모형을 최적화 하는 것입니다. $$\theta$$ 의 분포를 알 수 있기 때문에 분산을 알수 있고 그래서 신뢰도를 알 수 있습니다. $$\theta$$ 를 하나의 값으로 생각하지 않고 $$\theta$$ 에 대한 사전 정보(믿음)을 가지고 모수 $$\theta$$ 에 대해 추정할 수 있다. 베이즈 정리를 이용하여 사후 분포에서 $$\theta$$ 에 대해 극대화 하는 것입니다. 
$$\arg max_{\theta} \ P(\theta|D) = arg \ max_{\theta} \ \theta^{x+\alpha-1}(1-\theta)^{N-x-\beta-1}$$ 
- 분류 문제를 풀기 위해서는 각각의 출력 카테고리 $$𝑌$$ 에 대한 특징값 $$𝑋$$의 분포, 즉 likelihood를 알고 있어야 한다. 이렇게 베이즈 정리와 likelihood를 이용하여 각각의 $$𝑌$$ 값에 대한 확률 값을 모두 구한 다음, 가장 확률 값이 높은 $$𝑌$$값을 선택하여 분류 문제를 푸는 방법을 생성론적 방법(generative method)라고 한다.

_____________

### 베이즈 정리

**베이즈 정리(Bayesian rule)** 는 사건 B가 발생함으로써 사건 A의 확률이 어떻게 변화하는지를 표현한 정리다. 

$$
P(A|B) = \dfrac{P(B|A)P(A)}{P(B)} 
$$

-(증명)

$$
P(A|B) = \dfrac{P(A,B)}{P(B)} \;\; \rightarrow \;\; P(A,B) = P(A|B)P(B) \\
P(B|A) = \dfrac{P(A,B)}{P(A)} \;\; \rightarrow \;\; P(A,B) = P(B|A)P(A) \\
P(A,B) = P(A|B)P(B) = P(B|A)P(A) \\
P(A|B) = \dfrac{P(B|A)P(A)}{P(B)}
$$

##### 베이즈 정리 

$$
P(A|B) = \dfrac{P(B|A)P(A)}{P(B)}
$$

- $$𝑃(𝐴|𝐵)$$: 사후확률(posterior). 사건 B가 발생한 후 갱신된 사건 A의 확률
- $$𝑃(𝐴)$$: 사전확률(prior). 사건 B가 발생하기 전에 가지고 있던 사건 A의 확률
- $$𝑃(𝐵|𝐴)$$: 가능도(likelihood). 사건 A가 발생한 경우 사건 B의 확률
- $$𝑃(𝐵)$$: 정규화 상수(normalizing constant): 확률의 크기 조정

베이즈 정리는 사건 $$𝐵$$가 발생함으로서(사건 $$𝐵$$가 진실이라는 것을 알게 됨으로서 즉 사건 $$𝐵$$의 확률 $$𝑃(𝐵)=1$$이라는 것을 알게 됨으로서) 사건 $$𝐴$$의 확률이 어떻게 변화하는지를 표현한 정리이다. 따라서 베이즈 정리는 **새로운 정보가 기존의 추론에 어떻게 영향을 미치는지를 나타내고 있다.**

### 베이즈 정리의 확장 1

만약 사건 $$A_i$$ 가 다음의 조건을 만족하는 경우, 

- 서로 교집합이 없다
- 모두 합쳤을 때(합집합) 전체 표본 공간이면 $$A_1 \cup A_2 \cup \cdots = \Omega$$
- 전체 확률의 법칙을 이용하여 다음과 같이 베이즈 정리를 확장할 수 있다.

$$
\begin{align}
\begin{aligned}
P(A_1|B) 
&= \dfrac{P(B|A_1)P(A_1)}{P(B)} \\
&= \dfrac{P(B|A_1)P(A_1)}{\sum_i P(A_i, B)} \\
&= \dfrac{P(B|A_1)P(A_1)}{\sum_i P(B|A_i)P(A_i)} 
\end{aligned}
\end{align}
$$

$$A_1 = A, A_2 = A^C$$ 인 경우에는 다음과 같은 식이 성립한다.

$$
\begin{align}
\begin{aligned}
P(A|B) 
&= \dfrac{P(B|A)P(A)}{P(B)} \\
&= \dfrac{P(B|A)P(A)}{P(B,A) + P(B,A^C)} \\
&= \dfrac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^C)P(A^C)} \\
&= \dfrac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^C)(1 - P(A))} 
\end{aligned}
\end{align}
$$

### pgmpy을 사용한 베이즈 정리 적용

pgmpy 패키지에서는 베이즈 정리를 적용할 수 있는 `BayesianModel` 클래스를 제공한다. 베이즈 정리를 적용하려면 조건부확률을 구현하기 위한 클래스 `TabularCPD`클래스를 사용하여 사전확률과 가능도를 구현해야 한다. `TabularCPD` 클래스는 다음과 같이 만든다.

```python
TabularCPD(variable, variable_card, value, evidence=None, evidence_card=None)
```

- `variable`: 확률변수의 이름 문자열
- `variable_card`: 확률변수가 가질 수 있는 경우의 수
- `value`: 조건부확률 배열. 하나의 열(column)이 동일 조건을 뜻하므로 하나의 열의 확률 합은 1이어야 한다.
- `evidence`: 조건이 되는 확률변수의 이름 문자열의 리스트
- `evidence_card`: 조건이 되는 확률변수가 가질 수 있는 경우의 수의 리스트

`TabularCPD` 클래스는 원래는 조건부확률을 구현하기 위한 것이지만 `evidence=None`, `evidence_card=None`으로 인수를 주면 일반적인 확률도 구현할 수 있다.

예시 

우선 확률변수 X를 이용하여 병에 걸렸을 사전확률 $$𝑃(𝐷)=𝑃(𝑋=1)$$, 병에 걸리지 않았을 사전확률 $$𝑃(𝐷^𝐶)=𝑃(𝑋=0)$$를 정의한다.

~~~python
from pgmpy.factors.discrete import TabularCPD

cpd_X = TabularCPD('X', 2, [[1 - 0.002, 0.002]])
print(cpd_X)
#결과
+-----+-------+
| X_0 | 0.998 |
+-----+-------+
| X_1 | 0.002 |
+-----+-------+
~~~

다음으로는 양성 반응이 나올 확률 $$𝑃(𝑆)=𝑃(𝑌=1)$$, 음성 반응이 나올 확률 $$𝑃(𝑆^𝐶)=𝑃(𝑌=0)$$를 나타내는 확률변수 $$𝑌$$를 정의한다.

확률변수 $$𝑌$$의 확률을 베이즈 모형에 넣을 때는 `TabularCPD` 클래스를 사용한 조건부확률 $$𝑃(𝑌|𝑋)$$의 형태로 넣어야 하므로 다음처럼 조건부확률 $$𝑃(𝑌|𝑋)$$를 구현한다.

~~~python
cpd_Y_on_X = TabularCPD('Y', 2, np.array([[0.95, 0.01], [0.05, 0.99]]),
                        evidence=['X'], evidence_card=[2])
print(cpd_Y_on_X)
#결과
-----+------+------+
| X   | X_0  | X_1  |
+-----+------+------+
| Y_0 | 0.95 | 0.01 |
+-----+------+------+
| Y_1 | 0.05 | 0.99 |
+-----+------+------+
~~~

이제 이 확률변수들이 어떻게 결합되어 있는지는 나타내는 확률모형인 `BayesianModel`클래스 객체를 만들어야 한다.

```py
BayesianModel(variables)
```

`variables`: 확률모형이 포함하는 확률변수 이름 문자열의 리스트

`BayesianModel` 클래스는 다음 메서드를 지원한다.

`add_cpds`: 조건부확률을 추가

`check_model`: 모형이 정상적인지 확인. `True`면 정상적인 모형 

~~~python
from pgmpy.models import BayesianModel
 
model = BayesianModel([('X', 'Y')])
model.add_cpds(cpd_X, cpd_Y_on_X)
  model.check_model()

  #결과 True
~~~

`BayesianModel` 클래스는 변수 제거법(VariableElimination)을 사용한 추정을 제공한다. `VariableElimination` 클래스로 추정(inference) 객체를 만들고 이 객체의 `query` 메서드를 사용하면 사후확률을 계산한다.

```
  query(variables, evidences)
```

`variables`: 사후확률을 계산할 확률변수의 이름 리스트

`evidences`: 조건이 되는 확률변수의 값을 나타내는 딕셔너리

~~~python
from pgmpy.inference import VariableElimination

inference = VariableElimination(model)
posterior = inference.query(['X'], evidence={'Y': 1})
print(posterior['X'])
#결과
+-----+----------+
| X   |   phi(X) |
+=====+==========+
| X_0 |   0.9618 |
+-----+----------+
| X_1 |   0.0382 |
+-----+----------+
~~~

### 베이즈 정리의 확장 2

베이즈 정리는 사건 $$𝐴$$의 확률이 사건 $$𝐵$$에 의해 갱신(update)된 확률을 계산한다. 그런데 만약 이 상태에서 또 추가적인 사건 $$𝐶$$가 발생했다면 베이즈 정리는 다음과 같이 쓸 수 있다.
$$
P(A|B,C) = \dfrac{P(C|A,B)P(A|B)}{P(C|B)} 
$$

위 식에서 $$𝑃(𝐴|𝐵,𝐶)$$는 $$𝐵$$와 $$𝐶$$가 조건인 $$𝐴$$의 확률이다. 즉 $$𝑃(𝐴|(𝐵∩𝐶))$$를 뜻한다.

(증명)
$$
P(A,B,C) = P(A|B,C)P(B,C) = P(A|B,C)P(C|B)P(B) \\
P(A,B,C) = P(C|A,B)P(A,B) = P(C|A,B)P(A|B)P(B) \\
P(A|B,C)P(C|B)P(B) = P(C|A,B)P(A|B)P(B) \\
P(A|B,C) = \dfrac{P(C|A,B)P(A|B)}{P(C|B)}
$$

### 베이즈 정리와 분류 문제

베이즈 정리는 머신 러닝 중 분류(classification) 문제를 해결하는데 사용될 수 있다. 분류 문제는 입력 자료 $$𝑋$$의 값으로부터 카테고리 값인 출력 자료 $$𝑌$$의 값을 예측(prediction)하는 문제이다. 다음과 같은 문제는 분류 문제의 한 예이다.

분류 문제를 풀기 위해서는 각각의 출력 카테고리 $$𝑌$$ 에 대한 특징값 $$𝑋$$의 분포, 즉 likelihood를 알고 있어야 한다. 이렇게 베이즈 정리와 likelihood를 이용하여 각각의 $$𝑌$$ 값에 대한 확률 값을 모두 구한 다음, 가장 확률 값이 높은 $$𝑌$$값을 선택하여 분류 문제를 푸는 방법을 생성론적 방법(generative method)라고 한다.

