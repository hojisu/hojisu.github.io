<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# LP 문제와 QP문제

### Summary

- LP문제는 방정식이나 부등식 제한 조건을 가지는 선형모형의 값을 최소화하는 문제이다
- QP문제는 방정식이나 부등식 제한 조건을 가지는 일반화된 이차형식(quadratic form)의 값을 최소화 하는 문제이다.

___________

### Linear Programming 문제

방정식이나 부등식 제한 조건을 가지는 선형 모형(linear model)의 값을 최소화하는 문제를 **LP(Linear Programming) 문제**라고 한다. 

LP문제의목적함수는 $$\arg\min_x c^Tx$$ 이고 선형 연립방정식으로 된 등식 제한조건 $$Ax = b$$ 과 변수값이 모두 음수가 아니어야하는 부등식 제한조건 $$x \geq 0$$ 을(벡터 $$x$$ 의 모든 원소가 양수거나 0이 되어야 한다는 것을 의미) 동시에 가진다

위와 같은 형태를 LP 문제의 기본형(standard form)이라고 한다. 

표준형을 확장한 정규형(canonical form) LP문제는 부등식 조건을 허용한다.

$$
\arg\min_x c^Tx \\
Ax \leq b \\
x \geq 0
$$

예제로 어떤 공장에서 두가지 제품을 생산해야 한다고 하자.

- 제품 A와 제품 B 각각 100개 이상 생산해야 한다.
- 시간은 500시간 밖에 없다.
- 제품 A는 생산하는데 1시간이 걸리고 제품 B는 2시간이 걸린다.
- 특정 부품이 9800개밖에 없다.
- 제품 A는 생산하는데 특정 부품을 4개 필요로 하고 제품 B는 생산하는데 특정 부품을 5개 필요로 한다.
- 제품 A의 이익은 하나당 3만원이고 제품 B의 이익은 하나당 5만원이다.

제품 A와 제품 B의 생산량을 각각 $$𝑥_1,𝑥_2$$라고 하면 최소화(- 붙임)하려는 목적함수는
$$
-3x_1 -5x_2
$$


제한조건 (부호 바꿔주려고 $$-x_1, -x_2, -100, -100$$함)

$$
\begin{aligned}
-x_1 & & &\leq& -100 \\
 & & -x_2 &\leq& -100 \\
x_1 &+& 2 x_2 &\leq& 500 \\
4x_1 &+& 5 x_2 &\leq& 9800 \\
\end{aligned}
$$
정규형 LP문제로 표현하면 아래와 같다.
$$
\begin{align}
\min_x 
\begin{bmatrix} -3 & -5 \end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\begin{bmatrix}
-1 & 0 \\
0 & -1 \\
1 & 2 \\
4 & 5 \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 
\end{bmatrix} \leq
\begin{bmatrix}
-100 \\ -100 \\ 500 \\ 9800
\end{bmatrix}
\end{align}
$$

$$
\begin{align}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix}\geq
\begin{bmatrix}
0 \\ 0
\end{bmatrix}
\end{align}
$$

### SciPy를 이용한 LP 문제 계산

scipy.optimize 패키지의 `linprog` 명령을 사용하면 LP 문제를 풀 수 있다. 사용법은 다음과 같다.

~~~python
linprog(c, A, b)
~~~

`c`: 목적함수의 계수 벡터

`A`: 등식 제한조건의 계수 행렬

`b`: 등식 제한조건의 상수 벡터

다음 코드는 위 예제 LP 문제를 SciPy로 계산하는 코드이다.

```python
import scipy.optimize

A = np.array([[-1, 0], [0, -1], [1, 2], [4, 5]])
b = np.array([-100, -100, 500, 9800])
c = np.array([-3, -5])

result = sp.optimize.linprog(c, A, b)
result
```

```python
#결과
     con: array([], dtype=float64)
     fun: -1400.0
 message: 'Optimization terminated successfully.'
     nit: 3
   slack: array([ 200.,    0.,    0., 8100.])
  status: 0
 success: True
       x: array([300., 100.])
```

제품 A를 300개, 제품 B를 100개 생산할 때 이익이 1400으로 최대가 됨을 알 수 있다.

### Quadratic Programming 문제

방정식이나 부등식 제한 조건을 가지는 일반화된 이차형식(quadratic form)의 값을 최소화 하는 문제를 QP(Quadraitx Programming) 문제라고 한다.

QP문제의 목적함수는 $$\arg\min_x \dfrac{1}{2}x^TQx + c^Tx$$ 이다.

등식 제한조건과 부호 제한조건은 LP문제와 같다.
$$
Ax =b \\
x \geq 0
$$

잔차 제곱합(잔차 벡터의 각 원소를 제곱한 후 더한)을 최소화하는 예측 모형에 추가적인 제한조건이 있으면 QP 문제가 된다.

등식 제한조건이 있는 최적화 문제도 QP 문제이다.

$$
\arg\min_x x_1^2 + x_2^2 \\
x_1 + x_2 - 1 = 0
$$
이 문제를 QP 형식으로 바꾸면 다음과 같다
$$
\arg\min_x
\dfrac{1}{2}
\begin{bmatrix}
x_1 & x_2
\end{bmatrix}
\begin{bmatrix}
2 & 0 \\ 0 & 2
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix}
+ 
\begin{bmatrix}
0 & 0
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2
\end{bmatrix} = 1
$$

### CvxOpt를 이용한 LP 문제 계산

- CvxOpt라는 패키지를 사용하면 QP문제를 풀 수 있다. cvxopt를 쓸 때는 numpy 배열을 cvxopt 전용의 matrix 자료형으로 바꿔야 한다. 또 정수 자료형을 사용하지 못하므로 항상 부동소수점 실수가 되도록 명시해야 한다.

~~~python
from cvxopt import matrix, solvers

Q = matrix(np.diag([2.0, 2.0]))
c = matrix(np.array([0.0, 0.0]))
A = matrix(np.array([[1.0, 1.0]]))
b = matrix(np.array([[1.0]]))

sol = solvers.qp(Q, c, A=A, b=b)
np.array(sol['x'])
~~~

