<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 고윳값 분해

### Summary

- 고유벡터는 선형변환을 해도 방향은 안바뀌고 크기만 바뀌는 벡터이고 고유값은 그 크기의 배수이다.
- 임의의 실수 행렬 X에 대해 $$X^TX$$(정방이면서 대칭행렬)인 정방행렬을 **공분산행렬(covariance matrix)**이라고 한다. 
_________________

### 고윳값과 고유벡터

정방 행렬 𝐴에 대해 다음 식을 만족하는 영벡터가 아닌 벡터 𝑣, 실수 𝜆를 찾을 수 있다고 가정하자.

$$
Av = \lambda v
$$

이 식은 다음 처럼 쓸수도 있다.

$$
Av - \lambda v = (A - \lambda I) v = 0
$$

위 식을 만족하는 실수 𝜆를 **고윳값**(eigenvalue), 벡터 𝑣 를 **고유벡터**(eigenvector)라고 한다.

어떤 **벡터를 행렬에 곱한** 결과로 나타난 **벡터**가 **원래의 벡터와 같은 방향을 향하고 있다**면 이를 **고유벡터**라고 한다. 다만 **크기나 방향은 원래의 벡터와 달라질 수가 있는데** **달라진 크기를 나타내는 비율**을 **고윳값**이라고 부른다.

고윳값과 고유벡터를 찾는 작업을 **고유분해(eigen-decomposition)** 또는 **고윳값 분해(eigenvalue decomposition)**라고 한다.

예시)

$$
A=
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
$$

다음 스칼라 값(𝜆)과 벡터(𝑣)는 각각 고윳값, 고유벡터가 된다.

$$
\lambda = -1,\;\; v=
\begin{bmatrix}
1  \\
1
\end{bmatrix}
$$

$$
Av = 
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
\begin{bmatrix}
1  \\
1
\end{bmatrix}
=
\begin{bmatrix}
-1 \\
-1
\end{bmatrix}
=
(-1)
\begin{bmatrix}
1 \\
1 
\end{bmatrix}
=
\lambda v
$$

어떤 **벡터 𝑣가 고유벡터**가 되면 이 **벡터에 실수를 곱한 벡터 𝑐𝑣**, 즉 **𝑣와 방향이 같은 벡터**는 **모두 고유벡터**가 된다. 예를 들어 행렬 𝐴에 대해 다음 벡터는 모두 고유벡터이다.

$$
v=
c
\begin{bmatrix}
1  \\
1
\end{bmatrix}
$$

$$
Av = 
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
\left(
c
\begin{bmatrix}
1  \\
1
\end{bmatrix}
\right)
=
c
\begin{bmatrix}
-1 \\
-1
\end{bmatrix}
=
(-1)c
\begin{bmatrix}
1 \\
1 
\end{bmatrix}
=
\lambda (cv)
$$

보통 **고유벡터를 표시**할 때는 **길이가 1인 단위벡터**가 되도록 다음처럼 **정규화(normalization)**를 한다.

$$
\dfrac{v}{\|v\|}
$$

위 행렬 𝐴의 고유값-고유벡터는 보통 다음처럼 나타내는 경우가 많다.

$$
\lambda = -1
$$

$$
v=
\begin{bmatrix}
\dfrac{\sqrt{2}}{2}  \\
\dfrac{\sqrt{2}}{2} 
\end{bmatrix}
\approx
\begin{bmatrix}
0.7071 \\
0.7071
\end{bmatrix}
$$

### 특성방정식

행렬만 주어졌을 때 고윳값-고유벡터 구하는 방법은 아래와 같다.

행렬 𝐴의 고유값은 𝐴−𝜆𝐼의 행렬식이 0이 되도록 하는 **특성방정식(characteristic equation)**의 해를 구하면 된다.

$$
\det \left( A - \lambda I \right) = 0
$$

이 조건은 **행렬 𝐴−𝜆𝐼가 역행렬이 존재하지 않는다**는 뜻이다. 만약 𝐴−𝜆𝐼의 역행렬이 존재한다면 아래의 식에서 고윳값 조건을 만족하는 벡터가 항상 영벡터가 되기 때문이다.

$$
Av - \lambda v = (A - \lambda I) v = 0  
$$

$$
(A - \lambda I)^{-1}(A - \lambda I)v = 0 \;\; \rightarrow \;\; v = 0
$$

예시 1) 해가 1개(사실은 중근으로 해가 2개) -> 고윳값은 -1(중근으로 -1, -1)

$$
A=
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
$$

$$
\begin{eqnarray}
\det \left( A - \lambda I \right) 
&=&
\det 
\left(
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
-
\begin{bmatrix}
\lambda & 0 \\
0 & \lambda
\end{bmatrix}
\right) 
\\
&=&
\det 
\begin{bmatrix}
1 - \lambda & -2 \\
2 & -3 -\lambda
\end{bmatrix}
\\
&=& (1 - \lambda)(-3 -\lambda) + 4 \\
&=& \lambda^2 + 2\lambda + 1 = 0
\end{eqnarray}
$$

인수분해를 하여 이차방정식인 특성방정식을 풀면

$$
\lambda^2 + 2\lambda + 1 = (\lambda + 1)^2 = 0
$$

에서 고윳값은 -1이다. (중근(해가 두개)이다. -1, -1 )

원래 이차방정식은 최대 2개의 해를 가질 수 있지만 이 경우에는 하나의 해만 존재하기 때문에 이러한 해를 **중복고윳값(repeated eigenvalue)**이라고 한다.

예시 2) 해가 2개 -> 고윳값 4, -1

$$
B=
\begin{bmatrix}
2 & 3 \\
2 & 1
\end{bmatrix}
$$

$$
\begin{eqnarray}
\det \left( B - \lambda I \right) 
&=&
\det 
\left(
\begin{bmatrix}
2 & 3 \\
2 & 1
\end{bmatrix}
-
\begin{bmatrix}
\lambda & 0 \\
0 & \lambda
\end{bmatrix}
\right) 
\\
&=&
\det 
\begin{bmatrix}
2 - \lambda & 3 \\
2 & 1 -\lambda
\end{bmatrix}
\\
&=& (2 - \lambda)(1 -\lambda) -6 \\
&=& \lambda^2 - 3\lambda -4 = 0
\end{eqnarray}
$$

$$
\lambda^2 - 3\lambda -4= (\lambda -4)(\lambda +1) = 0
$$

예시 3) 해가 존재하지 않는 경우 : 고윳값이 없는 행렬

$$
C =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

$$
\begin{eqnarray}
\det \left( C - \lambda I \right) 
&=&
\det 
\left(
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
-
\begin{bmatrix}
\lambda & 0 \\
0 & \lambda
\end{bmatrix}
\right) 
\\
&=& \lambda^2 +1 \\
&=& 0
\end{eqnarray}
$$

이 특성방정식의 **실수해는 존재하지 않음**을 알 수 있다. 따라서 행렬 𝐶는 **실수인 고유값을 가지지 않는다.**

만약 고유값-고유벡터가 **복소수(complex number)가 되어도 괜찮**다면 **행렬 𝐶는 2개의 고윳값**을 가진다고 할 수 있다. 

$$
\lambda = i, \;\; \lambda = -i,  \;\; i = 허수, \;\; i^2 = -1
$$


### 고윳값의 갯수

𝑁차방정식이 항상 𝑁개의 복소수 해를 가진다는 사실을 이용하면 𝑁차원 정방행렬의 고윳값의 갯수는 다음과 같음을 알 수 있다.

중복된 고유값을 하나로 생각하고 실수 고윳값만 고려한다면 𝑁차원 정방행렬의 고윳값은 0개부터 𝑁개까지 있을 수 있다.

**중복된 고윳값을 각각 별개로 생각하고 복소수인 고윳값도 고려한다면 𝑁차원 정방행렬의 고윳값은 항상 𝑁개이다.**


### 고윳값과 대각합/행렬식

대각합(trace) : 정방 행렬에 대해서만 정의되며 다음처럼 대각 원소의 합으로 계산된다.
$$
\operatorname{tr}(A) = a_{11} + a_{22} + \dots + a_{NN}=\sum_{i=1}^{N} a_{ii}
$$
행렬식(determinant) : 크기가 2x2인 정방행렬의 행렬식의 값 공식
$$
\det \left( \begin{bmatrix}a&b\\c&d\end{bmatrix} \right) = ad-bc
$$
예시) 행렬 A에 대한 대각합과 행렬식
$$
A=
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
$$

$$
\text{tr}(A) = 1 + (-3) = -2
$$

$$
\text{det}(A) = 1 \cdot (-3) - 2 \cdot (-2) = 1
$$

​        고윳값으로 계산 𝜆1=−1, 𝜆2=−1 (중복된 고윳값)
$$
\lambda_1 + \lambda_2 = -2 = \text{tr}(A)
$$

$$
\lambda_1 \times \lambda_2 = 1 = \text{det}(A)
$$

이 관계는 모든 행렬에 대해 성립한다.

어떤 행렬의 고윳값이 $$𝜆_1,𝜆_2,⋯,𝜆_𝑁$$이라고 하면 **모든 고윳값의 곱은 행렬식의 값**과 같고 **모든 고윳값의 합은 대각합(trace)의 값**과 같다.
$$
\det(A)=\prod_{i=1}^N \lambda_i
$$

$$
\text{tr}(A) =\sum_{i=1}^N \lambda_i
$$


### 고유벡터의 계산

고윳값을 알면 다음 연립 방정식을 풀어 고유벡터를 구할 수 있다.
$$
(A - \lambda I)v = 0
$$
예시) 행렬 A , 𝜆 = -1
$$
A=
\begin{bmatrix}
1 & -2 \\
2 & -3
\end{bmatrix}
$$

$$
\begin{bmatrix}
1+1 & -2 \\
2 & -3+1
\end{bmatrix}
\begin{bmatrix}
v_1 \\ v_2
\end{bmatrix}
= 0
$$

$$
\begin{bmatrix}
2 & -2 \\
2 & -2
\end{bmatrix}
\begin{bmatrix}
v_1 \\ v_2
\end{bmatrix}
= 0
$$

$$
2v_1 - 2v_2 = 0
$$

$$
v_1 = v_2
$$

위에 식을 만족하는 모든 벡터가 고유벡터임을 알 수 있다. 즉
$$
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$
또는 단위벡터
$$
\begin{bmatrix} \dfrac{\sqrt{2}}{2} \\ \dfrac{\sqrt{2}}{2} \end{bmatrix}
$$
가 유일한 고유벡터이다. 중복된(repeated) 고유벡터라고도 한다.



고윳값이 중복되었다고 고유벡터도 항상 중복되는 것은 아니다. 예를 들어 항등행렬 𝐼의 고윳값은 1로 중복된 고윳값을 가진다.

$$
\det(I - \lambda I)
=
\det( \left(
\begin{bmatrix}
1-\lambda & 0 \\
0 & 1-\lambda
\end{bmatrix}
\right)
=(\lambda - 1)^2 = 0
$$

하지만 이 값을 아래의 식에 대입하면

$$
Av - \lambda v = (A - \lambda I) v = 0 
$$

$$
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\ v_2
\end{bmatrix}
= 0
$$

으로 임의의 2차원 벡터는 모두 고유벡터가 된다. 즉

$$
\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \;\;
\begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$
둘다 고유벡터이다.

- 중복된 고윳값 𝜆에 대해 서로 다른 고유벡터 𝑣1, 𝑣2가 존재하면 이 두 벡터의 선형조합도 고윳값 𝜆에 대한 고유벡터임을 증명하라.
  $$
  c_1v_1 + c_2v_2
  $$

  $$
  Av_1 = \lambda c_1 \\Av_2 = \lambda c_2 \\ c_1Av_1 = \lambda v_1c_1 \\ c_2Av_2 = \lambda v_2c_2 
  \\
  c_1Av_1 + c_2Av_2 = \lambda v_1c_1 + \lambda v_2c_2 \\
  A(c_1v_1+c_2v_2) = \lambda (v_1c_1 + v_2c_2)
  \\
  c_1v_1 + c_2v_2
  $$



### Numpy를 사용한 고유분해

numpy.linalg 서브패키지에서는 고윳값과 고유벡터를 구할 수 있는 `eig` 명령을 제공한다. **고윳값은 벡터의 형태**로, **고유벡터는 고유벡터 행렬의 형태로 묶여서** 나오고 **고유벡터는 크기가 1인 단위벡터**로 정규화가 되어있다.

~~~python
A = np.array([[1, -2], [2, -3]])
w1, V1 = np.linalg.eig(A)

w1 
#결과 
# array([-0.99999998, -1.00000002])
V1
#결과
# array([[0.70710678, 0.70710678],
#        [0.70710678, 0.70710678]])
~~~

~~~python
B = np.array([[2, 3], [2, 1]])
w2, V2 = np.linalg.eig(B)

w2
#결과
# array([ 4., -1.])

v1
#결과
# array([[ 0.83205029, -0.70710678],
#       [ 0.5547002 ,  0.70710678]])
~~~

B행렬의 고윳값 4의 고유벡터는 0.83205029, 0.5547002 이다. **고유벡터의 짝은 열벡터**로 봐야된다.

예시) 실수인 고윳값이 존재하지 않는 행렬에 대해서는 복소수인 고윳값과 고유벡터를 계산한다.

$$
C =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

~~~python
C = np.array([[0, -1], [1, 0]])
W3, V3 = np.linalg.eig(C)

W3
# 결과 array([0.+1.j, 0.-1.j]) 
# j는 허수

V3
# 결과 array([[0.70710678+0.j        , 0.70710678-0.j        ],
#           [0.        -0.70710678j, 0.        +0.70710678j]])
# 복소수로 나타남
~~~


### 대각화

𝑁 차원의 정방 행렬 𝐴가 𝑁개의 복소수 고윳값과 이에 대응하는 고유벡터를 가진다는 성질을 이용하면 다음처럼 행렬을 **분해**할 수 있다.

행렬 𝐴의 고윳값과 이에 대응하는 단위벡터인 고유벡터를 각각
$$
\lambda_1, \lambda_2, \cdots, \lambda_N \;\;\; v_1, v_2, \cdots, v_N
$$
이 고윳값과 고유벡터를 묶어서 다음과 같이 고유벡터행렬, 고윳값행렬을 정의할 수 있다.

**고유벡터행렬** 𝑉은 고유벡터를 **열벡터로 옆으로 쌓아서 만든 행렬**이다.
$$
V = \left[ v_1 \cdots v_N \right], \;\;\; V \in \mathbf{R}^{N \times N}
$$
**고윳값행렬** Λ은 **고윳값을 대각성분**으로 가지는 **대각행렬**이다.
$$
\Lambda =
\begin{bmatrix}
\lambda_{1} & 0 & \cdots & 0 \\
0 & \lambda_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{N} \\
\end{bmatrix}
, \;\;\; \Lambda \in \mathbf{R}^{N \times N}
$$

**행렬과 고유벡터행렬의 곱은 고유벡터행렬과 고윳값행렬의 곱과 같다**.

$$
\begin{eqnarray}
AV 
&=& A \left[ v_1 \cdots v_N \right] \\
&=& \left[ A v_1 \cdots A v_N \right] \\
&=& \left[ \lambda_1 v_1 \cdots \lambda_N v_N \right] \\
&=& \left[ v_1 \cdots v_N \right] 
\begin{bmatrix}
\lambda_{1} & 0 & \cdots & 0 \\
0 & \lambda_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{N} \\
\end{bmatrix}
\\
&=& V\Lambda
\end{eqnarray}
$$

$$
AV = V\Lambda
$$

만약 **고유벡터행렬 𝑉의 역행렬이 존재한다면** 행렬을 다음처럼 **고유벡터행렬과 고윳값행렬의 곱으로 표현할 수 있다.** 이를 행렬의 **대각화(diagonalization)**라고 한다.

$$
A = V \Lambda V^{-1} 
$$

예시) 행렬 B를 대각화 한다면
$$
V = 
\begin{bmatrix}
\dfrac{3}{\sqrt{13}} & -\dfrac{1}{\sqrt{2}} \\
\dfrac{2}{\sqrt{13}} &  \dfrac{1}{\sqrt{2}}
\end{bmatrix}
$$

$$
\Lambda = 
\begin{bmatrix}
4 & 0 \\
0 & -1
\end{bmatrix}
$$

$$
V^{-1} =  \dfrac{1}{5} \begin{bmatrix} \sqrt{13} & \sqrt{13} \\ -2\sqrt{2} & 3\sqrt{2} \end{bmatrix}
$$

$$
B=
\begin{bmatrix}
2 & 3 \\
2 & 1
\end{bmatrix}
= 
V\Lambda V^{-1} 
= \dfrac{1}{5}
\begin{bmatrix}
\dfrac{3}{\sqrt{13}} & -\dfrac{1}{\sqrt{2}} \\
\dfrac{2}{\sqrt{13}} &  \dfrac{1}{\sqrt{2}}
\end{bmatrix}
\begin{bmatrix}
4 & 0 \\
0 & -1
\end{bmatrix}
\begin{bmatrix}
\sqrt{13} & \sqrt{13} \\
-2\sqrt{2} & 3\sqrt{2}
\end{bmatrix}
$$

NumPy를 이용

  ~~~python
  B = np.array([[2, 3], [2, 1]])
  w2, V2 = np.linalg.eig(B)
  V2
  ~~~

  ~~~python
  V2_inv = np.linalg.inv(V2)
  V2_inv
  ~~~

  ~~~python
  V2 @ np.diag(w2) @ V2_inv
  ~~~

#### 정방행렬

- 정방행렬 $$A \in \mathbf{R}^{N \times N} $$
- $$\lambda$$는 복소수 
- $$\lambda$$는  N개 
- $$AV = V \Lambda $$ 
- $$\det(A)=\prod_{i=1}^N \lambda_i $$
- $$\text{tr}(A) =\sum_{i=1}^N \lambda_i$$


아래의 행렬은 선형종속이므로 대각화가 불가능하다.

$$
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
$$

### 대각화가능

**행렬이 대각화가능하려면 고유벡터가 선형독립이어야한다.**

행렬을 대각화할 수 있으면 **대각화가능(diagonalizable) 행렬**이라고 한다. 앞서 이야기했듯이 고유벡터인 열벡터로 이루어진 행렬에 역행렬이 존재하면 대각화가능이라고 했다. 그런데 앞절에서 정방행렬의 역행렬이 존재할 조건은 정방행렬의 열벡터 즉, 고유벡터들이 선형독립인 경우이다. 따라서 행렬이 대각화가능하려면 고유벡터가 선형독립이어야한다.


### 고윳값과 역행렬

**대각화가능한 행렬에 0인 고유값이 없으면 항상 역행렬이 존재한다.**

행렬 𝐴가 대각화가능하면 다음처럼 표현할 수 있다.
$$
A = V\Lambda V^{-1}
$$
이 행렬의 역행렬은 다음처럼 계산한다.
$$
A^{-1} = (V\Lambda V^{-1})^{-1} = V \Lambda^{-1} V^{-1} 
$$
**대각행렬의 역행렬**은 각 **대각성분의 역수로 이루어진 대각행렬**이므로 0인 고유값이 없으면 항상 역행렬이 존재한다.


### 대칭행렬의 고유분해
______________________________________________________________
#### 선형연립방정식과 선형 예측모형
가중치벡터 𝑤를 찾는 것은 계수행렬이 𝑋, 미지수벡터가 𝑤, 상수벡터가 𝑦인 선형 연립방정식의 답을 찾는 것과 같다. 그리고 만약 계수행렬, 여기에서는 특징행렬 𝑋의 역행렬 $$𝑋^{-1}$$이 존재하면 다음처럼 가중치벡터를 구할 수 있다.

  $$
  Xw = y
  $$

  $$
  w = X^{-1} y
  $$

의사역행렬(pseudo inverse)은 $$𝐴^𝑇𝐴$$가 항상 정방 행렬이 된다는 점을 이용하여
$$
Ax \approx b \\
A^TAx = A^Tb \\
(A^TA)^{-1}(A^TA)x = (A^TA)^{-1}A^Tb \\
x = (A^TA)^{-1}A^T b \\
x = ((A^TA)^{-1}A^T) \\
$$

행렬 $$(𝐴^𝑇𝐴)^{−1}𝐴^𝑇$$를 행렬 𝐴의 **의사역행렬(pseudo inverse)**이라고 하며 다음처럼 𝐴+ 로 표기한다.

$$
A^{+} = (A^TA)^{-1}A^T 
$$

$$
x = A^+ b
$$

______________________________________________________________

**행렬 𝐴가 실수인 대칭행렬**이면 **고유값이 실수이고 고유벡터는 직교(orthogonal)한다.**

​우리는 실수로 이루어진 행렬만을 다룰 것이므로 앞으로는 따로 언급하지 않더라도 항상 행렬의 원소가  실수라고 가정한다. 이 때 증명은 다음과 같다.

행렬 𝐴의 고유값과 고유벡터의 관계

$$
Av = \lambda v
\tag{2.7.1}
$$

​      의 양변에 고유벡터의 켤레 복소수(complex conjugate) $$𝑣^∗$$를 곱하면,
$$
(v^{\ast})^T Av = \lambda (v^{\ast})^T v \tag{2.7.12}
$$
​     이 되고 원래의 식(2.7.1)을 켤레 복소수로 만들면
$$
(Av)^{\ast} = (\lambda v)^{\ast}
\tag{2.7.13}
$$
​     그리고 𝐴가 실수인 대칭행렬이라는 점을 이용하면
$$
A^{\ast}v^{\ast} = Av^{\ast} = \lambda^{\ast} v^{\ast} \tag{2.7.14}
$$
​     이 식의 앞에 $$𝑣^𝑇$$를 곱하면
$$
v^T A v^{\ast} = \lambda^{\ast} v^T v^{\ast} \tag{2.7.15}
$$
​     그런데 $$(𝑣∗)^𝑇𝐴𝑣$$가 스칼라이므로
$$
(v^{\ast})^T Av = ((v^{\ast})^T Av)^T = v^T A^T v^{\ast} = v^T A v^{\ast}
$$
​     가 성립하여 식(2.7.12)와 식(2.7.15)의 좌변이 같다.

​     따라서 우변도 같아야 한다.
$$
\lambda (v^{\ast})^T v = \lambda^{\ast} v^T v^{\ast} \tag{2.7.16}
$$
​    그런데 $$(𝑣∗)^𝑇𝑣$$도 스칼라이므로
$$
(v^{\ast})^T v = ((v^{\ast})^T v)^T = v^T v^{\ast}
\tag{2.7.17}
$$
​    식(2.7.16)와 식(2.7.17)에서
$$
\lambda^{\ast} = \lambda
$$
​이다. 즉, 고유값은 실수이다.

고유벡터가 직교한다는 것은 다음과 같이 증명한다. 두 고윳값 $$𝜆_𝑖, 𝜆_𝑗$$에 대응하는 고유벡터 $$𝑣_𝑖, 𝑣_𝑗$$를 생각하자.
  
  $$
  \begin{eqnarray}
  \lambda_i v_i^T v_j 
  &=& (v_i^T A) v_j \\
  &=& v_i^T (A v_j )\\
  &=& v_i^T (\lambda_j v_j) \\
  &=& \lambda_j v_i^T v_j \\
  (\lambda_i - \lambda_j) v_i^T v_j &=& 0
  \end{eqnarray}
  $$
  만약 고윳값이 서로 다르면($$𝜆_𝑖≠𝜆_𝑗$$) 두 고유벡터는 직교한다.

  만약 고윳값이 같다면($$𝜆_𝑖=𝜆_𝑗$$) $$𝑣_𝑖$$와 $$𝑣_𝑗$$는 고윳값이 같은 두 고유벡터이고 이 두 벡터의 선형조합 $$𝑐_1𝑣_𝑖+𝑐_2𝑣_𝑗$$도 $$𝜆_𝑖$$의 고유벡터가 된다. 따라서 $$𝑣_𝑖,𝑣_𝑗$$로 이루어진 벡터공간내에서 직교하는 고유벡터 두개를 항상 선택할 수 있다.

  고유벡터가 크기 1이 되도록 정규화된 상태라면 고유벡터 행렬 𝑉는 정규직교(orthonormal) 행렬이므로 **전치행렬이 역행렬**이다. 따라서 대칭행렬은 항상 대각화가능하다.


$$
V^T V = V V^T = I
$$

$$
V^{-1} = V^T
$$

**대칭행렬은 항상 대각화가능하다.**


### 대칭행렬을 랭크-1 행렬의 합으로 분해

𝑁차원 대칭행렬 𝐴는 다음처럼 𝑁개의 랭크-1 행렬 $$𝐴_𝑖=𝑣_𝑖𝑣^𝑇$$ 의 합으로 표시할 수 있다.
$$
\begin{eqnarray}
A 
&=& V\Lambda V^T \\
&=& 
\begin{bmatrix}
v_1 & v_2 & \cdots & v_N
\end{bmatrix}
\begin{bmatrix}
\lambda_{1} & 0 & \cdots & 0 \\
0 & \lambda_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{N} \\
\end{bmatrix}
\begin{bmatrix}
v_1^T \\ v_2^T \\ \vdots \\ v_N^T
\end{bmatrix} \\
&=&
\begin{bmatrix}
\lambda_{1}v_1 & \lambda_{2}v_2 & \cdots & \lambda_{N}v_N
\end{bmatrix}
\begin{bmatrix}
v_1^T \\ v_2^T \\ \vdots \\ v_N^T
\end{bmatrix} \\
\end{eqnarray}
$$
따라서 𝑁차원 대칭행렬 𝐴는
$$
A = \sum_{i=1}^{N} {\lambda_i} v_i v_i^T  = \sum_{i=1}^{N} {\lambda_i} A_i = \lambda_1 A_1 + \cdots + \lambda_N A_N
$$
**만약 0인 고윳값이 없다면** 역행렬도 다음처럼 𝑁개의 랭크-1 행렬 $$𝐴_𝑖=𝑣_𝑖𝑣^𝑇_𝑖$$ 의 합으로 표시할 수 있다.
$$
A^{-1} =
V \Lambda^{-1} V^T = \sum_{i=1}^{N} \dfrac{1}{\lambda_i} v_i v_i^T 
= \dfrac{1}{\lambda_1} A_1 + \cdots + \dfrac{1}{\lambda_N} A_N
$$


### 대칭행렬의 간략화

대칭행렬을 랭크-1 행렬의 합으로 분해할 때 각 랭크-1 행렬 $$𝐴_𝑖$$는 모두 단위벡터인 고유벡터로 만들어진 행렬이므로 행렬의 놈이 1이다. 하지만 고유값은 아주 큰 값부터 작은 값까지 다양하게 나타날 수 있다. 따라서 다음처럼 **고유값이 작은 항을 몇 개 생략**해도 **원래의 행렬과 큰 차이가 나지 않**을 수 있다.
$$
\begin{eqnarray}
A 
&=&  \lambda_1 A_1 + \cdots + \lambda_{N} A_{N} \;\; ( \lambda_1 > \cdots > \lambda_N )\\
&\approx& \lambda_1 A_1 + \cdots + \lambda_{M} A_{M} \;\; ( N > M )
\end{eqnarray}
$$

#### 대칭행렬

- 대칭행렬 $$\; A = A^T $$
- $$\lambda$$는  실수 
- $$V^T$$ = $$V^{-1}$$ 
- $$A = VAV^T$$
- $$A = \sum_{i=1}^{N} {\lambda_i} v_i v_i^T $$
- 음수인 고유값 $$ \lambda_i < 0 $$포함


### 공분산행렬

임의의 실수 행렬 𝑋에 대해 $$𝑋^𝑇𝑋$$(정방이면서 대칭행렬)인 정방행렬을 **공분산행렬(covariance matrix)**이라고 한다. 공분산행렬의 의미는 확률 분포에서 더 자세하게 공부하게 될 것이며 일단 여기에서는 위와 같은 방법으로 계산되는 행렬을 가리키는 명칭이라는 것만 알면 충분하다.

**공분산행렬은 양의 준정부호(positive semidefinite)**이다. 즉 음수인 고윳값은 없다.

임의의 영벡터가 아닌 벡터 𝑥에 대해
$$
x^T(X^TX)x = (Xx)^T(Xx) = u^Tu \geq 0
$$

$$
x^T(X^TX)x = (Xx)^T(Xx)\;\;-> 이차형식이다. (X^TX)에 x^T, x를 곱한 형태
$$


#### 이차형식(Quadratic Form)

$$
w^T(X^TX)w
$$


모든 벡터 𝑥에 대해 공분산행렬에 대한 이차형식은 어떤 벡터의 제곱합이 된다. 따라서 0보다 같거나 크다.

그런데 대칭행렬은 아래와 같이 표시되고 (A는 대칭행렬의 이차형식이다)
$$
A = \sum_{i=1}^{N} {\lambda_i} v_i v_i^T
$$
 $$𝑥=𝑣_𝑗$$일 때
$$
v_j^TAv_j = \sum_{i=1}^{N} {\lambda_i} v_j^Tv_i v_i^Tv_j = \lambda_j \geq 0 \\
v_1^T\lambda _1v_1v_1^Tv_1 = \lambda _1v_1^Tv_1v_1^Tv_1 = \lambda _(v_1^Tv_1)^2 \\
v_2^T\lambda _2v_2v_2^Tv_2 = \lambda _2v_2^Tv_2v_2^Tv_2 = \lambda _(v_2^Tv_2)^2 \\
\lambda _(v_1^Tv_1)^2 >= 0 \; (0\;또는\;항상\;양수이다)\\
\lambda _(v_2^Tv_2)^2 >= 0 \; (0\;또는\;항상\;양수이다)\\
\lambda_i >= 0 \;는\;x^TAx>=0 이다. 반대로도 성립. 양의준정부호이다.
$$
이므로 0 또는 양수인 고윳값만 가진다. (음수인 고유벡터가 없다. 양의 준정부호) 

### 고윳값과 양의 정부호

대칭행렬의 경우에 다음 성질이 성립한다. (위에 증명 참고)

**대칭행렬의 고윳값이 모두 양수이면 그 행렬은 양의 정부호(positive definite)이다.**

**역으로 양의 정부호(positive definite)인 대칭행렬의 고윳값은 항상 양수이다.**

우선 대칭행렬의 고윳값이 모두 양수이면 그 행렬은 양의 정부호(positive definite)가 됨을 증명하자. 일단 다음처럼 고유분해로 만들어진 행렬 $$𝐴_𝑖=𝑣_𝑖𝑣^𝑇$$는 양의 준정부호(positive semidefinite)임을 증명할 수 있다.
$$
x^T A_i x = x^T v_iv_i^T x = (x^T v_i)(x^T v_i)^T = (x^T v_i)(x^T v_i) = \vert x^T v_i \vert ^2 \geq 0
$$
이 식에서 𝑥가 $$𝑣_𝑖$$와 수직(orthogonal)인 경우에만 0이 된다는 것을 알 수 있다. 여기에 양수인 고윳값을 곱한 행렬 $$𝜆_𝑖𝐴_𝑖$$도 마찬가지로 양의 준정부호(positive semidefinite)이다. 이러한 행렬 $$𝜆_𝑖𝐴_𝑖$$를 모두 더한 행렬 $$𝜆_1𝐴_1+⋯+𝜆_𝑁𝐴_𝑁$$은 양의 정부호(positive definite)이다.
$$
\begin{eqnarray}
A 
&=& \lambda_1 A_1 + \cdots + \lambda_N A_N \\
&=& \vert x^T v_1\vert ^2 + \cdots + \vert x^T v_N\vert ^2 > 0 \\
\end{eqnarray}
$$
왜나하면 이 값이 0이려면 모든 $$𝑥^𝑇𝑣_𝑖$$가 0, 다시 말해 𝑥와 모든 $$𝑣_𝑖$$가 직교해야 하는데 대칭행렬의 고유벡터의 집합은 𝑁 차원에서 기저벡터를 이루기 때문에 동시에 모든 기저벡터와 수직인 벡터는 존재하지 않기 때문이다. (연습문제 2.6.7 참고)

역으로 양의 정부호(positive definite)인 대칭행렬의 고윳값은 항상 양수이다. 만약 0이나 음수인 고윳값 𝜆≤0가 존재한다면 다음처럼 이차 형식이 양수가 아닐 수 있기 때문이다.
$$
v_i^T A v_i = v_i^T \lambda_i v_i = \lambda_i v_i^T v_i \leq 0
$$

### 공분산행렬의 역행렬

**행렬 𝑋가 풀랭크이면 이 행렬의 공분산행렬 $$𝑋^𝑇𝑋$$의 역행렬이 존재한다.**

정방행렬이 아닌 행렬 X를 말한다. 행렬X의 열벡터가 모두 독립이면 full rank이다.
정방행렬가 full rank이면  A <=> $$A^{-1}$$ 이 성립한다.

행렬 𝑋가 풀랭크이면 𝑋의 열벡터가 **기저벡터**(벡터 N개가 서로 선형독립이고, 이 벡터들을 선형조합하여 만들어지는 모든 벡터의 집합을 벡터공간이고 그 벡터들을 벡터공간의 기저벡터라고 한다.)를 이루기 때문에 영벡터가 아닌 모든 벡터 𝑣에 대해 𝑋𝑣=𝑢는 영벡터가 될 수 없다. (만약 영벡터 𝑢를 만드는 영벡터가 아닌 𝑣가 존재한다면 서로 독립이 아니다.) 그러면 𝑋^𝑇𝑋의 이차형식은 항상 양수가 된다.

$$
v^T(X^TX)v = (Xv)^T(Xv) = u^Tu > 0
$$
따라서 공분산행렬은 양의 정부호이고 역행렬이 존재한다.



#### 공분산행렬

공분산행렬 $$ A = X^TX $$이다. 
모든 고유값이 0  또는 양수 $$\lambda_i >= 0$$ (양의준정부호이다)이다. 



## 양의 정부호(PD) - 이 때만 데이터 분석할 수 있다.

X가 풀랭크 
역행렬이 존재 
모든 고유값이 양수 $$\lambda_i > 0$$ (PD)

### 고유분해의 성질 

𝑁차원 정방행렬 𝐴에 대해 다음 성질이 성립한다.

1. 행렬 𝐴는 𝑁개의 고윳값-고유벡터를 가진다. (복소수인 경우와 중복인 경우를 포함)
2. 행렬의 **대각합은 모든 고윳값의 합**과 같다.
3. 행렬의 **행렬식은 모든 고윳값의 곱**과 같다.
4. 행렬 𝐴가 **대칭행렬**이면 𝑁개의 **실수 고윳값**을 가지며 고유벡터들이 서로 **수직(orthogonal)**이다.
5. 행렬 𝐴가 **대칭행렬**이고 고윳값이 모두 **양수**이면 **양의 정부호(positive-definite)이고 역행렬이 존재한다**. 역도 성립한다.
6. 행렬 𝐴가 어떤 행렬 𝑋의 **공분산행렬 $$𝑋^𝑇𝑋$$**이면 **0 또는 양의 고윳값을 가진다**.
7. 행렬 𝑋가 **풀랭크이면 공분산행렬 $$𝑋^𝑇𝑋$$은 역행렬이 존재**한다.

