<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 벡터와 행렬의 연산

### Summary

- 브로드캐스팅은 1-벡터를 사용하여 스칼라를 벡터로 변환하는 연산이다. 데이터 분석에서 원래 데이터 벡터의 각 원소에 평균값을 뺀 것을 평균제거 벡터 혹은 0-벡터를 사용하는 경우가 많다.
- 벡터나 행렬에 스칼라 값을 곱한 후 더하거나 뺀 것을 선형조합이라고 한다.
- 벡터와 벡터 내적(곱)의 결과는 스칼라이다.
- 가중합은 복수의 데이터를 각각의 수에 어떤 가중치를 곱한 후 합한 것이다.
- 가중평균은 가중치 값을 전체 가중치 합으로 나눈 것을 말한다. 
- 유사도는 두 벡터의 닮은 정도를 정량화한 값이다. 
- 선형회귀 모형은 독립변수 x에 대응하는 종속변수 y와 가장 비슷한 값을 출력하는 선형함수를 찾는 과정이다. 독립 벡터 x에 가중치 벡터 w를 가중합하여 y에 대한 예측값을 구하는 수식이다.
- 제곱합은 분산, 표준편차를 구하는데 사용된다.
- 잔차제곱합(RSS)은 잔차의 크기를 구할 때 사용하고 잔차 벡터의 각 원소를 제곱한 후 더한다.
- 이차형식은 행벡터x정방행렬x열벡터 형식이다. $$x^TAx$$

______________

### 벡터/행렬의 덧셈과 뺄셈

같은 크기를 가진 두 개의 벡터나 행렬은 덧셈과 뺄셈을 할 수 있다. 두 벡터와 행렬에서 같은 위치에 있는 원소끼리 덧셈과 뺄셈을 한다. 이러한 연산을 요소별(element-wise) 연산이라고 한다.

### 스칼라와 벡터/ 행렬의 곱셈

벡터 𝑥 또는 행렬 𝐴에 스칼라 값 𝑐를 곱하는 것은 **벡터 𝑥 또는 행렬 𝐴의 모든 원소에 스칼라 값 𝑐를 곱하는 것**과 같다.
$$
\begin{align}
c
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
cx_1 \\
cx_2
\end{bmatrix}
\tag{2.2.1}
\end{align}
$$

$$
\begin{align}
c
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
=
\begin{bmatrix}
ca_{11} & ca_{12} \\
ca_{21} & ca_{22}
\end{bmatrix}
\tag{2.2.2}
\end{align}
$$

### 브로드캐스팅

원래 덧셈과 뺄셈은 크기(차원)가 같은 두 벡터에 대해서만 할 수 있다. 하지만 벡터와 스칼라의 경우에는 관례적으로 다음처럼 1-벡터를 사용하여 스칼라를 벡터로 변환한 연산을 허용한다. 이를 **브로드캐스팅(broadcasting)**이라고 한다.
$$
\begin{bmatrix}
10 \\
11 \\
12 \\
\end{bmatrix}
- 10
=
\begin{bmatrix}
10 \\
11 \\
12 \\
\end{bmatrix}
- 10\cdot \mathbf{1}
=
\begin{bmatrix}
10 \\
11 \\
12 \\
\end{bmatrix}
-
\begin{bmatrix}
10 \\
10 \\
10 \\
\end{bmatrix}
$$

데이터 분석에서는 원래의 데이터 벡터 $$x$$가 아니라 그 데이터 벡터의 각 원소의 평균값을 뺀 **평균제거(mean removed) 벡터** 혹은 **0-평균(zero-mean) 벡터**를 사용하는 경우가 많다.
$$
\begin{align}
x = 
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{bmatrix}
\;\; \rightarrow \;\;
x - m =
\begin{bmatrix}
x_1 - m\\
x_2 - m \\
\vdots \\
x_N - m
\end{bmatrix}
\tag{2.2.3}
\end{align}
$$

위 식에서 $$m$$ 은 샘플 평균이다.
$$
\begin{align}
m = \dfrac{1}{N}\sum_{i=1}^N x_i
\tag{2.2.4}
\end{align}
$$

### 선형조합

벡터/행렬에 다음처럼 스칼라 값을 곱한 후 더하거나 뺀 것을 벡터/행렬의 선형조합(linear combination)이라고 한다. 벡터나 행렬을 선형조합해도 크기는 변하지 않는다. 
$$
\begin{align}
c_1x_1 + c_2x_2 + c_3x_3 + \cdots + c_Lx_L = x
\tag{2.2.5}
\end{align}
$$

$$
\begin{align}
c_1A_1 + c_2A_2 + c_3A_3 + \cdots + c_LA_L = A
\tag{2.2.6}
\end{align}
$$

$$
c_1, c_2, \ldots, c_L \in \mathbf{R} \\
x_1, x_2, \ldots, x_L, x \in \mathbf{R}^M \\
A_1, A_2, \ldots, A_L, A \in \mathbf{R}^{M \times N}
$$

벡터나 행렬의 크기를 직사각형으로 표시하면 다음과 같다.
$$
\begin{align*}
\begin{matrix}
c_1\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ x_1 \\ \phantom{\LARGE\mathstrut} \end{matrix}} & + &
c_2\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ x_2 \\ \phantom{\LARGE\mathstrut} \end{matrix}} & + & 
\cdots \!\!\!\!& + & 
c_L\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_L \\ \phantom{\LARGE\mathstrut} \end{matrix}}
\end{matrix}
\end{align*}
$$

$$
\begin{align*}
\begin{matrix}
c_1\,\boxed{\begin{matrix} \phantom{} & \phantom{} & \phantom{} \\ & A_1 & \\ \phantom{} & \phantom{} & \phantom{} \end{matrix}} 
& + &
c_2\,\boxed{\begin{matrix} \phantom{} & \phantom{} & \phantom{} \\ & A_2 & \\ \phantom{} & \phantom{} & \phantom{} \end{matrix}} 
& + &
\cdots
& + &
c_L\,\boxed{\begin{matrix} \phantom{} & \phantom{} & \phantom{} \\ & A_L & \\ \phantom{} & \phantom{} & \phantom{} \end{matrix}} 
\end{matrix}
\end{align*}
$$



### 벡터와 벡터의 곱셈(내적)

- $$x^Ty$$
- 내적의 조건
  - 두 벡터의 차원(길이)가 같아야 한다.
  - 앞의 벡터가 행벡터이고 뒤의 벡터가 열벡터여야 한다. 
- 내적의 결과는 **스칼라** 이다. 

$$
x^T y = 
\begin{bmatrix}
x_{1} & x_{2} & \cdots & x_{N} 
\end{bmatrix}
\begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{N} \\
\end{bmatrix} 
= x_1 y_1 + \cdots + x_N y_N 
= \sum_{i=1}^N x_i y_i
$$

### 가중합

복수의 데이터를 각각의 수에 어떤 가중치를 곱한 후 이 곱셈 결과들을 다시 합한 것
$$
\begin{aligned}
\sum_{i=1}^N w_i x_i 
&= 
\begin{bmatrix}
w_{1} && w_{2} && \cdots && w_{N}
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_N
\end{bmatrix} 
&= w^Tx  
\\
&=
\begin{bmatrix}
x_{1} && x_{2} && \cdots && x_{N}
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_N
\end{bmatrix}
&= x^Tw  
\end{aligned}
$$

### 가중평균

가중합의 가중치값을 전체 가중치값의 합으로 나눈 것
$$
\bar{x} = \dfrac{1}{N}\sum_{i=1}^N x_i = \dfrac{1}{N} \mathbf{1}_N^T x
$$

### 유사도(similarity)

두 벡터가 닮음 정도를 정량적으로 나타낸 값이다. 비슷한 경우 유사도가 커진다. 

### 선형회귀 모형

독립변수 $$x$$ 에서 종속변수 $$y$$ 를 예측하는 방법의 하나로 , 독립변수 벡터 $$x$$ 와 가중치 벡터 $$w$$ 와의 가중합으로 $$y$$ 에 대한 예측값 $$\hat y$$ 를 계산하는 수식이다. 
$$
\hat y = w^Tx
$$

### 제곱합(sum of squares)

데이터 분산(variance), 표준편차(standard deviation) 구하는 경우에 사용
$$
x^T x = 
\begin{bmatrix}
x_{1} & x_{2} & \cdots & x_{N} 
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{N} \\
\end{bmatrix} = \sum_{i=1}^{N} x_i^2
$$

### 행렬과 행렬의 곱셈

앞의 행렬 열의 수가 뒤의 행렬 행의 수와 일치해야 한다
$$
A \in \mathbf{R}^{N \times L} , \; B \in \mathbf{R}^{L \times M} \;  \rightarrow \; AB \in \mathbf{R}^{N \times M}
$$

$$
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
a_{41} & a_{42} & a_{43} \\
\end{bmatrix}
\begin{bmatrix}
{b_{11}} & b_{12} \\
{b_{21}} & b_{22} \\
{b_{31}} & b_{32} \\    
\end{bmatrix}
=
\begin{bmatrix}
(a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31}) & (a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32}) \\
(a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31}) & (a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32}) \\
(a_{31}b_{11} + a_{32}b_{21} + a_{33}b_{31}) & (a_{31}b_{12} + a_{32}b_{22} + a_{33}b_{32}) \\
(a_{41}b_{11} + a_{42}b_{21} + a_{43}b_{31}) & (a_{41}b_{12} + a_{42}b_{22} + a_{43}b_{32}) \\
\end{bmatrix}
$$

### 교환법칙과 분배법칙

$$
AB \neq BA \\
A(B+C) = AB + AC \\
(A+B)C = AC + BC
$$

$$
(AB)^T = B^TA^T
$$

### 곱셉의 연결

$$
ABC = (AB)C = A(BC)
$$

### 항등행렬의 곱셈

$$
AI = IA = A
$$

### 행렬과 벡터의 곱

행렬 $$M$$ 과 벡터 $$v$$ 의 곱
$$
\boxed{\begin{matrix} 
\phantom{} & \phantom{} & \phantom{} & \phantom{} & \phantom{} \\ 
& & M & &\\ 
\phantom{} & \phantom{} & \phantom{} & \phantom{} & \phantom{} \\ 
\end{matrix}} \,
\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ v \\ \phantom{\LARGE\mathstrut} \end{matrix}}
=
\boxed{\begin{matrix} 
\phantom{} \\ 
Mv \\ 
\phantom{}  
\end{matrix}}
$$

### 열벡터의 선형조합

$$
\begin{align}
Xw=
\begin{bmatrix}
c_1 & c_2 & \cdots & c_M
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_M
\end{bmatrix}
=
w_1 c_1 + w_2 c_2 + \cdots + w_M c_M
\end{align}
$$

$$
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_{1_{\phantom{1}}} \\ \phantom{\LARGE\mathstrut} \end{matrix}} &
\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_{2_{\phantom{1}}} \\ \phantom{\LARGE\mathstrut} \end{matrix}} &
\cdots & 
\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_M \\ \phantom{\LARGE\mathstrut} \end{matrix}} 
\end{bmatrix} 
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_M
\end{bmatrix}
=
\begin{matrix}
w_1\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_{1_{\phantom{1}}} \\ \phantom{\LARGE\mathstrut} \end{matrix}} & + &
w_2\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_{2_{\phantom{1}}} \\ \phantom{\LARGE\mathstrut} \end{matrix}}& + & 
\cdots \!\!\!\!& + & 
w_M\,\boxed{\begin{matrix} \phantom{\LARGE\mathstrut} \\ c_M \\ \phantom{\LARGE\mathstrut} \end{matrix}}
\end{matrix}
$$

### 여러 개의 벡터에 대한 가중합 동시 계산

$$
\begin{aligned}
\hat{y} = 
\begin{bmatrix}
\hat{y}_1 \\
\hat{y}_2 \\
\vdots \\
\hat{y}_M \\
\end{bmatrix}
&= 
\begin{bmatrix}
w_1 x_{1,1} + w_2 x_{1,2} + \cdots + w_N x_{1,N} \\
w_1 x_{2,1} + w_2 x_{2,2} + \cdots + w_N x_{2,N} \\
\vdots  \\
w_1 x_{M,1} + w_2 x_{M,2} + \cdots + w_N x_{M,N} \\
\end{bmatrix}
\\
&= 
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,N} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,N} \\
\vdots  & \vdots  & \vdots & \vdots \\
x_{M,1} & x_{M,2} & \cdots & x_{M,N} \\
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_N
\end{bmatrix}
\\
&= 
\begin{bmatrix}
x_1^T \\
x_2^T \\
\vdots \\
x_M^T \\
\end{bmatrix}
\begin{bmatrix}
w_1 \\ w_2 \\ \vdots \\ w_N
\end{bmatrix}
\\
&= X w
\end{aligned}
$$

### 잔차(residual)

예측치와 실제값의 차이
$$
e_i = y_i -\hat y_i = y_i - w^Tx_i
$$

$$
\begin{aligned}
e 
&=
\begin{bmatrix}
e_{1} \\
e_{2} \\
\vdots \\
e_{M} \\
\end{bmatrix}
\\ 
&=
\begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{M} \\
\end{bmatrix}
-
\begin{bmatrix}
x^T_{1}w \\
x^T_{2}w \\
\vdots \\
x^T_{M}w \\
\end{bmatrix}
\\ 
&= y - Xw
\end{aligned}
$$

### 잔차제곱합(RSS, Residual Sum of Squares)

- 잔차의 크기를 구할때 사용
- 잔차벡터의 각 원소를 제곱한 후 더한다. 

$$
\sum_{i=1}^{N} e_i^2 = \sum_{i=1}^{N} (y_i - w^Tx_i)^2 = e^Te =  (y - Xw)^T (y - Xw)
$$

### 이차 형식

- $$w^TAw$$  (A는 정방행렬)
- 어떤 벡터와 정방 행렬이 '행벡터 x 정방행렬 x 열벡터'의 형식
- 이 수식을 풀면 $$i = 1, …, N$$, $$j = 1, …., N$$ 에 대해 가능한 모든 $$i, j$$ 쌍의 조합을 구한 다음 $$i, j$$ 에 해당하는 원소 $$x_i, x_j$$ 를 가중치 $$a_{i,j}$$ 와 같이 곱한 값 $$a_{i,j}x_ix_j$$ 의 총합이 된다. 

$$
\begin{align}
\begin{aligned}
x^T A x 
&= 
\begin{bmatrix}
x_{1} & x_{2} & \cdots & x_{N} 
\end{bmatrix}
\begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,N} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
a_{N,1} & a_{N,2} & \cdots & a_{N,N} \\
\end{bmatrix}
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{N} \\
\end{bmatrix} 
\\
&= \sum_{i=1}^{N} \sum_{j=1}^{N} a_{i,j} x_i x_j 
\end{aligned}
\end{align}
$$

### 부분행렬

앞에 곱해지는 행렬을 행벡터로 나누어 계산해도 된다.
$$
AB 
=
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{} & a_1^T & \phantom{} \end{matrix}} \\ 
\boxed{\begin{matrix} \phantom{} & a_2^T & \phantom{} \end{matrix}} \\ 
\end{bmatrix}
B
=
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{} & a_1^TB & \phantom{} \end{matrix}} \\ 
\boxed{\begin{matrix} \phantom{} & a_2^TB & \phantom{} \end{matrix}} \\ 
\end{bmatrix}
$$

뒤에 곱해지는 행렬을 열벡터로 나누어 계산해도 된다.
$$
AB 
=
A
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{\mathstrut} \\ b_1 \\ \phantom{\mathstrut} \end{matrix}} \!\!\!\! & 
\boxed{\begin{matrix} \phantom{\mathstrut} \\ b_2 \\ \phantom{\mathstrut} \end{matrix}}
\end{bmatrix}
=
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{\mathstrut} \\ Ab_1 \\ \phantom{\mathstrut} \end{matrix}} \!\!\!\! &
\boxed{\begin{matrix} \phantom{\mathstrut} \\ Ab_2 \\ \phantom{\mathstrut} \end{matrix}}
\end{bmatrix}
$$

앞에 곱해지는 행렬을 열벡터로, 뒤에 곱해지는 행렬을 행벡터로 나누어 스칼라처럼 계산해도 된다. 
$$
AB 
=
\begin{bmatrix}
a_1 & a_2
\end{bmatrix}
\begin{bmatrix}
b_1^T \\ b_2^T
\end{bmatrix}
=
a_1b_1^T + a_2b_2^T
$$

$$
AB 
=
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{\mathstrut} \\ a_1 \\ \phantom{\mathstrut} \end{matrix}}  \!\!\!\!& 
\boxed{\begin{matrix} \phantom{\mathstrut} \\ a_2 \\ \phantom{\mathstrut} \end{matrix}}  \!
\end{bmatrix}
\begin{bmatrix}
\boxed{\begin{matrix} \phantom{} & b_1^T & \phantom{} \end{matrix}} \\ 
\boxed{\begin{matrix} \phantom{} & b_2^T & \phantom{} \end{matrix}} \\ 
\end{bmatrix}
=
\boxed{\begin{matrix} \phantom{\mathstrut} \\ a_1 \\ \phantom{\mathstrut} \end{matrix}} 
\boxed{\begin{matrix} \phantom{} & b_1^T & \phantom{} \end{matrix}} 
+
\boxed{\begin{matrix} \phantom{\mathstrut} \\ b_1 \\ \phantom{\mathstrut} \end{matrix}} 
\boxed{\begin{matrix} \phantom{} & b_2^T & \phantom{} \end{matrix}} 
$$



___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 