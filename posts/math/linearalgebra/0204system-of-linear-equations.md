<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 선형 연립방정식(system of linear equations) & 역행렬
### Summary

- 대각행렬의 역행렬은 각 대각성분의 역수로 이루어진 대각행렬과 같다.
- 역행렬은 행렬식이 0이 아닌 경우에만 존재한다.
- 최소자승문제는(least square ploblem) 예측값과 목표값의 차이인 잔차를 최소화 하는 문제이다. 잔차는 벡터이므로 벡터의 놈을 최소화하는 문제로 볼 수 있다. 벡터의 놈을 최소화하는 것은 놈의 제곱을 최소화 하는 것과 같다. 잔차 제곱합은 놈의 제곱이 된다. 
  - $$e^Te = ||e||^2 = (Ax-b)^T(Ax-b)$$, $$x=\text{arg} \  \text{min}_x \ e^Te = \text{arg} \  \text{min}_x \ (Ax-b)^T(Ax-b)$$
____________

- 복수의 미지수를 포함하는 복수의 선형 방정식

$$
A = 
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1M} \\
a_{21} & a_{22} & \cdots & a_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
a_{N1} & a_{N2} & \cdots & a_{NM} \\
\end{bmatrix}
, \;\;
x = 
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_M
\end{bmatrix}
, \;\;
b=
\begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_N
\end{bmatrix}
$$

$$
Ax = b  
$$

- A는 계수행렬(coefficient matrix), x는 미지수벡터(unknown vector),  b는 상수벡터(constant vector)

### 역행렬(inverse matrix)

- 정방 행렬 A에 대한 역행렬은 원래의 행렬 A와 다음 관계를 만족하는 정방 행렬을 말한다. I는 항등 행렬(identity matrix) 이다.


$$
A^{-1} A = A A^{-1} = I
$$

- 역행렬은 항상 존재하는 것이 아니라 **행렬 A에 따라서는 존재하지 않을 수도 있다**. 역행렬이 존재하는 행렬을 **가역행렬(invertible matrix)**, 정칙행렬(regular matrix) 또는 비특이행렬(non-singular matrix)이라고 한다. 반대로 역행렬이 존재하지 않는 행렬을 비가역행렬(non-invertible) 또는 **특이행렬(singular matrix)**, 퇴화행렬(degenerate matrxi)이라고 한다.

- 대각행렬의 역행렬은 각 대각성분의 역수로 이루어진 대각행렬과 같다.
  $$
  \begin{bmatrix}
  \lambda_{1} & 0 & \cdots & 0 \\
  0 & \lambda_{2} & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & \lambda_{N} \\
  \end{bmatrix}^{-1}
  =
  \begin{bmatrix}
  \dfrac{1}{\lambda_{1}} & 0 & \cdots & 0 \\
  0 & \dfrac{1}{\lambda_{2}} & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & \cdots & \dfrac{1}{\lambda_{N}} \\
  \end{bmatrix}
  \tag{2.4.6}
  $$

### 역행렬의 성질

- 행렬 A, B, C 는 모두 각각 역행렬이 존재한다고 가정한다.

- **전치 행렬의 역행렬은 역행렬의 전치 행렬과 같다. 따라서 대칭 행렬의 역행렬도 대칭 행렬이다.**
  $$
  (A^{T})^{-1} = (A^{-1})^{T}
  $$

- **두 개 이상의 정방 행렬의 곱은 같은 크기의 정방 행렬이 되는데 이러한 행렬의 곱의 역행렬은 다음 성질이 성립한다.**
  $$
  (AB)^{-1} = B^{-1} A^{-1}
  $$

  $$
  (ABC)^{-1} = C^{-1} B^{-1} A^{-1}
  $$

### 역행렬의 계산

$$
A^{-1} = \dfrac{1}{\det (A)} C^T = \dfrac{1}{\det (A)} 
\begin{bmatrix}
C_{1,1} & \cdots & C_{N,1}  \\
\vdots  & \ddots & \vdots   \\
C_{1,N} & \cdots & C_{N,N}  \\
\end{bmatrix}
$$

- 이 식에서 𝐶𝑖,𝑗는 𝐴의 𝑖,𝑗번째 원소에 대해 정의한 코팩터이다.

  -----------------------------------------------------------------------------------------

  코팩터(cofactor, 여인수)는
  $$
  C_{i,j} = (-1)^{i+j}M_{i,j}  
  $$

  $$
  \det(A) = \sum_{i=1}^N C_{i,j_0} a_{i,j_0}  =  \sum_{j=1}^N C_{i_0,j} a_{i_0,j}
  $$

  ------------------------------------------

  코팩터로 이루어진 행렬 𝐶을 **코팩터행렬(matrix of cofactors, 또는 cofactor matrix, comatrix)**이라고 한다. 또 코팩터행렬의 전치행렬 $$𝐶^𝑇$$를 **어드조인트행렬**(**adjoint matrix, adjugate matrix**, 수반행렬)이라고 하며 **adj(A)**로 표기하기도 한다.

  위 식에서 det(A)=0이면 역수가 존재하지 않으므로 **역행렬은 행렬식이 0이 아닌 경우에만 존재한다**는 것을 알 수 있다.

  

### 역행렬에 대한 정리

셔먼-모리슨(Sherman-Morrison) 공식

- 정방행렬 𝐴와 벡터 𝑢,𝑣에 대해 다음 공식이 성립한다.
  $$
  (A+uv^T)^{-1} = A^{-1} - {A^{-1}uv^T A^{-1} \over 1 + v^T A^{-1}u}
  $$

우드베리(Woodbury) 공식

- 정방행렬 𝐴와 이에 대응하는 적절한 크기의 행렬 𝑈,𝑉,𝐶에 대해 다음 공식이 성립한다.
  $$
  \left(A+UCV \right)^{-1} = A^{-1} - A^{-1}U \left(C^{-1}+VA^{-1}U \right)^{-1} VA^{-1}
  $$

분할 행렬의 역행렬

- 4개의 블록(block)으로 분할된 행렬(partitioned matrix)의 역행렬은 각 분할 행렬을 이용하여 계산할 수 있다.
  $$
  \begin{bmatrix}
  A_{11} & A_{12} \\
  A_{21} & A_{22} 
  \end{bmatrix}^{-1}
  =
  \begin{bmatrix}
  A_{11}^{-1}(I + A_{12}FA_{11}^{-1}) & -A_{11}^{-1}A_{12}F \\
  -FA_{21}A_{11}^{-1} & F
  \end{bmatrix}
  $$

  $$
  F = (A_{22} - A_{21}A_{11}^{-1}A_{12})^{-1}
  $$

  $$
  F = (A_{11} - A_{12}A_{22}^{-1}A_{21})^{-1}
  $$

### 역행렬과 선형 연립방정식의 해

$$
Ax = b
$$

$$
A^{-1}Ax = A^{-1}b
$$

$$
Ix = A^{-1}b
$$

$$
x = A^{-1}b
$$

### 선형 연립방정식과 선형 예측모형

- 선형 예측모형의 가중치벡터를 구하는 문제는 선형 연립방정식을 푸는 것과 같다. 예를 들어 𝑁개의 입력차원을 가지는 특징벡터 𝑁개를 입력 데이터로 이용하고 이 입력에 대응하는 목푯값벡터를 출력하는 선형 예측모형을 생각하자.

$$
\begin{matrix}
x_{11} w_1 & + \;& x_{12} w_2   &\; + \cdots + \;& x_{1N} w_N &\; = \;& y_1 \\
x_{21} w_1 & + \;& x_{22} w_2   &\; + \cdots + \;& x_{2N} w_N &\; = \;& y_2 \\
\vdots\;\;\; &   & \vdots\;\;\; &                & \vdots\;\;\; &     & \;\vdots \\
x_{N1} w_1 & + \;& x_{N2} w_2   &\; + \cdots + \;& x_{NN} w_N &\; = \;& y_N \\
\end{matrix}
$$

$$
Xw = y
$$

- 이 예측 모형의 가중치벡터 𝑤를 찾는 것은 계수행렬이 𝑋, 미지수벡터가 𝑤, 상수벡터가 𝑦인 선형 연립방정식의 답을 찾는 것과 같다. 그리고 만약 계수행렬, 여기에서는 특징행렬 𝑋의 역행렬 $$𝑋^{−1}$$이 존재하면 다음처럼 가중치벡터를 구할 수 있다.

$$
w = X^{-1} y  
$$

### 미지수의 수와 방정식의 수

- 미지수의 수와 방정식의 수를 고려해 볼 때 연립방정식에는 다음과 같은 세 종류가 있을 수 있다.
  1. 방정식의 수가 미지수의 수와 같다. (𝑁=𝑀)
  2. 방정식의 수가 미지수의 수보다 적다. (𝑁<𝑀)
  3. 방정식의 수가 미지수의 수보다 많다. (𝑁>𝑀)

- 선형 예측모형을 구할 때는 3번과 같은 경우가 많다는 것을 알 수 있다.

### 최소자승문제

따라서 미지수의 갯수보다 방정식의 갯수가 많아서 선형 연립방정식으로 풀수 없는 문제는 좌변과 우변의 차이를 최소화하는 문제로 바꾸어 풀 수 있다. 앞서 예측값과 목푯값의 차이를 **잔차(residual)**라고 한다고 했다.
$$
e = Ax - b  
$$

잔차는 벡터이므로 최소자승문제에서는 벡터의 크기 중에서 **벡터의 놈(norm)을 최소화**하는 문제를 푼다. 앞 절에서 놈을 최소화하는 것은 놈의 제곱을 최소화하는 것과 같다고 했다. 여기에서는 잔차제곱합이 놈의 제곱이 된다.
$$
e^Te = \Vert e \Vert^2 = (Ax-b)^T(Ax-b)  
$$

이 값을 최소화 하는 𝑥 값은 수식으로 다음처럼 표현한다.
$$
x = \text{arg} \min_x e^Te = \text{arg} \min_x  \; (Ax-b)^T(Ax-b)
$$
위 식에서 **$$\text{arg} \min_x 𝑓(𝑥)$$는 함수 𝑓(𝑥)를 가장 작게 만드는 x 값**을 의미한다.

이러한 문제를 **최소자승문제(least square problem)**라고 한다.
$$
Ax \approx b
$$

$$
A^TAx = A^Tb
$$

$$
(A^TA)^{-1}(A^TA)x = (A^TA)^{-1}A^Tb
$$

$$
x = (A^TA)^{-1}A^T b
$$

$$
x = ((A^TA)^{-1}A^T) b  
$$

- $$(𝐴^𝑇𝐴)^{−1}𝐴^𝑇$$를 행렬 A의 **의사 역행렬(pseudo inverse)**이라고 하며 다음처럼 A+ 로 표기한다.

$$
A^{+} = (A^TA)^{-1}A^T
$$

$$
x = A^+ b
$$

- NumPy의 `lstsq` 명령은 사실 이러한 최소자승문제를 푸는 명령이다.

~~~python
A = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 2]])
b = np.array([[2], [2], [3], [4.1]])
Apinv = np.linalg.inv(A.T @ A) @ A.T
x = Apinv @ b
A @ x
~~~

~~~python
x, resid, rank, s = np.linalg.lstsq(A, b)
x
~~~

- 위 코드에서 `resid`는 잔차벡터의 𝑒=𝐴𝑥−𝑏e=Ax−b의 제곱합, 즉 놈의 제곱이다.

~~~python
resid, np.linalg.norm(A @ x - b) ** 2
~~~


___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 