<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 데이터와 행렬(용어정리)

### 데이터와 행렬

- 선형대수(linear algebra): 데이터 분석에 필요한 각종 계산을 돕기 위한 학문

### 데이터 유형

- 스칼라(scalar) : 하나의 숫자만으로 이루어진 데이터
- 벡터(vector) : 여러개의 숫자가 특정한 순서대로 모여있는 것
- 특징벡터(feature vector) : 예측에 사용되는 벡터
- 텐서(tensor) : 같은 크기의 행렬이 여러개 같이 묶여 있는 것

### 전치연산(transpose) : 행과 열을 바꾸는 연산

- $$x$$ -> $$x^T$$ 

- 전치연산으로 만든 행렬을 원래행렬의 전치행렬이라고 한다. 열벡터 $$x$$ 에 대해 전치 연산을 적용하여 만든 $$x^T$$ 는 행수가 1인 행벡터(row vector) 라고 한다.

$$
x = 
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{N} \\
\end{bmatrix}
\; \rightarrow \;
x^T = 
\begin{bmatrix}
x_{1} & x_{2} & \cdots & x_{N}
\end{bmatrix}
$$

### 특수한 벡터와 행렬

- 영벡터(zero-vector) : 모든 원소가 0인 N차원 벡터
- 일벡터(ones-vector) : 모든 원소가 1인 N차원 벡터
- 정방행렬(square-vector) : 행과 열의 개수가 같은 행렬
- 대각행렬(diagonal matrix) : 모든 비대각 요소는 0인 행렬, 행렬에서 행과 열이 같은 위치를 대각(diagonal), 대각 위치에 있지 않은 것들은 비대각(off-diagonal)이라고 한다. 
- 항등행렬(identity matirx) : 대각행렬 중에서도 모든 대각 성분의 값이 1인 대각 행렬
- 대칭행렬(symmetric matirx) : 전치행렬과 원래의 행렬이 같은 행렬, **정방행렬만 대칭행렬이 될 수 있다**


___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 
