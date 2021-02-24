<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 행렬의 성질

### 정부호, 준정부호

양의 정부호(positive definite)
$$
x^TAx > 0
$$

양의 준정부호(positive semi-definite)
$$
x^TAx \geq 0
$$

### 행렬 놈(norm)

$$
\Vert A \Vert_p = \left( \sum_{i=1}^N \sum_{j=1}^M |a_{ij}|^p \right)^{1/p}  
$$

$$p$$=2 ; 프로베니우스 놈
$$
\Vert A \Vert = \Vert A \Vert_2 = \Vert A \Vert_F = \sqrt{\sum_{i=1}^N \sum_{j=1}^M a_{ij}^2}
$$

**놈은 항상 0보다 같거나 크다**.**벡터의 놈의 제곱이 벡터의 제곱합과 같다**
$$
\Vert x \Vert^2 = \sum_{i=1}^N x_{i}^2 = x^Tx
$$

**놈을 최소화하는 것은 벡터의 제곱합을 최소화 하는 것과 같다**

#### 놈 성질

1. 놈의 값은 0이상이다. 영행렬 일 때만 놈의 값이 0이 된다.
   $$
   \Vert A \Vert \geq 0
   $$

2. 행렬에 스칼라를 곱하면 놈의 값도 그 스칼라의 절댓값을 곱한 것과 같다.
   $$
   \|\alpha A\|=|\alpha| \|A\|
   $$

3. 행렬의 합의 놈은 각 행렬의 놈의 합보다 작거나 같아.
   $$
   \|A+B\| \le \|A\|+\|B\|
   $$

4. 정방행렬의 곱의 놈은 각 정방행렬의 놈의 곱보다 작거나 같아.

$$
\|AB\| \le \|A\|\|B\|
$$

### 대각합(trace)

- 정방행렬에 대해서만 정의된다. 

$$
\text{tr}(A) = a_{11} + a_{22} + \dots + a_{NN}=\sum_{i=1}^{N} a_{ii}
$$

$$
tr(I_N) = N
$$

- 절댓값을 취하거나 제곱을 하지 않기 때문에 음수가 될 수 도 있다.

#### 대각합의 성질

1. 스칼라를 곱하면 대각합은 스칼라와 원래의 대각합의 곱니다.
$$
tr(cA)=c\ tr(A)
$$

2. 전치연산을 해도 대각합이 달라지지 않는다.
$$
tr(A^T) = tr(A)
$$

3. 두 행렬의 합의 대각합은 두 행렬의 대각합의 합니다.
$$
tr(A+B)=tr(A)+tr(B)
$$

4. 두 행렬의 곱의 대각합은 행렬의 순서를 바꾸어도 달라지지 않는다.
$$
tr(AB) = tr(BA)
$$

5. 세 행렬의 곱의 대각합은 다음과 같이 순서를 순환시켜도 달라지지 않는다.(트레이스 트릭)
   - A,B,C가 각각 정방행렬일 필요는 없다. 최종적으로 대각합을 구하는 행렬만 정방행렬이면 된다.
$$
tr(ABC)=tr(BCA)=tr(CAB)\\
x^TAx = tr(x^TAx) = tr(Axx^T) = tr(xx^TA)
$$

### 행렬식

정방행렬 A의 행렬식은 det(A), detA, |A| 기호로 표기한다. 

행렬 A가 스칼라인 경우 행렬식은 자기 자신의 값이 된다.
$$
det([a]) = a
$$

행렬 A가 스칼라가 아닌 경우 여인수 전개(cofactor expansion)을 이용한다. 
$$
\det(A) = \sum_{i=1}^N \left\{ (-1)^{i+j_0}M_{i,j_0} \right\} a_{i,j_0} 
$$

$$
\det(A) = \sum_{j=1}^N \left\{ (-1)^{i_0+j} M_{i_0,j} \right\} a_{i_0,j} 
$$

$$M_{i,j}$$ 는 마이너(minor, 소행렬식) : 정방행렬 A에서 $$i$$ 행, $$j$$ 열을 지워서 얻어진(원래의 행렬보다 크기가 1만큼 작은) 행렬의 행렬식이다. 

$$(-1)^{i+j}M_{i,j}$$ 는 여인수(cofactor, 코펙터) 라고 한다. 
$$
C_{i,j} = (-1)^{i+j}M_{i,j}
$$

여인수 전개식
$$
\det(A) = \sum_{i=1}^N C_{i,j_0} a_{i,j_0}  =  \sum_{j=1}^N C_{i_0,j} a_{i_0,j}
$$

2 X 2 행렬의 행렬식
$$
\det \left( \begin{bmatrix}a&b\\c&d\end{bmatrix} \right) = ad-bc
$$

3 X 3 행렬의 행렬식
$$
\det \left( \begin{bmatrix}a&b&c\\d&e&f\\g&h&i\end{bmatrix} \right) =aei+bfg+cdh-ceg-bdi-afh
$$

행렬식도 음수가 될 수 있다.

#### 행렬식 성질

1. 전치행렬의 행렬식은 원래의 행렬의 행렬식과 같다.
$$
det(A^T) = det(A)
$$

2. 항등행렬의 행렬식은 1이다.
$$
det(I) = 1
$$

3. 두 행렬의 곱의 행렬식은 각 행렬의 행렬식의 곱과 같다.
$$
det(AB) = det(A)det(B)
$$

4. 역행렬 $$A^{-1}$$ 은 원래의 행렬 A와 다음 관계를 만족하는 정방행렬을 말한다. I는 항등행렬이다.
$$
A^{-1}A = AA^{-1} = I
$$

5. 역행렬의 행렬식은 원래의 행렬의 행렬식의 역수와 같다.
$$
det(A^{-1}) =  \dfrac{1}{\det(A)}  
$$


___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 