<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 행렬의 미분

### Summary

- 그레디언트(gradient)는 스칼라를 벡터로 미분한 값이다. 결과를 열벡터로 표시한다.
- 선형 모형을 미분하면 그레디언트 벡터는 가중치 벡터이다. $$\nabla f = \frac{\partial {w}^{T}{x}}{\partial {x}} = \frac{\partial {x}^{T}{w}}{\partial {x}} = {w}$$
- 이차형식을 미분하면 행렬과 벡터의 곱으로 나타난다. $$\nabla f(x) = \frac{\partial {x}^{T}{A}{x}}{\partial {x}} = ({A} + {A}^{T}){x}$$
- 행렬 $$A$$ 와 벡터 $$x$$ 의 곱 $$Ax$$ 를 벡터 $$x$$로 미분하면 행렬 $$A^T$$ 가 된다. $$\nabla f(x) = \dfrac{\partial ({Ax})}{\partial {x}} = A^T$$
- 자코비안 행렬은 함수의 출력변수와 입력변수 모두 벡터인 경우 입력변수 각각 출력변수 각각의 조합에 대해 미분이 존재하고 행렬 형태가 된다. 이렇게 만들어진 도함수 행렬이다. 벡터함수를 벡터변수로 미분해서 생기는 행렬의 전치행렬이다. 
- 다변수 함수의 2차 도함수는 그레디언트 벡터를 입력변수 벡터로 미분한 것으로 헤시안 행렬(Hessian matirx) 이라고 한다. 헤시안 행렬은 그레디언트 벡터의 자코비안 행렬의 전치 행렬로 정의한다. 
- 두 정방행렬을 곱해서 만들어진 행렬의 대각성분(trace)는 스칼라이다. 이 스칼라를 뒤의 행렬로 미분하면 앞의 행렬의 전치행렬이 나온다. $$\dfrac{\partial f}{\partial X} =
  \dfrac{\partial \text{tr} ({W}{X})}{\partial {X}} = {W}^T$$
- 행렬식(determinant)은 스칼라 값이고 이 값의 로그 값도 스칼라이다. 이 값을 원래의 행렬로 미분하면 원래 행렬의 역행렬의 전치 행렬이 된다. $$\dfrac{\partial f}{\partial X} = \dfrac{\partial \log | {X} | }{\partial {X}} = ({X}^{-1})^T$$

______________

여려개의 입력을 가지는 다변수 함수는 함수의 독립변수가 벡터인 경우로 볼 수 있다.
$$
f\left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = f(x) = f(x_1, x_2)\\
\text{벡터 } x \;\; \rightarrow \;\; \text{스칼라 } f
$$

이를 확장하면 행렬을 입력으로 가지는 함수도 생각 할 수 있다.
$$
f\left( \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix} \right) 
= f(X) = f(x_{11}, \cdots, x_{22}) \\
\text{행렬 } x \;\; \rightarrow \;\; \text{스칼라 } f
$$

벡터나 행렬을 출력하는 함수는 여러개의 함수를 합쳐 놓은 것이다. 
$$
f(x) =
\begin{bmatrix}
f_1(x) \\
f_2(x)
\end{bmatrix} \\
\text{스칼라 } x \;\; \rightarrow \;\; \text{벡터 } f\\
f(x) =
\begin{bmatrix}
f_{11}(x) & f_{12}(x) \\
f_{21}(x) & f_{22}(x)
\end{bmatrix} \\
\text{스칼라 } x \;\; \rightarrow \;\; \text{행렬 } f
$$

벡터나 행렬을 입력받아 벡터나 행렬을 출력할 수도 있다.
$$
f(x) =
\begin{bmatrix}
f_1(x_1, x_2) \\
f_2(x_1, x_2)
\end{bmatrix} \\
\text{벡터 } x \;\; \rightarrow \;\; \text{벡터 } f \\
f(x) =
\begin{bmatrix}
f_{11}(x_1, x_2) & f_{12}(x_1, x_2) \\
f_{21}(x_1, x_2) & f_{22}(x_1, x_2)
\end{bmatrix} \\
\text{벡터 } x \;\; \rightarrow \;\; \text{행렬 } f
$$

이러한 행렬을 입력이나 출력으로 가지는 함수를 미분하는 것을 **행렬미분(matrix differentiation)** 이라고 한다.

사실 행렬미분은 정확하게는 미분이 아닌 편미분(partial derivative)이지만 여기에서는 편의상 미분이라고 쓰겠다. 또한 행렬미분에는 분자중심 표현법(Numerator-layout notation)과 분모중심 표현법(Denominator-layout notation) 두 가지가 있는데 여기에서는 분모중심 표현법으로 서술한다.

### 스칼라를 벡터로 미분하는 경우

데이터 분석에서는 함수의 출력변수가 스칼라이고 입력변수 𝑥가 벡터인 다변수 함수를 사용하는 경우가 많다. 따라서 편미분도 $$\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \cdots$$ 등으로 여러 개가 존재한다.

스칼라를 벡터로 미분하는 경우에는 **결과를 열벡터**로 표시한다. 이렇게 만들어진 벡터를 **그레디언트 벡터(gradient vector)**라고 하고 ∇𝑓로 표기한다.
$$
\nabla f = 
\frac{\partial f}{\partial {x}} =
\begin{bmatrix}
\dfrac{\partial f}{\partial x_1}\\
\dfrac{\partial f}{\partial x_2}\\
\vdots\\
\dfrac{\partial f}{\partial x_N}\\
\end{bmatrix}
$$

예시 $$f(x, y) = 2x^2 + 6xy + 7y^2 - 26x - 54y + 107$$  에 대한 그레디언트 벡터를 구하면 
$$
\nabla f = 
\begin{bmatrix}
\dfrac{\partial f}{\partial x}\\
\dfrac{\partial f}{\partial y}\\
\end{bmatrix} =
\begin{bmatrix}
4x + 6y - 26\\
6x + 14y - 54\\
\end{bmatrix}
$$

그레디언트 기호를 이용하면 다변수 함수의 테일러전개를 다음처럼 간단하게 표시할 수 있다.
$$
f(x) \approx f(x_0) + \nabla f(x_0) (x - x_0)
$$

2차원의 경우를 예로 들어 그레디언트 벡터를 표시하는 법을 알자보자.

2개의 입력변수를 가지는 2차원 함수 𝑓(𝑥,𝑦)는 평면상에서 **컨투어(contour)플롯**으로 나타낼 수 있다. 그리고 입력 변수 𝑥,𝑦 위치에서의 그레디언트 벡터 $$\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}$$는 그 위치를 원점으로 하는 화살표로 표현할 수 있다. 그리고 그레디언트 벡터의 방향은 편미분 성분 $$\frac{\partial f}{\partial x}$$와 $$ \frac{\partial f}{\partial y}$$의 부호에 의해 결정된다. 만약 어떤 위치 𝑥,𝑦에서 𝑥가 증가할수록 𝑓가 커지면 도함수 $$\frac{\partial f}{\partial x}$$은 양수이다. 반대로 𝑦가 증가할수록 𝑓가 커지면 도함수 $$\frac{\partial f}{\partial x}$$은 음수이다. 벡터는 2차원 평면에서 화살표로 나타낼 수 있다. 가로 성분이 양수이고 세로 성분이 음수인 화살표는 우측 아래를 가리키는 화살이 될 것이다. 컨투어 플롯 위에 그레디언트 벡터를 화살표로 나타낸 것을 플롯을 **퀴버(quiver)플롯**이라고 한다. 퀴버플롯에서 화살표는 화살표 시작 지점의 그레디언트 벡터를 나타낸다.

#### 퀴버플롯에서 그레디언트 벡터의 특징

그레디언트 벡터의 크기는 기울기를 의미한다. 즉 벡터의 크가기 클수록 함수 곡면의 기울기가 커진다.  그레디언트 벡터의 방향은 함수 곡면의 기울기가 가장 큰 방향, 즉 단위 길이당 함수값(높이)이 가장 크게 증가하는 방향을 가리킨다. 그레디언트 벡터의 방향은 등고선(isoline)의 방향과 직교한다.

어떤 점 𝑥0에서 다른 점 𝑥로 이동하면서 함수 값이 얼마나 변하는지는 테일러 전개를 써서 근사할 수 있다.
$$
f(x) - f(x_0) = \Delta f \approx  \nabla f(x_0)^T (x - x_0)
$$

변화의 방향 $$𝑥−𝑥_0$$가 그레디언트 벡터와 같은 방향일 때 Δ𝑓가 가장 커지는 것을 알 수 있다. 등고선은 𝑓(𝑥)의 값이 일 정한 𝑥의 집합이므로 다음과 같은 방정식으로 표현할 수 있다.
$$
f(x) = f(x_0) \\
f(x) - f(x_0) = 0
$$
같은 등고선 위의 다른 점 $$𝑥_1$$를 향해 움직이는 등고선 방향의 움직임은 $$𝑥_1−𝑥_0$$이고 $$𝑥_0$$, $$𝑥_1$$ 모두 같은 등고선 위의 점이므로 $$𝑓(𝑥_0)$$=$$𝑓(𝑥_1)$$이다. 따라서 테일러 전개로부터 $$\nabla f(x_0)^T (x_1 - x_0) = f(x_1) - f(x_0) = 0$$ 등고선 방향 $$𝑥_1−𝑥_0$$과 $$∇𝑓(𝑥_0)$$이 직교한다는 것을 알 수 있다.



### 행렬 미분 법칙

다변수 함수를 미분하여 그레디언트 벡터를 구할 때는 다음 두가지 법칙이 유용하게 쓰인다. 

#### 행렬 미분 법칙 1: 선형모형

**선형 모형을 미분하면 그레디언트 벡터는 가중치 벡터이다.**
$$
f(x) = w^T x \\
\nabla f = \frac{\partial {w}^{T}{x}}{\partial {x}} = \frac{\partial {x}^{T}{w}}{\partial {x}} = {w}
$$

#### 행렬미분법칙 2: 이차형식

**이차형식을 미분하면 행렬과 벡터의 곱으로 나타난다. **
$$
f(x) = x^T A x \\
\nabla f(x) = \frac{\partial {x}^{T}{A}{x}}{\partial {x}} = ({A} + {A}^{T}){x}
$$

##### 벡터를 스칼라로 미분하는 경우

벡터 $${f}(x) = \begin{bmatrix} f_1 \\ f_2 \\ \vdots\\ f_M \\ \end{bmatrix}$$  를 스칼라 $$x$$ 로 미분하는 경우에는 결과를 **행벡터** 로 표시한다. 
$$
\frac{\partial {f}}{\partial x} = 
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x} &
\dfrac{\partial f_2}{\partial x} &
\cdots &
\dfrac{\partial f_M}{\partial x}
\end{bmatrix}
$$

##### 벡터를 벡터로 미분하는 경우

벡터를 벡터로 미분하면 미분을 당하는 벡터의 원소가 여러개(𝑖=1,…,𝑁)이고 미분을 하는 벡터의 원소도 여러개(𝑗=1,…,𝑀)이므로 미분의 결과로 나온 도함수는 2차원 배열 즉, 행렬이 된다.
$$
\dfrac{\partial {f}}{\partial {x}}
= 
\begin{bmatrix}
\dfrac{\partial f_1}{\partial {x}} &
\dfrac{\partial f_2}{\partial {x}} &
\cdots                             & 
\dfrac{\partial f_N}{\partial {x}}
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial {f}}{\partial x_1} \\
\dfrac{\partial {f}}{\partial x_2} \\
\vdots                             \\
\dfrac{\partial {f}}{\partial x_M}
\end{bmatrix}
= 
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_2}{\partial x_1} & \cdots & \dfrac{\partial f_N}{\partial x_1} \\
\dfrac{\partial f_1}{\partial x_2} & \dfrac{\partial f_2}{\partial x_2} & \cdots & \dfrac{\partial f_N}{\partial x_2} \\
\vdots & \vdots & \ddots & \vdots \\
\dfrac{\partial f_1}{\partial x_M} & \dfrac{\partial f_2}{\partial x_M} & \cdots & \dfrac{\partial f_N}{\partial x_M} \\
\end{bmatrix}
$$

#### 행렬미분법칙 3: 행렬과 벡터의 곱의 미분

**행렬 $$A$$ 와 벡터 $$x$$ 의 곱 $$Ax$$ 를 벡터 $$x$$로 미분하면 행렬 $$A^T$$ 가 된다.**
$$
f(x) = Ax \\
\nabla f(x) = \dfrac{\partial ({Ax})}{\partial {x}} = A^T
$$

함수의 출력변수와 입력변수가 모두 벡터(다차원) 데이터인 경우에는 입력변수 각각과 출력변수 각각의 조합에 대해 모두 미분이 존재한다. 따라서 도함수는 **행렬** 형태가 된다. 이렇게 만들어진 도함수의 행렬을 **자코비안 행렬(Jacobian matrix)** 이라고 한다. 자코비안 행렬은 벡터함수를 벡터변수로 미분해서 생기는 행렬의 **전치행렬** 이다. 따라서 행/열의 방향이 다르다는 점에 유의한다. 
$$
Jf(x) = J = \left(\frac{\partial  f}{\partial  x}\right)^T = 
\begin{bmatrix}
\left(\dfrac{\partial f_1}{\partial x}\right)^T \\ \vdots \\ \left(\dfrac{\partial f_M}{\partial x}\right)^T 
\end{bmatrix} =
\begin{bmatrix}
\nabla f_1^T \\  \vdots \\ \nabla f_M^T \\ 
\end{bmatrix} =
\begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_N}\\
\vdots & \ddots & \vdots\\
\dfrac{\partial f_M}{\partial x_1} & \cdots & \dfrac{\partial f_M}{\partial x_N} 
\end{bmatrix}
$$

다변수 함수의 2차 도함수는 그레디언트 벡터를 입력변수 벡터로 미분한 것으로 **헤시안 행렬(Hessian matirx)** 이라고 한다. 헤시안 행렬은 그레디언트 벡터의 자코비안 행렬의 전치 행렬로 정의한다. 
$$
Hf(x) = H = J(\nabla f(x))^T
$$

$$
H_{ij} = \dfrac{\partial^2 f}{\partial x_i\,\partial x_j}
$$

$$
H = \begin{bmatrix}
  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_N} \\
  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_N} \\
  \vdots & \vdots & \ddots & \vdots \\
  \dfrac{\partial^2 f}{\partial x_N\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_N\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_N^2}
\end{bmatrix}
$$

**함수가 연속이고 미분가능한 함수라면 헤시안 행렬은 대칭행렬이 된다**

벡터 -> 스칼라 

- 1차 미분 : 그레디언트 벡터
- 2차 미분 : 해시안 행렬

##### 스칼라를 행렬로 미분

출력변수 $$f$$ 가 스칼라값이고 입력변수 $$X$$ 가 행렬인 경우에는 도함수 행렬의 모양이 입력변수 행렬 $$X$$ 와 같다. 
$$
\dfrac{\partial f}{\partial {X}} =
\begin{bmatrix}
\dfrac{\partial f}{\partial x_{1,1}} & \dfrac{\partial f}{\partial x_{1,2}} & \cdots & \dfrac{\partial f}{\partial x_{1,N}}\\
\dfrac{\partial f}{\partial x_{2,1}} & \dfrac{\partial f}{\partial x_{2,2}} & \cdots & \dfrac{\partial f}{\partial x_{2,N}}\\
\vdots & \vdots & \ddots & \vdots\\
\dfrac{\partial f}{\partial x_{M,1}} & \dfrac{\partial f}{\partial x_{M,2}} & \cdots & \dfrac{\partial f}{\partial x_{M,N}}\\
\end{bmatrix}
$$

#### 행렬미분법칙 4: 행렬 곱의 대각성분

**두 정방행렬을 곱해서 만들어진 행렬의 대각성분(trace)는 스칼라이다. 이 스칼라를 뒤의 행렬로 미분하면 앞의 행렬의 전치행렬이 나온다.** 
$$
f(X) = \text{tr} ({W}{X}) \\
({W} \in {R}^{N \times N}, {X} \in {R}^{N \times N}) \\
\dfrac{\partial f}{\partial X} =
\dfrac{\partial \text{tr} ({W}{X})}{\partial {X}} = {W}^T
$$

#### 행렬미분법칙 5 : 행렬식의 로그

**행렬식(determinant)은 스칼라 값이고 이 값의 로그 값도 스칼라이다. 이 값을 원래의 행렬로 미분하면 원래 행렬의 역행렬의 전치 행렬이 된다. **
$$
f(X) = \log | {X} | \\
\dfrac{\partial f}{\partial X} = \dfrac{\partial \log | {X} | }{\partial {X}} = ({X}^{-1})^T
$$

- (증명) 역행렬 계산 참고

$$
\begin{align}
A^{-1} = \dfrac{1}{\det (A)} C^T = \dfrac{1}{\det (A)} 
\begin{bmatrix}
C_{1,1} & \cdots & C_{N,1}  \\
\vdots  & \ddots & \vdots   \\
C_{1,N} & \cdots & C_{N,N}  \\
\end{bmatrix}
\tag{2.4.10}
\end{align}
$$

 행렬식의 정의에서
$$
\dfrac{\partial}{\partial x_{i,j}} \vert X \vert = C_{i,j}
$$
행렬식와 역행렬의 관계에서
$$
\dfrac{\partial}{\partial X} \vert X \vert = C = | X | (X^{-1})^T
$$
로그 함수 공식에 대입하면
$$
\dfrac{d}{dx} \log f(x) = \dfrac{f'(x)}{f(x)} = \dfrac{\vert X \vert (X^{-1})^T}{\vert X \vert} = (X^{-1})^T
$$

___________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 