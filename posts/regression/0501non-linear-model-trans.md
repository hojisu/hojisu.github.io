<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 비선형 모형 변형

### 비선형 변형

독립변수 값을 제곱한 비선형 독립변수를 추가할 수 있다.

~~~python
# 보스턴 집값 LSTAT
model2 = sm.OLS.from_formula("MEDV ~ LSTAT + I(LSTAT**2)", data=df_boston)
result2 = model2.fit()
print(result2.summary())
~~~

### 범주형을 사용한 비선형성

독립변수의 비선형성을 포착하는 또 다른 방법 중 하나는 강제로 범주형 값으로 만드는 것이다. 범주형 값이 되면서 독립변수의 오차가 생기지만 이로 인한 오차보다 비선형성으로 얻을 수 있는 이익이 클 수도 있다.

~~~python
# 보스턴 집값 RM
rooms = np.arange(3, 10)
labels = [str(r) for r in rooms[:-1]]
df_boston["CAT_RM"] = np.round(df_boston.RM)

sns.barplot(x="CAT_RM", y="MEDV", data=df_boston)
plt.show()
~~~

### 시간 독립변수의 변형

파이썬 `datetime` 자료형은 `toordinal` 명령으로 특정 시점으로부터 경과한 시간의 일단위 값을 구하거나 `timestamp` 메서드로 초단위 값을 구할 수 있다. 시간 값의 경우 크기가 크므로 반드시 스케일링을 해 주어야 한다.

연도, 월, 일, 요일 데이터를 별도의 독립변수로 분리하거나 한 달 내에서 몇번째 날짜인지 월의 시작 또는 끝인지를 나타내는 값은 모두 특징값이 될 수 있다. 판다스에서는 `dt` 특수 연산자를 사용하여 이러한 값을 구할 수 있다.

### 주기성을 가지는 독립변수

$$
x \;\; \rightarrow
\begin{cases}
x_1 = \cos\left(\frac{2\pi}{360}x\right) \\
x_2 = \sin\left(\frac{2\pi}{360}x\right) 
\end{cases}
$$

~~~python
f = 2.0 * np.pi / 360

model8 = sm.OLS.from_formula("""
Hillshade_9am ~ 
np.cos(f * Aspect) +
np.sin(f * Aspect)
""", data=df_covtype
)
result8 = model8.fit()
print(result8.summary())
~~~

### 종속변수 변형

모형이 올바르다면 예측치와 실제 종속변수값을 그린 스캐터 플롯은 선형적인 모습이 나와야 한다. 하지만 실제로는 제곱근이나 로그 그래프와 더 유사하다. 이러한 경우에는 이 스캐터 플롯을 선형적으로 만들어 주도록 예측치를 비선형 변환한다. 여러가지 모형을 비교해보면 독립변수와 종속변수를 모두 로그 변환한 모형이 가장 좋다는 것을 알 수 있다.


________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 