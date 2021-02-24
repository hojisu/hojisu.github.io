<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 회귀분석모형의 진단(diagnosis)과 수정

회귀분석에 사용된 모형 가정을 제대로 만족하고 있는지를 확인하는 과정이다.

### 잔차 정규성

데이터가 모형 가정을 만족하면 분석결과로 나온 잔차는 **정규분포** 를 따라야 한다.

~~~python
# QQ플롯
sp.stats.probplot(result2.resid, plot=plt)
plt.show()
~~~

~~~python
# 정규성 검정
test = sm.stats.omni_normtest(result2.resid)
for xi in zip(['Chi^2', 'P-value'], test):
    print("%-12s: %6.3f" % xi)
~~~

### 잔차와 독립 변수의 관계

데이터가 모형 가정을 따르지 않지만 잔차는 정규 분포를 따를 때는 잔차와 독립 변수간의 관계를 살펴보는 것이 도움이 될 수 있다. 
- 데이터가 올바른 모형으로 분석되었다면 잔차는 더이상 독립 변수와 상관관계를 가지지 않아야 한다. 
- 잔차와 특정 독립 변수간의 관계는 전체 모형이 올바른 모형이 아니라는 것을 알려줄 뿐이지 어떤 모형이 올바른 모형인지에 대한 정보는 주지 않는다.

~~~python
plt.plot(x3, result3.resid, 'o')
plt.axhline(y=0, c='k')
plt.xlabel("X1")
plt.ylabel("Residual")
plt.show()
~~~

### 이분산성

선형 회귀 모형에서는 종속 변수 값의 분산이 독립 변수의 값과 상관없이 고정된 값을 가져야 한다. 

실제 데이터는 독립 변수 값의 크기가 커지면 종속 변수 값의 분산도 커지는 **이분산성(heteroskedasit)** 문제가 발생한다. 종속변수를 로그변환한 트랜스로그(translog) 모형을 사용하면 이분산성 문제가 해결되는 경우도 있다. 

### 자기 상관 계수

선형 회귀 모형에서는 오차(disturbance)들이 서로 (모수-조건부) 독립이라고 가정하고 있다. 그래서 잔차(residual)도 서로 독립이어야 한다. 만약 서로 독립이 아니라면 선형회귀 모형이 아닌 ARMA 모형 등의 시계열 모형을 사용해야 한다.

오차가 독립인지 검정하는 방법은 잔차를 시계열로 가정하여 자기상관계수를 구하는 것이다.
- 만약 독립이라면 시차(lag) = 0인 경우를 제외하고는 자기상관계수 $$\rho_l$$ = 0 이어야 한다. 
- 검사하는 검증은 Box-Pierce 검정, Ljung-Box 검정, Durbin-Watson 검정이 있다. 
  
________________________________
###### Reference
김도형 박사님 강의를 수강하며 데이터사이언티스트스쿨(https://datascienceschool.net/) 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 