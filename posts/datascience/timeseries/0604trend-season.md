# 추세와 계절성

### 결정론적 추세(trend)

결정론적 추세는 확률 과정의 기대값이 시간 t에 대한 함수로 표현 될 수 있는 것이다.

$$
\mu_t = \text{E}[Y_t] = f(t)
$$

### 결정론적 추세 추정

추세추정(trend estimation)은 확률 과정의 결정론적 기댓값 함수를 알아내는 것이다.

확률 과정 $$Y_t$$ 이 일반적인 비정상 과정이 아니라 추정이 가능한 **결정론적 추세 함수 $$f(t)$$와 정상 확률 과정 $$X_t$$ 의 합** 으로 표현될 수 있다.

$$
Y_t = f(t) + X_t
$$

### 다항식 추세

다항식 추세는 추세 함수 즉, 확률 과정의 기댓값을 시간에 대한 다항식으로 나타낼 수 있다고 가정하는 것이다.

~~~python
# 예시
data = sm.datasets.get_rdataset("CO2", package="datasets")
df = data.data

def yearfraction2datetime(yearfraction, startyear=0):
    import datetime
    import dateutil
    year = int(yearfraction) + startyear
    month = int(round(12 * (yearfraction - year)))
    delta = dateutil.relativedelta.relativedelta(months=month)
    date = datetime.datetime(year, 1, 1) + delta
    return date

df["datetime"] = df.time.map(yearfraction2datetime)
df["month"] = df.datetime.dt.month

result = sm.OLS.from_formula("value ~ time", data=df).fit()
print(result.summary())

# 만약 추세가 2차 함수 형태라면
result2 = sm.OLS.from_formula("value ~ time + I(time ** 2)", data=df).fit()
print(result2.summary())

trend2 = result2.params[0] + result2.params[1] * t + result2.params[2] * t**2
plt.plot(t, y, '-', t, trend2, '-')
plt.title("CO2 농도 시계열과 그에 대한 추세 함수")
plt.show()

# 구한 결정론적 추세 모형을 이용하면 1998년 1월의 CO2 농도는 다음과 같이 예측할 수 있다.
t_test = 1998 + 1 / 12
X_test = pd.DataFrame([[t_test]], columns=["time"])
result2.predict(X_test)

trend2 = result2.params[0] + result2.params[1] * t + result2.params[2] * t**2
plt.plot(t[-30:], y[-30:], 'o-', t[-30:], trend2[-30:], '-')
plt.plot(t_test, result2.predict(X_test).values[0], 'o', ms=10, lw=5)
plt.title("CO2 농도 시계열과 1998년 1월 예측치")
plt.show()
~~~

### 계절성 추세

계절성 추세는 특정한 달(month)이나 요일(day of week)에 따라 기댓값이 달라지는 것을 말한다. 이는 달 이름이나 요일 이름을 카테고리(category) 값으로 사용하여 회귀분석하여 추정할 수 있다.