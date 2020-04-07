<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 부분회귀

**새로운 독립변수 그룹 $$X_2$$ 를 추가해서 다시 회귀분석을 한다면 기존 가중치 벡터의 값이 달라진다.**
단, 다음과 같은 경우에는 두가지 회귀분석의 결과가 같을 수 있다.
- $$w'_2=0$$ . 즉 $$X_2$$ 와 $$y$$ 의 상관관계가 없는 경우
- $$𝑋^𝑇_1𝑋_2=0$$. 즉 독립변수 $$𝑋_1$$과 독립변수 $$𝑋_2$$가 직교하는 경우. 독립변수 $$𝑋_1$$과 독립변수 $$𝑋_2$$이 서로 상관관계가 없으면 직교할 가능성이 높다.

### FWL(프리슈-워-로벨(Frisch–Waugh–Lovell)) 정리

1) 특정한 독립변수 그룹 $$X_1$$ 로 종속변수 $$y$$ 를 선형회귀분석하여 잔차 $$y^*$$ 를 구한다.

2) $$X_1$$ 로 다른 독립변수 $$x_2$$ 를 선형 회귀분석하여 나온 잔차 $$x_2^*$$ 를 구한다.

3) $$y^*$$ 를 종속변수로 하고 $$x_2^*$$ 를 독립변수로 하여 선형 회귀분석하여 구한 가중치는 $$X_1$$ 과 $$x_2$$ 를 모두 사용하여 $$y$$ 를 선형 회귀 분석하였을 때 $$x_2$$ 에 대한 가중치와 같다. 

### 평균 제거 데이터

**독립변수에서 평균을 제거한 데이터와 종속변수에서 평균을 제거한 데이터로 얻은 회귀분석 결과는 상수항을 포함하여 구한 회귀분석 결과와 같다.**

평균을 제거한 데이터를 사용하는 경우에는 독립변수에 상수항을 포함하지 않는다는 점에 주의한다.

### 부분회귀 플롯

**부분회귀 플롯(Partial Regression Plot)**은 **특정한 하나의 독립변수의 영향력을 시각화하는 방법**이다. 부분회귀 방법을 아래와 같다. 

- 특정한 독립변수 $$x_2$$ 를 제외한 나머지 독립변수 $$X_1$$ 들로 종속변수 $$y$$ 를 선형 회귀분석하여 잔차 $$y^*$$ 를 구한다.
- 특정한 독립변수 $$x_2$$ 를 제외한 나머지 독립변수 $$X_1$$ 들로 특정한 독립변수 $$x_2$$ 를 선형회귀분석하여 잔차 $$x_2^*$$ 를 구한다. 
- 잔차 $$x_2^$$를 독립변수로, 잔차 $$y^*$$ 를 종속변수로 하여 선형회귀분석한다. 

statsmodels 패키지의 `sm.graphics.plot_partregress` 명령을 쓰면 부분회귀 플롯을 그릴 수 있다. 이 때 다른 변수의 이름을 모두 지정해 주어야 한다.

  ```
  plot_partregress(endog, exog_i, exog_others, data=None, obs_labels=True, ret_coords=False)
  ```

  - `endog`: 종속변수 문자열
  - `exog_i`: 분석 대상이 되는 독립변수 문자열
  - `exog_others`: 나머지 독립변수 문자열의 리스트
  - `data`: 모든 데이터가 있는 데이터프레임
  - `obs_labels`: 데이터 라벨링 여부
  - `ret_coords`: 잔차 데이터 반환 여부

부분회귀 플롯에서 가로축의 값은 독립변수 자체의 값이 아니라 어떤 독립변수에서 다른 독립변수의 영향을 제거한 일종의 "순수한 독립변수 성분"을 뜻한다.

- `sm.graphics.plot_partregress_grid` 명령을 쓰면 전체 데이터에 대해 한번에 부분회귀 플롯을 그릴 수 있다.

```
plot_partregress_grid(result, fig)
```

` result`: 회귀분석 결과 객체

`fig`: `plt.figure` 객체

## CCPR 플롯

CCPR(Component-Component plus Residual)은 특정한 하나의 변수의 영향을 살펴보기 위한 것이다.

  - $$x_i$$를 가로축으로

  - $$w_ix_i+e$$을 세로축으로

statsmodels 패키지의 `sm.graphics.plot_ccpr` 명령으로 CCPR 플롯을 그릴 수 있다.

  ```
  plot_ccpr(result, exog_idx)
  ```

  - `result`: 회귀분석 결과 객체
  - `exog_idx`: 분석 대상이 되는 독립변수 문자열

CCPR 플롯에서는 부분회귀 플롯과 달리 독립변수가 원래의 값 그대로 나타난다.

마찬가지로 `sm.graphics.plot_ccpr_grid` 명령을 쓰면 전체 데이터에 대해 한번에 CCPR 플롯을 그릴 수 있다.

  `plot_ccpr_grid` 명령은 모든 독립변수에 대해 CCPR 플롯을 그려준다.

  ```
  plot_ccpr_grid(result, fig)
  ```

  - `result`: 회귀분석 결과 객체
  - `fig`: `plt.figure` 객체

~~~python
fig = plt.figure(figsize=(8, 15))
sm.graphics.plot_ccpr_grid(result_boston, fig=fig)
fig.suptitle("")
plt.show()
~~~

`plot_regress_exog` 명령은 부분회귀 플롯과 CCPR을 같이 보여준다.

  ```
  plot_regress_exog(result, exog_idx)
  ```

  - `result`: 회귀분석 결과 객체
  - `exog_idx`: 분석 대상이 되는 독립변수 문자열

