<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 특징 선택

### 분산에 의한 선택

예측모형에서 중요한 특징데이터란 종속데이터와의 상관관계가 크고 예측에 도움이 되는 데이터를 말한다. 

상관관계 계산에 앞서 특징 데이터의 값 자체가 표본에 따라 그다지 변하지 않는다면 종속데이터 예측에도 도움이 되지 않을 가능성이 높다. 표본 변화에 따른 데이터 값의 변화 즉, 분산이 기준치보다 낮은 특징 데이터는 사용하지 않는 방법이 분산에 의한 선택 방법이다. 

예를 들어 종속데이터와 특징데이터가 모두 0 또는 1 두가지 값만 가지는데 종속데이터는 0과 1이 균형을 이루는데 반해 특징데이터가 대부분(예를 들어 90%)의 값이 0이라면 이 특징데이터는 분류에 도움이 되지 않을 가능성이 높다.

하지만 분산에 의한 선택은 반드시 상관관계와 일치한다는 보장이 없기 때문에 신중하게 사용해야 한다.

### 단일 변수 선택

각각의 독립변수를 하나만 사용한 예측모형의 성능을 이용하여 가장 분류성능 혹은 상관관계가 높은 변수만 선택하는 방법이다. 
- `chi2`: 카이제곱 검정 통계값
- `f_classif`: 분산분석(ANOVA) F검정 통계값
- `mutual_info_classif`: 상호정보량(mutual information)

하지만 단일 변수의 성능이 높은 특징만 모았을 때 전체 성능이 반드시 향상된다는 보장은 없다.

### 다른 모형을 이용한 특성 중요도 계산

특성 중요도(feature importance)를 계산할 수 있는 랜덤포레스트 등의 다른 모형을 사용하여 일단 특성을 선택하고 최종 분류는 다른 모형을 사용할 수도 있다.
