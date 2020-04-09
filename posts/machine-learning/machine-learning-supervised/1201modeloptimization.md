<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 모형최적화

머신 러닝 모형이 완성된 후에는 최적화 과정을 통해 예측 성능을 향상시킨다. 

### Scikit-Learn 의 모형 하이퍼 파라미터 튜닝 도구

- `validation_courve`  : 단일 하이퍼 파라미터 최적화
- `GridSearchCV`  : 그리드를 사용한 복수 하이퍼 파라미터 최적화
- `ParameterGrid`  : 복수 파라미터 최적화용 그리드

### `validation_curve`

최적화할 파라미터 이름과 범위, 그리고 성능 기준을 인수로 받아 파라미터 범위의 모든 경우에 대해 성능 기준을 계산한다. 

### `GridSearchCV`

모형 래퍼(Wrapper) 성격의 클래스이다. 

클래스 객체에 `fit` 메서드를 호출하면 grid search를 사용하여 자동으로 복수개의 내부 모형을 생성하고 이를 모두 실행시켜서 최적 파라미터를 찾아준다. 생성된 복수개와 내부 모형과 실행 결과는 다음 속성에 저장된다.
- `grid_scores_`
    - param_grid 의 모든 파리미터 조합에 대한 성능 결과. 각각의 원소는 다음 요소로 이루어진 튜플이다.
    - parameters: 사용된 파라미터
    - mean_validation_score: 교차 검증(cross-validation) 결과의 평균값
    - cv_validation_scores: 모든 교차 검증(cross-validation) 결과
- `best_score_` : 최고 점수
- `best_params` : 최고 점수를 낸 파라미터
- `best_estimator_` : 최고 점수를 낸 파라미터를 가진 모형

### `ParameterGrid`

scikit-learn이 제공하는 GridSearchCV 이외의 방법으로 그리드 탐색을 해야하는 경우도 있다. 이 경우 파라미터를 조합하여 탐색 그리드를 생성해 주는 명령어이다. 탐색을 위한 iterator 역할을 한다.

### 병렬 처리

`GridSearchCV` 명령에는 `n_jobs` 라는 인수가 있다. 디폴트 값은 1인데 이 값을 증가시키면 내부적으로 멀티 프로세스를 사용하여 그리드서치를 수행한다. 만약 CPU 코어의 수가 충분하다면 `n_jobs` 를 늘릴 수록 속도가 증가한다. 