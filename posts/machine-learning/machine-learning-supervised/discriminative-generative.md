<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

## discriminative model

discriminative model이란 데이터 $$X$$가 주어졌을 때 레이블 $$Y$$가 나타날 조건부확률 $$P(Y|X)$$를 직접적으로 반환하는 모델을 가리킵니다. 레이블 정보가 있어야 하기 때문에 지도학습(supervised learning) 범주에 속하며 X의 레이블을 잘 구분하는 결정경계(decision boundary)를 학습하는 것이 목표가 됩니다. discriminative model은 generative model에 비해 가정이 단순하고, 학습데이터 양이 충분하다면 좋은 성능을 내는 것으로 알려져 있습니다. 선형회귀와 로지스틱회귀는 disciminative model의 대표적인 예시입니다.

## generative model

generative model이란 데이터 $$X$$가 생성되는 과정을 두 개의 확률모형, 즉 $$P(Y)$$, $$P(X|Y)$$으로 정의하고, 베이즈룰을 사용해 $$P(Y|X)$$를 간접적으로 도출하는 모델을 가리킵니다. generative model은 레이블 정보가 있어도 되고, 없어도 구축할 수 있습니다. 전자를 지도학습기반의 generative model이라고 하며 선형판별분석이 대표적인 사례입니다. 후자는 비지도학습 기반의 generative model이라고 하며 가우시안믹스처모델, 토픽모델링이 대표적인 사례입니다.

generative model은 discriminative model에 비해 가정이 많습니다. 그 가정이 실제 현상과 맞지 않는다면 generative model의 성능은 discriminative model보다 성능이 좋지 않을 수 있지만, 가정이 잘 구축되어 있다면 이상치에도 강건하고 학습데이터가 적은 상황에서도 좋은 예측 성능을 보일 수 있습니다. generative model은 범주의 분포(distribution)을 학습하는 것이 목표가 됩니다. 또한 generative model은 일반적인 베이지안 추론과 마찬가지로 학습데이터가 많을 수록 discriminative model과 비슷한 성능으로 수렴하는 경향이 있다고 합니다. 아울러 generative model은 $$P(Y|X)$$을 구축하기 때문에 이 모델을 활용해 $$X$$를 샘플링할 수도 있습니다.

Reference
- https://ratsgo.github.io/generative%20model/2017/12/17/compare/