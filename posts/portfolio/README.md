# 포트폴리오


#### 회귀분석 프로젝트
https://github.com/hojisu/sberbank-russian-housing-market


캐글에서 러시아 집 값 예측 문제를 선택하였습니다. 데이터는 300개의 독립변수로 이루어진 3만건의 주택거래 내역을 사용하였습니다. 독립변수 수가 많고 다양하기 때문에 변수변환과 결측치를 처리하고 OLS로 회귀하고 성능분석을 하면서 차원축소 및 정규화를 진행하였습니다. 회귀분석모형 진단은 잔차 정규성 테스트와 부분회귀플롯, 교차검증을 통해 성능을 확인하였습니다. 성능 매트릭스는 RMSLE를 사용하였습니다. 정규화 적용된 Ordinary Least Square, XGBoost 모델을 사용하여 퍼포먼스를 비교하였습니다.

#### 머신러닝 프로젝트
https://github.com/hojisu/recommendation-project


추천시스템 프로젝트로 자료는 제6회 L.POINT Big Data Competition에서 제공받았습니다. 데이터는 사용자, 아이템, 구매수량으로 implicit data이며 sparse matirx 형태입니다. 사용자와 아이템을 Latent Factor로 분해하고 이를 각각 학습시키는 Matrix Factorization기법을 사용하였습니다. 오차함수를 최소화하는 latent vector를 찾기 위해 ALS 알고리즘을 이용하여 최적화 하였습니다.

#### 딥러닝 프로젝트
https://github.com/hojisu/phishing-url-classification


딥러닝 프로젝트는 15000건의 실제 피싱 URL과 45000건의 정상 URL을 사용하여 피싱 URL 탐지모델을 구현하였습니다. 탐지모델에는 컨볼루션 오토인코더에 기반하여 문자수준 URL 정상 모형을 구축하고 재구축 오류에 기반한 정상 및 피싱 URL 분류용 컨볼루션 신경망을 모델링하였습니다. 딥러닝 및 머신러닝 알고리즘을 10겹 교차검증하여 최고 정확도를 달성하였습니다.


