# 이상 탐지 (Anomaly Detection)

이상 탐지(Anomaly Detection)이란 예상되거나 기대하는 관찰값, item, event가 아닌 데이터 패턴(이상징후, anomalies), 개체(이상값) 등을 찾아내는 것을 말한다. 

이상 탐지 기법은 특정 알고리즘이 있는 것이 아니라 "기대하는 결과"를 얻기 위해 여러 가지 알고리즘과 분석론을 활용한 분석을 의미한다. 
어떤 데이터인지, 어떤 분야인지, 목적은 무엇인지 등에 따라 매우 광범위하고 다양한 방법들이 활용되고 있기 때문에 정해진 기법이나 모델을 사용하는 게 아니라 활용 가능한 모든 방안들을 사용할 수 있다. 


### 이상 탐지 목적
- Chance Discovery의 목적
  - 예 : 새로운 고객 층의 발견, 보험 상품의 개발, segmentation이 필요할 때
- Fault Discovery의 목적
  - 예 : 불량율이 증가하는 장비, 공정 탐지 등
- Novelty Detection : 새로운 것을 찾기 위해
  - 예 : 보안 분야의 새로운 침입 패턴 탐지
- Noise Removal 노이즈 제거
  - 예 : 잘못 입력된 값이나 대표성이 없는 극단값의 자동 제거

### 이상 탐지 이슈
- 정상과 이상 상태를 명확하기 정의하기 어렵다. 이상 패턴은 정상 패턴과 매우 유사한 패턴을 보일 수 있다.
- 알려진 이상값, 이상 패턴과 전혀 다른 이상값, 이상 패턴이 발생할 수 있다.
- 양질의 training/validation 데이터 셋을 확보하기가 어렵다.
- 정상 패턴도 시간, 상황에 따라 변한다.
- 이상 탐지의 대상 변수가 너무 많고 데이터의 양도 방대하다.


### Anomaly Detection 기법 

#### 기계학습 기반의 이상 탐지

**군집(Clustering)기반 이상 탐지 기법** 
군집화는 기계 학습 방법 중 비지도학습이다. 


