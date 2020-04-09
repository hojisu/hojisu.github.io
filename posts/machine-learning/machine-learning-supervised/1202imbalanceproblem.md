<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 비대칭 데이터 문제 Imbalanced data problem

데이터 클래스 비율이 너무 차이가 나면(highly-imbalanced data) 단순히 우세한 클래스를 택하는 모형의 정확도가 높아지므로 모형의 성능판별이 어려워진다. 즉, 정확도(accuracy)가 높아도 데이터 갯수가 적은 클래스의 재현율(recall-rate)이 급격히 작아지는 현상이 발생할 수 있다. 각 클래스에 속한 데이터의 갯수의 차이에 의해 발생하는 문제들을 비대칭 데이터(imbalanced data problem)이다.

### 해결 방법

다수 클래스 데이터에서 일부만 사용하는 언더 샘플링이나 소수 클래스 데이터를 증가시키는 오버 샘플링 사용하여 데이터 비율을 맞추면 정밀도(precision)가 향상된다. 
- 오버샘플링(Over-sampling)
- 언더샘플링(Under-sampling)
- 복합샘플링(Combining Over-and Under-Sampling)

### imbalanced-learn 패키지

#### 언더샘플링

- Ramdom Under-Sampler : 무작위로 데이터를 없애는 단순 샘플링
- Tomek's link method
  - 토멕링크(Tomek's link)란 서로 다른 클래스에 속하는 한 쌍의 데이터($$x_+, x_-$$)로 서로에게 더 가까운 다른 데이터가 존재하지 않는 것이다. 
  - 즉 클래스가 다른 두 데이터가 아주 가까이 붙어있으면 토멕링크가 된다. 
  - 토멕링크를 찾은 다음 그 중에서 다수클래스에 혹하는 데이터를 제외하는 방법으로 경계선을 다수 클래스 쭉으로 밀어 붙이는 효과가 있다. 

- Condensed Nearest Neighbour (CNN)
  - CNN 방법은 1-NN 모형으로 분류되지 않는 데이터만 남기는 방법이다. 선택된 데이터 집합을 $$S$$ 라고 하자.
    - 소수 클래스 데이터를 모두 $$S$$ 에 포함시킨다. 
    - 다수 데이터 중에서 하나를 골라서 가장 가까운 데이터가 다수 클래스이면 포함시키지 않고 아니면 S에 포함시킨다.
    - 더이상 선택되는 데이터가 없을 때까지 반복한다. 
  - 기존에 선택된 데이터와 가까이 있으면서 같은 클래스인 데이터는 선택되지 않기 때문에 다수 데이터의 경우 선택되는 비율이 적어진다. 
- One Sided Selection
  - 토멕링크 방법과 Condensed Nearest Neighbour 방법을 섞은 것이다. 
    - 토멕링크 중 다수 클래스를 제외하고 나머지 데이터 중에서도 서로 붙어있는 다수 클래스 데이터는 1-NN 방법으로 제외한다. 
- Edited Nearest Neighbours (ENN)
  - 다수 클래스 데이터 중 가장 가까운 k(n_neighbors)개의 데이터가 모두(kind_sel='all')  또는 다수(kind_sel='mode') 클래스가 아니면 삭제하는 방법이다. 소수 클래스 주변의 다수 클래스 데이터는 사라진다.
- Neighbourhood Cleaning Rule
  - CNN + ENN

#### 오버 샘플링

- RandomOVerSampler : 소수 클래스의 데이터를 반복해서 넣는 것(replacement)이다. 가중치를 증가시키는 것과 비슷하다. 

- ADASYN(Adaptive Synthetic Sampling) : 소수 클래스 데이터와 그 데이터에서 가장 가까운 k개의 소수 클래스 데이터 중 무작위로 선택된 데이터 사이의 직선상에 가상의 소수 클래스 데이터를 만드는 방법이다.

- SMOTE(Synthetic Minority Over-sampling Technique) : ADASYN 방법처럼 데이터를 생성하지만 생성된 데이터를 무조건 소수 클래스라고 하지 않고 분류 모형에 따라 분류한다. 

#### 복합 샘플링

- SMOTE + ENN 
- SMOTE + Tomek

