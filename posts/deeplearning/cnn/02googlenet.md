<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# GoogLeNet

### Summary

- Inception module은 노드간의 연결을 줄이면서(sparse connectivity) 행렬연산은 Dense 연산을 하도록 처리한 것이다.
__________

convolution 연산으로 Densely하게 연결되어 있는 것을 높은 관련성(correlation)을 가진 노드끼리만 연결하도록(연결을 sparse하도록) 바꾸면 연산량과 파라미터 수가 줄고 overfitting 또한 개선될 것이라는 생각에서 출발하였다. Fully connected network에서 사용되는 Dropout과 비슷한 기능을 할 것이라고 본 것이다.  요약하면 어떻게 노드 간의 연결을 줄이면서(sparse connectivity) 행렬 연산은 Dense 연산을 하도록 처리하도록 하는 것이다. 이 고민의 결과가 Inception module이다.

### Inception module

![image-20200405185318341](../../../resource/img/image-20200405185318341.png)

위의 그림에서 알 수 있듯이 입력값에 대해 4가지 종류의 연산을 수행하고 4개의 결과를 채널방향으로 합친다. 이러한 Inception module이 모델에 총 9개가 있다. 

##### Inception module의 4가지 연산

- 1x1 convolution
- 1x1 convolution -> 3x3 convolution
- 1x1 convolution -> 5x5 convolution
- 3x3 maxpooling -> 1x1 convolution
- 위의 결과를 Channel-wise concat(feature map을 쌓는 것)

1x1 convolution 연산은 Inception module에서 핵심 역할을 한다. 

- 첫번째 기능은 채널의 수를 조절하는 기능이다. 채널 간의 Correlation을 연산한다는 의미이다. 기존의 convolution연산은 3x3 커널을 이용해 연산할 경우 3x3크기의 지역정보 뿐 아니라 커널 간의 정보 또한 같이 고려하여 하나의 값으로 나타내었다. 즉 하나의 커널이 2가지 역할(지역정보, 커널간 정보)을 모두 수행하였다. 

  3x3 커널 이전에 1x1 convolution을 사용한다면 1x1은 채널을 조절하는 역할을 하기 때문에 최적화 과정에서 채널 간의 특징을 추출할 것이고 3x3은 이미지의 지역정보에만 집중하여 특징을 추출하려 할 것이다. 즉 역할을 세분화 해준 것이다. 

  1x1 convolution에 사용되는 파라미터들 끼리, 이미지의 지역 정보는 3x3 convolution에 사용되는 파라미터들 끼리 연결된다는 점에서 노드 간의 연결을 줄였다고 볼 수 있다.

- 두번째는 1x1 convolution 연산으로 이미지의 채널을 줄여준다면, 3x3, 5x5 convolution레이어에서의 파라미터 개수를 절약할 수 있다. 그래서 망을 기존의 CNN 구조보다 깊게 만들어도 파라미터가 크지 않다. 

### Inception V2 & V3

Inception V2와 Inception V3은 GoogLeNet을 발전시킨 형태로 다음 두 가지 문제를 정의하고 이에 대한 해결책을 제시했다.

1. representation bottleneck은 신경망 안에서 차원을 과도하게 줄일경우 정보의 양이 크게 줄어드는 현상이다.
2. factorization은 기존의 convolution 연산에서 사용하는 커널을 개선하면 연산의 복잡도가 줄어들 수 있다는 것이다. 

해결책은 다음과 같다.

##### factorization

기존의 5x5 convolution은 두 번의 3x3 convolution 연산으로 교체하였다. 

![image-20200409175900964](../../../resource/img/image-20200409175900964.png)

그리고 기존에 3x3 convolution은 1x3, 3x1 convolution 연산으로 바꾸어 연산의 복잡도를 줄어주었다. 예를 들어, (3,3,3) 형태의 이미지에 대해 convolution 연산을 한다면, 3x3 커널의 경우, 입출력을 같게 하기위해 padding을 사용한 경우, 3x3x3의 파라미터를 사용해 9번의 연산을 해야한다. 곱 연산만 고려했을 때, 243번의 곱을 하게된다. 같은 경우에 1x3, 3x1의 커널을 사용하면 각각 3x3의 파라미터를 사용하고 9번의 연산을 해야한다. 총 162번의 곱을 한다. 

![image-20200409180306919](../../../resource/img/image-20200409180306919.png)

##### representaion bottleneck

pooling과 convolution을 병렬적으로 수행하고 합친다. 

![image-20200409180421532](../../../resource/img/image-20200409180421532.png)

마지막으로 Average pooing 전의 Inception module을 더 넓게 형성 했다. 차원을 줄이면서 발생하는 정보 손실을 막기위한 방법이다. 같은 입력값에 대하여 다양한 연산을 하도록 했다.





Reference
- https://datascienceschool.net/