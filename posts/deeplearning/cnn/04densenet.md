<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# DenseNet

### Summary

- 입력값을 계속해서 출력값 방향으로 합쳐주는 알고리즘이다. 
- 이미지에서 저수준의 특징들이 잘 보존되고, gradient가 수월하게 흘러 gradient vanishing 문제가 발생하지 않으며 깊이에 비해 파라미터 수가 적기에 연산량이 절약됨과 동시에 적은 데이터셋에서도 비교적 잘 학습이 된다는 점이다. 
- DenseNet은 Dense connectivity로 입력값을 계속해서 출력값의 방향으로 합쳐주기 때문에 최초의 정보가 비교적 온전히 남아있다. 
- 이전 레이어의 특징맵을 계속해서 다음 레이어의 입력과 연결하는데 이때 연결을 더하기가 아닌 이어붙이는 방식이다. 
______________

### Dense connectivity

![image-20200324151511467](../../../resource/img/image-20200324151511467.png)

DenseNet의 핵심은 Dense connectivity이다. Dense connectivity란 입력값을 계속해서 출력값의 채널 방향으로 합쳐주는 것(Concat)이다. 이를 ResNet과 수식으로 비교하면 다음과 같다.

$$
x_{l+1} = F(x_l) + x_l \\
x_{l+1} = F([x_0, x_1, \dots, x_l])
$$

ResNet 경우에는 입력이 출력에 더해지는 것이기 때문에 종단에 가서는 최초의 정보가 흐려질 수 밖에 없다. 그에 반해 DenseNet의 경우에는 채널 방향으로 그대로 합쳐지는 것이기 때문에 최초의 정보가 비교적 온전히 남아있게 된다. 

이러한 구조를 통해 얻을 수 있는 이점은 아래와 같다. 
- 그레디언트 소멸 개선
- feature propagation 강화
- feature 재사용
- 파라미터 수 절약

### DenseNet 구조

![image-20200324152158664](../../../resource/img/image-20200324152158664.png)

DenseNet의 구조를 표현한 것이다. 첫번째 convolution과 maxpooling 연산은 ResNet과 똑같다. 이 후 Dense Block과 Transition layer가 반복되고, 마지막의 fully connected layer와 softmax로 예측을 수행한다.

### Dense Block

Dense connectivity를 적용하기 위해서는 피쳐맵의 크기가 동일해야 한다. **같은 피쳐맵 크기를 공유하는 연산을 모아서** Dense Block을 구성하고 이 안에서 Dence Connectivity를 적용한다. 이 때, ResNet에서 배웠던 병목레이어(bottleneck layer)를 사용한다. 이 또한 연산량을 줄이기(계산의 복잡성을 줄이기) 위해 적용한 것이다. 

전체 convolution 연산의 출력 피쳐맵 갯수가 동일하다. 이 피쳐맵의 갯수를 `growth rate`라고 하고 k로 표현한다. 이는 하이퍼 파라미터이며, 논문에서는 k=32로 설정하였다. 따라서, Dense Block 내의 3x3 convolution 연산의 출력 피쳐맵의 갯수는 32이다. 1x1 convolution의 출력 피쳐맵의 갯수 또한 하이퍼 파라미터 이지만, 논문에서 4k를 추천한다. 따라서 Dense Block 내의 1x1 convolution 연산의 출력 피쳐맵 갯수는 128이다.

#### growth rate

DenseNet은 각 layer의 feature map의 channel 개수를 굉장히 작은 값을 사용하며 각 layer의 feature map의 channel 개수를 growth rate(k) 이라 부른다.

#### bottleneck layer

![image-20200412234703880](../../../resource/img/image-20200412234703880.png)

3x3 convolution 전에 1x1 convolution을 거쳐서 입력 feature map의 channel 개수를 줄이는 것까지는 같은데 그 뒤로 다시 입력 feature map의 channel 개수 만큼을 생성하는 대신 growth rate 만큼의 feature map을 생성하는 것이 차이이다. 이를 통해 computational cost를 줄일 수 있다고 한다.

### Transition layer

Dense Block 사이에 있는 1x1 convolution 연산과 average pooling 연산을 묶어 Transition layer 라고 한다. 이 부분을 통과하면서 **피쳐맵의 크기가 줄어**들게 된다. 앞에 있는 1x1 convolution은 다음 Dense Block으로 전해지는 **피쳐맵의 갯수를 조절**하는 역할을 한다. 입력되는 피쳐맵의 갯수를 m이라고 했을 때, $$[\theta m], (0 < \theta \leq1)$$을 출력하는 피쳐맵의 갯수로 한다. $$\theta$$ 또한 하이퍼 파리미터로 논문에서는 0.5로 설정했다.

마지막 Dense Block 뒤에 연결이 되며 Batch Normalization, ReLU, 1x1 convolution, 2x2 average pooling으로 구성되어 있다. 1x1 convolution을 통해 featrue map의 개수를 줄어주고 줄여주는 정도 theta는 하이퍼 파라미터 이다. 

transition layer를 통과하면 feature map의 개수(channel)이 줄어들고 2x2 average pooling layer를 통해 feature map의 가로세로 크기 또한 절반으로 줄어 든다. theta를 1로 사용하면 feature map의 개수를 그대로 가져가는 것을 의미한다.

### ResNet과 비교
- 파라미터의 크기(네트워크 크기)대비 성능이 우수하다. 즉, 네트워크의 학습이 더 쉽다.

### 장점
- 기울기 소실에 대한 해결(skip connection)
- 네트워크의 크기가 작아진다.

### 단점
- 깊어질 수록 depth가 늘어남에 따라 연산량이 증가한다. (Depth압축으로 해결 가능)


Reference
- https://datascienceschool.net/
- https://hoya012.github.io/blog/DenseNet-Tutorial-1/