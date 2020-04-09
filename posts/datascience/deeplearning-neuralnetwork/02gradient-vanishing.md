<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# 그레디언트 소멸 문제 (gradient vanishing)

### Summary

- gradient vanishing은 오차역전파시에 오차가 뉴런을 거치면서 활성화 함수의 기울기가 곱해지면서 값이 1보다 작아서 계속 크기가 감소하면서 발생한다.

________________

일반적으로 레이어의 수가 많을수록 복잡한 형태의 베이시스 함수를 만들 수 있다. 하지만 레이어 수가 증가하면 가중치가 잘 수렴하지 않는 현상이 발생한다. 

### 기울기와 수렴속도 문제

은닉계층의 수가 너무 증가하면 수렴 속도 및 성능이 급격히 저하된다. 가중치가 감소하는 원인은 오차역전파(backpropagation)시에 오차가 뉴런을 거치면서 활성화 함수의 기울기가 곱해지는데 이 값이 1보다 작아서 계속 크기가 감소하기 때문이다. 

일반적으로 사용하는 잔차 제곱합(sum of square) 형태의 오차 함수는 대부분의 경우에 기울기 값이 0 이므로 (near-zero gradient) 수렴이 느려지는 단점이 있다.





Reference
- https://datascienceschool.net/