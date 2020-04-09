<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# GAN (Generative Adversarial Network)

GAN은 생성기와 판별기가 서로 대립하며 서로의 성능을 점차 개선해 나가자는 것이 주요 개념이다. 
- 생성을 목적으로 한다
- 생성기와 판별기가 서로 대립(**Adversarial**)하며 서로의 성능을 점차 개선해 나가자는 것이 주요 개념입니다.
- GAN의 목표는 학습할 때 사용했던 데이터 셋에는 없지만 실제와 다를바 없는 데이터를 생성하는 것이다. 

![image-20200312180116875](../../../resource/img/image-20200312180116875.png)

### GAN 학습과정

먼저 생성기(Generator)는 고정하고 판별기(Discriminator)가 Value function V(D, G)를 최대화한다. 의미적으로는 진짜 데이터에 대해서는 1을 출력하고 생성된 가짜 데이터에 대해서는 0을 출력하여 $$V(D, G) \approx 0$$ 으로 수렴하도록 한다. 그 다음으로 판별기를 고정하고 생성기가 $$\mathbb{E}_{z \tilde{} p_z(z)}[\log(1 - D(G(z)))]$$ 를 최소화 하도록 즉 $$D(G(z)) \approx 1$$ 으로 수렴하도록 최적화 한다. 



Reference
- https://datascienceschool.net/