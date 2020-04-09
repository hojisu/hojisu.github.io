<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script> 

# 확률론적 언어 모형(Probabilistic Language Model)

$$m$$ 개의 단어 $$w_1, w_2, …, w_m$$ 열(word sequence) 이 주어졌을 때 문장으로써 성립될 확률 $$P(w_1, w_2, … , w_m)$$ 을 출력함으로써 이 단어 열이 실제로 현실에서 사용될 수 있는 문장(sentence) 인지를 판별하는 모형이다. 각 단어의 확률과 단어들의 조건부 확률을 이용하여 계산할 수 있다.

$$
\begin{eqnarray}
P(w_1, w_2, \ldots, w_m) &=& P(w_1, w_2, \ldots, w_{m-1}) \cdot P(w_m\;|\; w_1, w_2, \ldots, w_{m-1}) \\
&=& P(w_1, w_2, \ldots, w_{m-2}) \cdot P(w_{m-1}\;|\; w_1, w_2, \ldots, w_{m-2}) \cdot P(w_m\;|\; w_1, w_2, \ldots, w_{m-1}) \\
&=& P(w_1) \cdot P(w_2 \;|\; w_1) \cdot  P(w_3 \;|\; w_1, w_2) P(w_4 \;|\; w_1, w_2, w_3) \cdots P(w_m\;|\; w_1, w_2, \ldots, w_{m-1})
\end{eqnarray}
$$

문맥(context)은 지금까지 나온 단어 정보이다.  $$P(w_m | w_1, w_2, … , w_{m-1})$$ 은 지금까지 $$w_1, w_2, … , w_{m-1}$$ 라는 단어 열이 나왔을 때, 그 다음 단어로 $$w_m$$ 이 나올 조건부 확률을 말한다. 

조건부 확률을 어떻게 모형화하느냐에 따라
- 유니그램 모형(Unigram Model)
- 바이그램 모형(Bigram Model)
- N그램 모형(N-gram Model)

### 유니그램 모형(Unigram Model)

모든 단어의 활용이 완전히 서로 독립이라면 단어 열의 확률은 다음과 같이 각 단어의 확률의 곱이 된다. 

$$
P(w_1, w_2, \ldots, w_m) = \prod_{i=1}^m P(w_i)
$$

### 바이그램 모형(Bigram Model)

단어의 활용이 바로 전 단어에만 의존할 때 (=마코프 모형(Markov Model))

$$
P(w_1, w_2, \ldots, w_m) = P(w_1) \prod_{i=2}^{m} P(w_{i}\;|\; w_{i-1})
$$

### N그램 모형(N-gram Model)

단어의 활용이 바로 전 $$n-1$$ 개의 단어에만 의존할 때 

$$
P(w_1, w_2, \ldots, w_m) = P(w_1) \prod_{i=n}^{m} P(w_{i}\;|\; w_{i-1}, \ldots, w_{i-n})
$$

### NLTK의 N그램 기능

NLTK 패키지에는 바이그램과 N-그램을 생성하는 `bigrams`, `ngrams` 명령이 있다.

~~~python
from nltk import bigrams, word_tokenize
from nltk.util import ngrams

sentence = "I am a boy."
tokens = word_tokenize(sentence)

bigram = bigrams(tokens)
trigram = ngrams(tokens, 3)

print("\nbigram:")
for t in bigram:
    print(t)

print("\ntrigram:")
for t in trigram:
    print(t)
    
# 결과
bigram:
('I', 'am')
('am', 'a')
('a', 'boy')
('boy', '.')

trigram:
('I', 'am', 'a')
('am', 'a', 'boy')
('a', 'boy', '.')
~~~

조건부 확률을 추정할 때는 문장의 시작과 끝이라는 조건을 표시하기 위해 모든 문장에 문장의 시작과 끝을 나타내는 특별 토큰을 추가한다. 예를 들어 문장의 시작은 `SS`, 문장의 끝은 `SE`이라는 토큰을 사용할 수 있다.

예를 들어 ["I", "am", "a", "boy", "."]라는 토큰열(문장)은 ["SS", "I", "am", "a", "boy", ".", "SE"]라는 토큰열이 된다. ngrams 명령은 padding 기능을 사용하여 이런 특별 토큰을 추가할 수 있다.

~~~python
bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SS", right_pad_symbol="SE")
for t in bigram:
    print(t)

# 결과
('SS', 'I')
('I', 'am')
('am', 'a')
('a', 'boy')
('boy', '.')
('.', 'SE')
~~~

### 조건부 확률 추정 방법

NLTK패키지를 사용하면 바이그램 형태의 조건부 확률을 쉽게 추정할 수 있다. 우선 `ConditionalFreqDist` 클래스로 각 문맥별 단어 빈도를 측정한 후에 `ConditionalProbDist` 클래스를 사용하면 조건부 확률을 추정한다.

~~~python
from nltk import ConditionalFreqDist

sentence = "I am a boy."
tokens = word_tokenize(sentence)
bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SS", right_pad_symbol="SE")
cfd = ConditionalFreqDist([(t[0], t[1]) for t in bigram])
~~~

`ConditionalFreqDist` 클래스는 문맥을 조건으로 가지는 사전 자료형과 비슷하다.

~~~python
cfd.conditions()

# 결과
['SS', 'I', 'am', 'a', 'boy', '.']
~~~

~~~python
cfd["SS"]

# 결과
FreqDist({'I': 1})
~~~

예시

nltk 패키지의 샘플 코퍼스인 movie_reviews의 텍스트로부터 바이그램 확률을 추정하는 예제이다.

~~~python
  import nltk
  nltk.download('movie_reviews')
  nltk.download('punkt')
  from nltk.corpus import movie_reviews
  
  sentences = []
  for tokens in movie_reviews.sents():
      bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SS", right_pad_symbol="SE")
      sentences += [t for t in bigram]
  
  sentences[:20]
~~~

문장의 처음(SS 문맥), i라는 단어 다음, 마침표 다음에 나오는 단어의 빈도는 다음과 같다.

~~~python
cfd = ConditionalFreqDist(sentences)
~~~

문장의 처음에 올 수 있는 단어들

~~~python
cfd["SS"].most_common(5)
~~~

- i 다음에 올 수 있는 단어들

~~~python
cfd["i"].most_common(5)
~~~

- 마침표 다음에 올 수 있는 단어들

~~~python
cfd["."].most_common(5)
~~~

빈도를 추정하면 각각의 조건부 확률은 기본적으로 다음과 같이 추정
$$
P(w\;|\; w_c) = \dfrac{C((w_c, w))}{C((w_c))}
$$

$$C((w_c, w))$$ 은 전체 말뭉치에서 $$(w_c, w)$$ 라는 바이그램이 나타나는 횟수이고, $$C(w_c)$$ 은 전체 말뭉치에서 $$C(w_c)$$ 라는 유니그램(단어)이 나타나는 횟수이다.

NLTK의 `ConditionalProbDist` 클래스에 `MLEProbDist` 클래스 팩토리를 인수로 넣어 위와 같이 빈도를 추정할 수 있다.

~~~python
    from nltk.probability import ConditionalProbDist, MLEProbDist
    cpd = ConditionalProbDist(cfd, MLEProbDist)
~~~

트레이닝이 끝나면 조건부 확률의 값을 보거나 샘플 문정을 입력해서 문장의 로그 확률을 구할 수 있다.

~~~python
    cpd["i"].prob("am")
    cpd["i"].prob("is")
    cpd["we"].prob("are")
    cpd["we"].prob("is")
~~~

    



Reference
- https://datascienceschool.net/