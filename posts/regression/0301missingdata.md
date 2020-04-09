<script> MathJax.Hub.Queue(["Typeset",MathJax.Hub]); </script>

# 누락 데이터 처리

실수형 자료에서는 `NaN`(not a number) 값을 이용하여 누락 데이터를 표시할 수 있지만 정수형에는 `NaN` 값이 없기 때문에 정수값을 실수로 자동 변환한다. 만약 정수형을 유지하고 싶다면 다음 코드와 같이 누락 데이터 표현이 가능한 정수형(nullable integer)인 `Int64Dtype` 자료형을 명시하여야 한다. 날짜 자료형도 마찬가지로 `parse_dates` 인수로 날짜시간형 파싱을 해주어야 `datetime64[ns]` 자료형이 되어 누락 데이터가 `NaT`(not a time) 값으로 표시된다.

### 누락 데이터 포착

~~~python
df.isnull()
~~~

~~~python
df.isnull().sum
~~~

`missingno`  누락데이터에 대한 시각화 

  - `matrix()`명령은 매트리스 형태로 누락데이터를 시각화하는 명령이다. 누락 데이터는 흰색으로 나타난다. 가장 오른쪽에 있는 것은 스파크라인(spark line)이라고 부르고, 각 행의 데이터 완성도를 표현한다.

  ~~~python
  import missingno as msno
  
  msno.matrix(df)
  plt.show()
  ~~~

  - 각 열의 누락데이터가 얼마나 존재하는지에 대해서만 시각화 하고 싶다면, `bar()`명령을 사용하면 된다.

  ~~~python
  msno.bar(df)
  plt.show()
  ~~~

  

### 누락 데이터 제거

판다스의 `dropna()`명령을 사용하면 누락 데이터가 존재하는 행이나 열을 지울 수 있다.

`thresh` 인수를 사용하면 특정 갯수 이상의 (비누락) 데이터가 있는 행만 남긴다.

`axis` 인자를 1로 설정하면 누락데이터가 있는 열을 제거한다.

### 누락 데이터 데체

누락 데이터를 처리하는 또다른 방법은 다른 독립변수 데이터로부터 누락된 데이터를 추정하여 대체(imputation)하는 것이다.

scikit-learn 패키지의 `SimpleImputer` 클래스를 사용하면 누락된 정보를 채울 수 있다.

~~~python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_copy1 = df.copy()
df_copy1["age"] = imputer.fit_transform(df.age.values.reshape(-1,1))

msno.bar(df_copy1)
plt.show()
~~~

