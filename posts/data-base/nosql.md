# NoSQL

## Summary
- 데이터 테이블 사이에 관계가 없이 저장하는 데이터베이스
- 데이터 사이의 관계가 없으므로 복잡성이 줄고 많은 데이터를 저장 가능


## NoSQL

- Not only SQL
- Mongodb, Hbase, Cassandra
- RDBMS 한계를 극복하기 위해 만들어진 데이터 베이스
- 확장성이 좋음
  - **데이터 분산처리 용이**
- 데이터 저장이 유연함 
  - RDBMS와 다르게 구조 변경 불필요
- 스키마(Schema) 및 Join이 없음
- Collection 별로 관계가 없기 때문에 모든 데이터가 들어있어야 함
- 저장되는 데이터는 key-value 형태의 JSON 포멧을 사용
- Select는 RDBMS 보다 느리지만 Insert 가 빨라 대용량 데이터 베이스에 많이 사용
- Transaction 지원이 되지 않음(동시 수정에 대한 신뢰성이 지원되지 않음)
- scale out