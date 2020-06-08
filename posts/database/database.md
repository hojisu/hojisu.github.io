<script> MathJax.Hub.Queue(["Typeset", MathJax.Hub]); </script>

# Database

## DB(Database)
데이터를 통합하여 관리하는 데이터의 집합입니다.

## DBMS(Database Management System)
데이터베이스를 관리하는 미들웨어 시스템을 데이터베이스 관리시스템입니다. 

## RDBMS (Relational Database Management System) 
Oracel, Mysql등이 있습니다.
데이터 테이블 사이에 키값으로 관계를 가지고 있는 데이터베이스 입니다. 
데이터 사이의 관계 설정으로 최적화된 스키마를 설계 할 수 있습니다.

### Feature
데이터 분류, 정렬, 탐색 속도가 빠릅니다.
오래 사용된 만큼 신뢰성이 높습니다.
단점으로는 스키마 수정이 어렵습니다. 

## NoSQL
Mongodb, Hbase, Cassandra가 있습니다.
데이터 테이블 사이에 관계가 없이 저장하는 데이터베이스 입니다.
데이터 사이의 관계가 없으므로 복잡성이 줄고 많은 데이터를 저장 가능합니다. 

### Feature
확장성이 좋아 데이터의 분산처리에 용이합니다. 
데이터 저장이 유연합니다. 
스키마 및 Join이 없습니다. 
Collection 별로 관계가 없기 때문에 모든 데이터가 들어 있어야 합니다. 
저장되는 데이터는 Key-Value 형태의 Json 포멧을 사용합니다. 
Select는 RDBMS 보다 느리지만 Insert가 빨라 대용량 데이터 베이스에 많이 사용됩니다. 
Transaction이 지원되지 않습니다. 
