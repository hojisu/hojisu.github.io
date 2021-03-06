# RDBMS

## Summary
- 데이터 테이블 사이에 **키값**으로 관계를 가지고 있는 데이터베이스
- 데이터 사이의 관계 설정으로 최적화된 스키마를 설계 가능
- 트랜잭션은 하나의 논리적 작업 단위를 구성하는 일련의 연산들의 집합 (ACID)

## 장단점
- 장점
    - 데이터 분류, 정렬, 탐색속도가 빠름
    - 오래 사용된 만큼 신뢰성이 높다
- 단점
    - 스키마 수정이 여려움

## 트랜잭션 
하나의 논리적 작업 단위를 구성하는 일련의 연산들의 집합

### ACID
- Atomicity(원자성) : 트랜잭션의 모든 작업들이 수행 완료 되거나 전혀 어떠한 연산도 수행되지 않은 상태 보장
- Consistency(일관성) : 성공적으로 수행한 트랜잭션은 정당한 데이터만을 데이터베이스에 반영
- Isolation(독립성) : 여러 트랜잭션이 동시에 수행 되더라도 각각의 트랜잭션은 다른 트랜잭션에 영향을 받지 않고 독립적으로 수행
- Durability(지속성) : 트랜잭션이 성공적으로 완료되어 커밋하면 해당 트랜잭션에 의한 모든 변경은 향후 어떤 소프트웨어나 하드웨어 장애가 발생되더라도 보존

## Table

- 행(row)과 열(column)로 이루어져 있는 데이터베이스를 이루는 기본 단위
- Storage Engine
    - MyISAM : full text index 지원, table 단위 lock, select가 빠름, 구조 단순
    - InnoDB : transaction 지원, row 단위 lock, 자원을 많이 사용, 구조 복잡

### Column(열)

- 테이블의 세로축 데이터
- Field, Attribute 라고도 불림

### Row(행)

- 테이블의 가로축 데이터
- Tuple, Recode 라고도 불림

### Value

- 행(row)과 열(column)에 포함되어 있는 데이터

### Key

- 행(row)의 식별자로 사용

## Relationship

- 1:1
- 1:n
- n:n

## Schema

- 스키마는 데이터 베이스의 구조를 만드는 디자인