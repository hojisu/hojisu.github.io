# 리눅스 

## shall 종류
- 쉘(shell) : 사용자와 컴퓨터 하드웨어 또는 운영체제간 인터페이스이다.
  - 사용자의 명령을 해석해서 커널에 명령을 요청해주는 역할이다. 
  - 관련된 시스템콜을 사용해서 프로그래밍이 작성되었다.
- Bourne-Again Shell(bash) : GNU 프로젝트의 일환으로 개발되었다. 
- Bourne Shell(sh)
- C Shell(csh)
- Korn Shell(ksh) : 유닉스에서 가장 많이 사용 된다.

## 리눅스 기본 명령어 정리
- 리눅스 명령어는 쉘이 제공하는 명령어이다.
- 리눅스 기본 쉘이 bash 이므로, bash에서 제공하는 기본 명령어를 배우는 것이다.

- whoami : 로그인한 사용자 ID 확인
- passwd : 로그인한 사용자 ID의 암호변경
- sh : 사용자 변경
  - 보통 su - 와 함께 사용
    - su root : 현재 사용자의 환경설정 기반, root로 변경
    - su - root : 변경되는 사용자의 환경설정을 기반으로 root로 전환
- sudo 명령어 : root 권한으로 실행하기
ex) sudo apt-get update

### 파일 및 권한 관리 
- pwd : 현재 디렉토리 위치
- cd : 디렉토리 이동
- ls : 파일 목록 출력
  - ls와 와일드 카드
    - * : 임의 문자열
    - ? : 문자 하나
  - ls와 파일 권한
    ![ls_file](../../../resource/img/ls_file.png)
- man rm : 메뉴얼