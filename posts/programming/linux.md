# 리눅스 

## 리눅스
- 컴퓨터의 운영체제 중 하나로 리누스 토르발스(Linus Torvalds)에 의해 시작된 다중 사용자, 다중 작업을 지원하는 유닉스(UNIX)와 유사한 운영체제이다. 

#### 특징
- 유닉스 기반이다. 
- 다중 사용자(하나의 컴퓨터에 여러 사용자가 로그인 및 사용 가능)와 멀티 태스킹(한번에 여러 프로세스 실행 가능)을 지원한다. 
- 자유 소프트웨어이다. 
  - 리눅스 커널 및 관련 다양한 소프트웨어를 패키지로 묶어서 배포하는 것을 리눅스 배포판이라고 한다.(ex: ubuntu)

### 리눅스 구조
- 리눅스 커널 + 쉘 + 컴파일러 + 다양한 소프트웨어를 포함한 하나의 패키지를 지칭한다. 
- 운영체제가 시스템 자원을 관리하고, 다양한 소프트웨어는 리눅스 커널이 제공하는 시스템 콜을 통해 시스템 자원 사용을 요청한다. 
- 시스템 콜은 쉘, 언어별 컴파일러, 라이브러리를 통해 호출된다.
- 리눅스 커널은 시스템 자원을 관리 
  - 프로세스 관리(Process Management)
  - 메모리 관리(Memory Management)
  - 파일 시스템 관리(File System Management)
  - 디바이스 관리(Device Management)
  - 네트워크 관리(Network Management)


## shall 종류
- 쉘(shell) 
  - 운영체제 커널과 사용자 사이를 이어주는 역할
  - 사용자의 명령을 해석해서 커널에 명령을 요청해주는 역할이다. 
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
- pwd : 현재 디렉토리 위치
- cd : 디렉토리 이동
- ls : 파일 목록 출력
  - ls와 와일드 카드
    - `*` : 임의 문자열
    - `?` : 문자 하나
  - ls와 파일 권한
    ![ls_file](../../../resource/img/ls_file.png)
    출처 : http://gomguard.tistory.com/76
- cat : 파일 보기
- head/tail : head는 파일 시작부분, tail은 끝 부분을 보여준다.
- more : 파일 보기(화면이 넘어갈 경우, 화면이 넘어가기 전까지 보여줌)
- rm : 파일 및 폴더 삭제
  - rm -rf 
  - r옵션: 하위 디렉토리를 포함한 모든 파일 삭제
  - f옵션: 강제로 파일이나 디렉토리 삭제
- man : manual이라는 의미, mam rm을 입력하면 메뉴얼이 나온다.

### 리눅스 리다이렉션(redirection)과 피아프(pipe)
- standard stream
  - command로 실행되는 process는 세가지 스트림을 가지고 있다.
    - 표준 입력 스트림(standard input stream)
    - 표준 출력 스트림(standard output stream)
    - 오류 출력 스트림(standard error stream)
  - 모든 스트림은 일반적으로 plain text로 console에 출력하도록 되어 있다.

- 리다이렉션(redirection)
  - 스트림 흐름을 바꿔주는 것으로 < 또는 >을 사용한다.
  
- 파이프(pipe)
  - 두 프로세스 사이에서 한 프로세스의 출력 스트림이 또다른 프로세스의 입력 스트림으로 사용될 때 쓰인다. 
  - grep : 검색 명령
    - grep [-option] [pattern] [file or directory name]
      - <option>
        - `-i` : 영문의 대소문자를 구별하지 않는다. 
        - `-v` : pattern을 포함하지 않는 라인을 출력한다.
        - `-n` : 검색 결과의 각 행의 선두에 행 번호를 넣는다. (first line is 1)
        - `-l` : 파일명만 출력한다.
        - `-c` : 패턴과 일치하는 라인의 개수만 출력한다.
        - `-r` : 하위 디렉토리까지 검색한다.
reference
- 이태일 강사님 온라인 강의를 수강하며 강의자료를 토대로 공부하며 정리한 내용임을 말씀드립니다. 


