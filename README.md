# HLF
Solving Hidden Linear Function Problem using a quantum computer. Under development.
1. 터미널에서  pip install -r requirements.txt  명령어로 필요한 라이브러리를 다운받습니다.

2. 터미널에서  python -m visdom.server  를 실행하여 visdom 서버를 엽니다.

3. 크롬 등 인터넷 브라우저에서 주소창에  localhost:8097 을 입력하여 이동합니다.

4. jupyter notebook을 활용하여 Example.ipynb 파일을 참고하셔서 원하는 큐빗 array를 선언하고 연결 및 Sgate를 적용할 큐빗 등을 입력합니다.

5. 코드를 실행시키면 인터넷 브라우저에 해당 grid의 모양이 표시됩니다.

6. ipython에서 interactive하게 수행하실 수 있습니다.

7. circuitImplementation을 실행한 뒤 performStatevectorSim을 수행하면 회로의 결과상태를 구할 수 있습니다.
