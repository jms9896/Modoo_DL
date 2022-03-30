# 모두를 위한 딥러닝 시즌1 Lec1 + Lab1

<hr>
목차.<br>
1. Supervise learning<br>
2. Linear ligression 모델 만들기 & 학습<br>
3. Tensor, Ranks, Shapes, and Types<br>
<hr>

## 1. Supervise learning

### Supervised learning / Unsupervided learning
이번 강의에서는 Supervised learning과 Unsupervided learning에 대해서 알고, Supervised learning에 초점을 두고 배운다.
![image](https://user-images.githubusercontent.com/91587463/160830214-a7bafb97-c996-43e2-86df-e508796f6650.png)
#### Supervided learning
정해져있는, labeling된 data, training set을 가지고 학습을 하는 친구.  
eg. 얘가 cat이냐 dog냐처럼 구분할 때 cat 사진 데이터 쫘라락 넣어주고, dog 데이터 쫘라락 넣어줘서 학습 인식하는 방법.
#### Unsupervised learning  
구글 뉴스나, word clustering처럼 labeling되지 않은 데이터를 주로 다룬다.
왜냐하면 labeling해주기 어려운 데이터가 있기 때문.  
자기가 스스로 data를 보고 학습을 한다.

### Supervise learning
일반적인 ML이라고 생각하면 된다.
Image labeling이라든지, Email spam filter, 시험점수 예측 등등...
![image](https://user-images.githubusercontent.com/91587463/160832930-b2002e37-a4af-441f-9070-40005125f907.png)  
모델 안에 학습시킬 데이터를 넣고 학습을 시킨다.
그 다음 testset을 넣어서 output을 확인한다.
이 표의 X를 특징, pitch라고 부른다.  
X_test를 집어넣을 때 pitch에 있는 [3, 6, 9]와 유사하기 때문에 Y값이 3이 도출된다.
### Supervised learning으로 알아보는 시험점수
1. Regression  
시험 점수를 공부시간으로 예측해보자!  
2. Classification  
2-1. binary classification
  시험점수로 합/불합 __두 가지__ 로 분류한다.
2-2. multi-label classification
  A, B, C, D, F 처럼 여러가지로 나눠서 분류한다.  
이번에도 X Y training set을 가지고 학습시킨다.  
ML regression model에 집어넣어서, X가 몇 인지 입력하면 Y(결과) 도출!


![image](https://user-images.githubusercontent.com/91587463/160830254-c366fdf0-150b-4861-a9a8-12f0d0055dcc.png)
왼쪽 페이지까지 classification. 
오른쪽 페이지부터 Lab 시작.

## 2. Linear ligression 모델 만들기 & 학습

### Tensorflow
이하 tf.  
google에서 만든 open source sw library  
현재 tf2(2.8.0)까지 나왔다.  
얘는 우리가 Data flow graph에 따른 연산을 하거나, 그에 따른 작업 결과물(return)을 얻을 수 있게 하는 친구다.  
설치방법은 다음과 같다.  
0. 파이썬과 VSCode가 설치되어있다고 가정. 혹은 colab 사용
1. 터미널에 `pip install --upgrade tensorflow-gpu` 입력. 자동설치.
  단, gpu가 없다면 `-gpu`는 빼고 설치하고, gpu는 nvidia의 cuda core를 사용하므로 nvidia cudart를 설치해줘야한다.  
  만약 VSCode에서 pip 인식을 못 한다면 "고급 시스템 설정보기" -> "환경변수" -> "path" 추가시켜주고 재부팅하자.
2. 실행 및 버전체크
  필자는 cmder를 사용하는데, cmd나 터미널 혹은 그냥 vscode에서 python을 실행한다.  
  `import tensorflow as tf #tensorflow가 길기 때문에 tf로 사용.
  tf.__version__`
  tf2가 설치되어있다면 
  `import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  tf.__version__`
  버전이 확인되면 설치 성공
  
### TF Hello world!
그래프에 노드를 생성해서 출력하는 과정이다.
`hello = tf.constant("Hello tensorflow!')`
이것만 가지고는 실행이 되지 않는다. 노드에 얹은 것 뿐이기 때문이다.  
`sess = tf.Session()`  
세션을 만들고,  
`print(sess.run(hello)` 세션을 run 해줘야 나타난다.
-> `b'Hello, tensorflow!'` 에서 b는 bytes를 의미하는 것. 
  
이 과정을 정리하면,  
1. Data flow graph를 설계한다.
2. graph를 실행한다.
3. 그러면 graph의 값이 변경되거나 return된다.


![image](https://user-images.githubusercontent.com/91587463/160830268-52881d7c-2e8f-4044-9efd-39985033999d.png)
이 작업은 그래프에 노드를 만들어서 실행시킨 것인데, 그래프를 만들어두고 실행시키고 싶다면 node를 placeholder로 만들면 된다.
`node = tf.constant(3.0, tf.float32)` 처럼 만들면 노드를 만들어서 실행하는 방법,  
`a = tf.placeholder(tf.float32)` 처럼 만들면 placeholder로 만들어둔 것. 이제 나중에 `feed_dict`를 이용해서 값을 넣을 수 있다. rank가 0인것도 가능하고 1인것도 가능.

## 3. Tensor, Ranks, Shapes, and Types.

#### Ranks
간단하게 몇 차원이냐를 의미하는 것.
`4` 이런건 rank 0, `[1, 2, 3]` 같은 vector는 rank 1, matrix는 rank 2, tensor(cube부터는 tensor라고 부른다)는 3, ... rank n 까지.

#### Shape
각각의 elements에 몇 개씩 들어있느냐를 의미한다. 
`t = [[1, 2, 3], [4, 5, 6]]`를 예로 들자면, 첫 번째 row에 column 3개 [1, 3], 두 번째 row에 column 3개 [2, 3], ...

#### Data type
보통 float32를 사용한다고 생각하고 넘어가자. 여러 타입이 있긴 하다.


