...
$ cd ./home
$ chmod -R 755 e381
$ source ./venv/bin/activate
$ nohup python -u w045runMTF.py > nohup2.out 2>&1 &

$ python filename.py &


$ tail -f nohup.out   ---> 빠져나오고 싶으시면 ctrl + z  (not command + z)
$ top or top &        ---> ctrl + z

https://blkcoding.blogspot.com/2018/03/nohup.html
https://choseongho93.tistory.com/124

$ nohup python filename.py &
만약 print문을 바로 바로 보고 싶다면 파이썬의 자체에 가지고 있는 옵션이 있다.
$ nohup python -u filename.py &
$ nohup python -u filename.py > filename.out &
$ nohup some_command > nohup2.out 2>&1 &


$ tail -f nohup.out   ---> 빠져나오고 싶으시면 ctrl + z  (not command + z)
$ top or top &        ---> ctrl + z

    1  passwd
    2  ls -la
    4  python --version
    5  git
    7  mkdir home
    8  ls -la
    9  cd home
   12  git init
   13  git remote add origin https://github.com/tobby2002/e381.git
   14  git clone https://github.com/tobby2002/e381.git
   16  cd e381
   18  python -m venv venv
   20  source ./venv/bin/activate
   21  pip list
   22  pip install --upgrade pip
   23  pip install -r requirements.txt 
   34  git pull origin main
   35  pip install -r requirements.txt 
   36  sudo yum install gcc
   37  yum install gcc-c++
   39  yum install gcc libffi-devel python-devel openssl-devel -y
   40  pip install -r requirements.txt 
   42  python n382_longshort_binance_trade_v0.2.py &
   43  netstat -nltp
   44  ps aux | grep python
   46  vi n382_longshort_binance_trade.log 
   47  :q!
   49  ps aux | grep python
   50  cd home
   52  cd e381/
   54  less n382_longshort_binance_trade.log 
   58  history
   59  ps aux | grep python
   60  cd home/e381/
   61  ls -la
   63  ps aux | grep python
   64  rm
   66  rm n382_longshort_binance_trade.log 
   68  rm open_order_history.pkl 
   69  chmod -755 n382_longshort_binance_trade_v0.2.py 
   71  nohup python n382_longshort_binance_trade_v0.2.py &
   72  ps -ef | grep n382_longshort_binance_trade_v0.2.py
   74  ps aux | grep python
   77  nohup python -u n382_longshort_binance_trade_v0.2.py > nohup2.out 2>&1 &
   78  ps -ef | grep python

   82  source ./venv/bin/activate
   83  nohup python -u n382_longshort_binance_trade_v0.2.py > nohup2.out 2>&1 &
   85  ps -ef | grep python
   87  tail -f nohup2.out
   89  cd e381
   93  chmod -R 755 e381/
   95  cd e381/
   97  history
   
   
   lsof -i
   kill -9 xxxxx