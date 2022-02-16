# 多线程 threading

## 准备
```python
from threading import Thread
```
## 执行
```python
thread1 = Thread(target=fuc_name,args=(param,))
thread1.setName("T-Name")  # 设置线程名称
thread1.getName()          # 获取线程名称
thread1.start()            # 启动线程
thread1.join()             # 等待线程结束
```
### Event
>event 对象最好单次使用,
创建一个 event 对象，让某个线程等待这个对象，一旦这个对象被设置为真，就应该丢弃它。

```python
from threading import Event
event = Event()  # 创建Event对象
event.wait()     # 等待Event对象为真
event.set()      # 设置Event为真
event.clear()    # 重置Event对象
```
### Lock
```python
from threading import Lock
lock = Lock()   # 创建线程锁
lock.acquire()  # 上锁
lock.release()  # 释放
```
### Condition
```python
from threading import Condition
con = Condition() 
con.acquire()       # 获取锁
con.release()       # 释放锁
con.wait()          # 等待通知
con.notify()        # 通知
con.notify_all()
```
### Class
```python
class myThread(threading.Thread):
    Lock = threading.Lock()

    def __init__(self, threadID, name, counter):
        super(myThread, self).__init__()
        self.threadID = threadID
        self.setName(name)
        self.counter = counter

    def run(self):
        name = self.getName()
        print("Start thread：" + name)
        self.print_time(name, self.counter, 5)
        print(u"Stop thread：" + name)

    def print_time(self, threadName, delay, counter):
        while counter:
            time.sleep(delay)
            self.Lock.acquire()
            print("%s: %s" % (threadName, time.ctime(time.time())))
            self.Lock.release()
            counter -= 1
```
# 线程池 
## 准备
```python
from concurrent.futures import ThreadPoolExecutor
```
## 执行
```python
pool = ThreadPoolExecutor(max_workers=5)
pool.submit(func_name,*args)
pool.submit(func_name,*args)
pool.shutdown(wait=True)

pool.map(func_name,[params1],[parasm2],)
# pool.map(print_time, [1,2],["No_1","No_2"])
```

