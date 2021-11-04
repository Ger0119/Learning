 # Itertools

 ## count
 無限ループの発生器

```python
count(start=0, step=1) --> start, start+step, start+2*step, ...
```

 ## cycle

繰り返し

```python
cycle(p) --> p0, p1, ... plast, p0, p1, ...
```
```python
t = cycle("ABC")
for x in t:
	print(x)
```
```python
A
B
C
A
B
C
...
```

 ## repeat
 オブジェクトをｎ回繰り返し

```python
repeat(elem [,n]) --> elem, elem, elem, ... endlessly or up to n times
```
```python
t = repeat((1,2,3),3)
for x in t:
    print(x)
```
```python
(1, 2, 3)
(1, 2, 3)
(1, 2, 3)
```
 ## accumulate

前のパラメータの和

```python
accumulate(p[, func]) --> p0, p0+p1, p0+p1+p2
```
```python
t = accumulate((1,2,3))
for x in t:
    print(x)
```
```python
1
3
6
```

 ## chain 
入力タイプはlist tuple str三種類
```python
chain(p, q, ...) --> p0, p1, ... plast, q0, q1, ...
```
```python
t = chain((1,2,3),(4,5,6))
for x in t:
    print(x)
```
```python
1
2
3
4
5
6
```
 ## compress
 selectorでdataのパラメータを選択して繰り返す
```python
compress(data, selectors) --> (d[0] if s[0]), (d[1] if s[1]), ...
```
```python
t1 = list("abcde")
t2 = [1,0,1,1,0]
d = compress(t1,t2)
for x in d:
    print(x)
```
```
a
c
d
```
 ## product
複数の発生器を組み合わせて繰り返す
```python
product(p, q, ... [repeat=1]) --> cartesian product
```
```python
t1 = (0,1)
d = product(t1,t1,t1)
for x in d:
    print(x)
```
```python
(0, 0, 0)
(0, 0, 1)
(0, 1, 0)
(0, 1, 1)
(1, 0, 0)
(1, 0, 1)
(1, 1, 0)
(1, 1, 1)
```
