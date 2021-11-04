# Collections
## Counter
```python
s = "abcbcaccbbad"
l = ["a","b","c","a","b","b"]
d = {"1":3, "3":2, "17":2}

#Counter各元素の数をとる、辞書を返す
print ("Counter(s):", Counter(s))
print ("Counter(l):", Counter(l))
print ("Counter(d):", Counter(d))
```
```python
Counter(s): Counter({'b': 4, 'c': 4, 'a': 3, 'd': 1})
Counter(l): Counter({'b': 3, 'a': 2, 'c': 1})
Counter(d): Counter({'1': 3, '3': 2, '17': 2})
```
|function|notice|
|:---- |:----|
|most_common(n)|降順で辞書の前のn個元素を返す|
|elements|計算後の元素発生器を返す|
```python
t = Counter("aabbccbccaabc")
t.most_common()
t.elements()
```
```python
(('c', 5), ('a', 4), ('b', 4))
('a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c')
```


## deque
二重終端キュー
|function|notice|
|:---------------------|:----|
|append()|キューの右に元素を追加|
|appendleft()|キューの左に元素を追加|
|clear()|キューのすべての元素を削除|
|count(para)|paraの数を返す|
|extend()|キューを右に拡張する　list tuple dict可能|
|extendleft()|キューを左に拡張する　list tuple dict可能|
|pop()|キューの右端の元素を抽出|
|popleft()|キューの左端の元素を抽出|
|remove(value)|キューから最初のvalueを削除|
|reverse()|キューを反転|
|rotate(n)|キュー元素右にn回移動する　負数の場合左に|

## defaultdict
辞書のすべての方法を継承
辞書を定義する時、値のタイプも定義する
```python
dic = defaultdict(str)   #list tuple dict range など
```
## OrderedDict

順番ありの辞書
