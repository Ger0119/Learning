## Numpy
### 数据类型
```python
dt  = np.dtype(np.int32)
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
```
### 创建数组
```python
npa = np.array([(10,),(20,),(30,)],dtype = dt)
npa = np.asarray([(10,),(20,),(30,)],dtype="int32")  #从已有的队列或元组获取数据
npa = np.arange(1,101,5,"int")
npa = np.linspace(1,10,10,endpoint=False) 	  #(start,stop,count)
npa = np.logspace(1.0 , 2.0 , num=10,base=2)  #默认底数为2
```
#### 数组属性
```python
npa.ndim      #维度
npa.shape     #大小
npa.size      #元素总个数
npa.reshape() #调整大小
```
#### 创建指定形状数组
```python
npa = np.empty([3,2],dtype='int8') #随机数据
```

>[[114  16]
 [124    2]
 [ -96    0]]

#### 创建0/1填充数组
```python
npa = np.zeros((2,2))
npa = np.ones([2,2],dtype=int)
```

>[[0. 0.]     [[1 1]
 [0. 0.]]     [1 1]]

#### 创建数组
```python
npa = np.arange(10)
```
>[0 1 2 3 4 5 6 7 8 9]

#### 重复指定数组
```python
npa = np.array([[1,2,3],[4,5,6]])
np.tile(npa,(2,1))   #(row重复次数,column重复次数)
```
>[[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]]

#### 创建字符串数组
```python
a = np.frombuffer(b'Hello word',dtype = 'S1')  #S1 字符个数
```
>[b'H' b'e' b'l' b'l' b'o' b' ' b'w' b'o' b'r' b'd']

#### 使用迭代器创建数组
```python
a = np.fromiter(iter(range(5)), dtype=int)
```
>[0 1 2 3 4]

### 索引
#### 切片索引
```python
npa[1:3,1:3]
npa[1:3,[1,2]]
npa[...,1:] # [row,column]
```
#### 布尔索引
```python
npa[npa>5]
```
#### 过滤
```python
npa[~np.isnan(npa)]    #过滤NaN
npa[np.iscomplex(npa)] #过滤非复数
```
#### 花式索引
```python
npa = np.arange(32).reshape(8,4)
npa[[4,2,1,7]]
```
>[[16 17 18 19]
 [  8   9  10 11]
 [  4   5    6   7]
 [28 29 30 31]]

```python
npa[np.ix_([1,5,7,2],[0,3,1,2])] #多个数组 需要使用np.ix_()
```
>[[  4   7   5   6]
 [20 23 21 22]
 [28 31 29 30]
 [  8 11   9 10]]

### 数组操作
#### 数组反转
```python
npa = npa.T
npa = np.transpose(npa)
```
#### 修改数组形状
```python
npa.reshape()			#不能大于原数组
np.resize(array,(size)) #可以大于原数组 自动填充
```
#### 修改数组维度
```python
npa = np.array([[1],[2],[3]])
npb = np.array([4,5,6])
# 对 y 广播 x
b = np.broadcast(npa,npb)   #返回两个迭代器
npc = np.empty(b.shape)
npc.flat = [u + v for (u,v) in b]  # flat 元素
```
>[[5. 6. 7.]
 [6. 7. 8.]
 [7. 8. 9.]]
```python
np.boradcast_to(npa,(3,3))
```
>[[1 1 1]
 [2 2 2]
 [3 3 3]]


numpy.expand_dims(arr, axis)

指定位置插入新的轴

numpy.squeeze(arr, axis)

删除指定的轴

#### 连接数组
|函数|描述|
|:---|:---|
|numpy.concatenate((n1,n2,...),axis)|连接数组|
|numpy.stack((n1,n2,..),axis)|沿轴连接数组|
|numpy.hstack(n1,n2,..) |水平堆叠(列方向)|
|numpy.vstack(n1,n2,..) |竖直堆叠(行方向)|
#### 分割数组
|函数|描述|
|:---|:---|
|numpy.split(arrray, indices_or_sections, axis)|indices_or_sections：如果是一个整数，就用该数平均切分,如果是一个数组，为沿轴切分的位置（左开右闭|
|numpy.hsplit(array,axis)|水平分割(列方向分割)|
|numpy.vsplit(array,axis)|竖直分割(行方向分割)|
```python
npa = np.array(range(9))
npb = np.split(npa,3)
npc = np.split(npa,[4,7])
```
>[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]
[array([0, 1, 2, 3]), array([4, 5, 6]), array([7, 8])]
```python
npa = np.array([[1,2,3,4],[2,3,4,5],[4,5,6,7]])
npb = np.hsplit(npa,2)
```
>[array([[1, 2], [2, 3], [4, 5]]),
> array([[3, 4],[4, 5], [6, 7]])]

```python
npa = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
npb = np.vsplit(npa,2)      
```
>[array([[1, 2, 3], [4, 5, 6]]), 
> array([[ 7,  8,  9],[10, 11, 12]])]

#### 数组添加插入删除
|函数|描述|
|:---|:---|
|numpy.append(arr,values,axis=None)|添加到末尾|
|numpy.insert(arr,obj,valuse,axis)|添加到指定index|
|numpy.delete(arr,obj,axis)|删除某个子数组|
|numpy.unique(arr,return_index,return_inverse,return_counts) |去重|

```python
npa = np.array([[1,2,3],[4,5,6]])
np.append(npa,[7,8,9])
np.append(npa,[[7,8,9]],axis=0)
np.append(npa,[[5,6,7],[7,8,9]],axis=1)
```
>[1 2 3 4 5 6 7 8 9]
[[1 2 3]
 [4 5 6]
 [7 8 9]]
[[1 2 3 5 6 7]
 [4 5 6 7 8 9]]

```python
npa = np.array([[1, 2], [3, 4], [5, 6]])
#未传递 Axis 参数。 在插入之前输入数组会被展开
np.insert(npa, 3, [11, 12])
#传递了 Axis 参数。 会广播值数组来配输入数组
np.insert(npa, 1, [11], axis=0)
```
>[ 1  2  3 11 12  4  5  6]
>[[ 1  2]
 [11 11]
 [ 3  4]
 [ 5  6]]

```python
npa = np.arange(3,15).reshape(3, 4)
# 未传递 Axis 参数。 在插入之前输入数组会被展开。
np.delete(npa, 5)
np.delete(npa,1,axis = 1)   #删除第二列
np.delete(npa,np.s_[::2])   #传入切片将展开
```
>[ 3  4  5  6  7  9 10 11 12 13 14]
[[ 3  5  6]
 [ 7  9 10]
 [11 13 14]]
[ 4  6  8 10 12 14]

```python
npa = np.array([5,2,6,2,7,5,6,8,2,9])
np.unique(npa,return_index=True,return_inverse=True,return_counts=True)
```
>[2 5 6 7 8 9]                #去重值
[1 0 2 4 7 9]                #去重值index
[1 0 2 0 3 1 2 4 0 5]	#原数组index
[3 2 2 1 1 1]	              #去重重复数量

### 位运算
|函数|说明|
|:---|:---|
|bitwise_and|对数组元素按位**与**|
|bitwise_or|对数组元素按位**或**|
|invert|按位取反|
|left_shift|向左移动二进制位|
|right_shift|向右移动二级制位|
### 字符串运算
| 函数           | 描述                                  |
| :------------- | :----------------------------------- |
| np.char.add()          | 对两个数组的逐个字符串元素进行连接         |
| np.char.multiply(char,time) | 返回按元素多重连接后的字符串              |
| np.char.center(char,len,fillchar)       | 居中字符串            |
| np.char.capitalize()   | 将字符串第一个字母转换为大写              |
| np.char.title()        | 将字符串的每个单词的第一个字母转换为大写    |
| np.char.lower()        | 数组元素转换为小写                      |
| np.char.upper()        | 数组元素转换为大写                      |
| np.char.split(char,sep) | 指定分隔符对字符串进行分割，并返回数组列表  |
| np.char.splitlines()   | 返回元素中的行列表，以换行符分割          |
| np.char.strip(chararray,str) | 移除元素开头或者结尾处的特定字符          |
| np.char.join([unit],arrays) | 通过指定分隔符来连接数组中的元素          |
| np.char.replace(char,before,after) | 使用新字符串替换字符串中的所有子字符串| | np.char.decode()       | 数组元素依次调用str.decode             |
| np.char.encode()       | 数组元素依次调用str.encode             |

### 数学函数

|函数|描述|
|:---|:---------------------------------|
|np.sin()|sin函数|
|np.cos()|cos函数|
|np.tan()|tan函数|
|np.arcsin()|反正弦函数|
|np.degrees()|变成角度|
|np.around(num,decimals)|四舍五入|
|np.floor()|向上取整|
|np.ceil()|向下取整|
|np.add()|加|
|np.substract()|减|
|np.multiply()|乘|
|np.divide()|除|
|np.reciprocal()|倒数|
|np.power()|幂|
|np.mod()|取余|
|np.remainder()|取余|

### 统计函数
没有传入axis时，将展开计算
|函数|描述|
|:---|:---|
|np.amin()|最小值|
|np.amax()|最大值|
|np.ptp()|最大值 - 最小值|
|np.percentile(a,q,axis)|百分位数|
|np.median()|中位数|
|np.mean()|算术平均值|
|np.average()|加权平均值|
|np.std()|标准差|
|np.var()|方差|

### 排序，条件函数

| 种类                      | 速度 | 最坏情况      | 工作空间 | 稳定性 |
| :------------------------ | :--- | :------------ | :------- | :----- |
| 'quicksort'（快速排序） | 1    | O(n^2)     | 0        | 否     |
| 'mergesort'（归并排序） | 2    | O(n\*log(n)) | ~n/2     | 是     |
| 'heapsort'（堆排序）    | 3    |O(n\*log(n)) | 0        | 否     |

