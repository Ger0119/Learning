## Numpy
[numpy doc](https://numpy.org/doc/stable/reference/index.html)

### 数据类型
```python
dt  = np.dtype(np.int32)
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
```
### 创建数组
```   python
npa = np.array([(10,),(20,),(30,)],dtype = dt)
npa = np.asarray([(10,),(20,),(30,)],dtype="int32")  #从已有的队列或元组获取数据
npa = np.arange(1,101,5,"int")
npa = np.linspace(1,10,10,endpoint=True) 	  #(start,stop,count)
npa = np.logspace(1.0 , 2.0 , num=10,base=2)  #默认底数为2
npa = np.full((3,4),np.nan)      #创建数据全部为np.nan的3x4的数组
```
#### 创建随机数组
|函数|描述|
|:---|:---|
|np.random.seed()|种子，相同种子生成的随机数一样|
|np.random.rand()|随机数组，输入形状|
|np.random.randn()|同rand，具有标准正态分布|
|np.random.random(())|随机数组，输入元组|
|np.random.randint(low,high,size)|大于等于low小于high的int数据随机数组|
|np.random.uniform(low,high,size)|大于等于low小于high的float数据随机数组|
|np.random.normal(loc,scale,size)|loc平均scale标准偏差size形状|

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
npa = np.eye(3)
```

>[[0. 0.]     [[1 1]    [[1,0]
> [0. 0.]]     [1 1]]    [0,1]]

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
npa.reshape()			#不能大于原数组 参数为-1时 自动调整
np.resize(array,(size)) #可以大于原数组 自动填充
```
#### 展开数组
|函数|描述|
|:---|:---|
|flat|数组元素迭代器|
|flatten|返回展开数组的拷贝|
|ravel|返回展开数组，修改会影响原数组|

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
|numpy.hstack((n1,n2,..)) |水平堆叠(列增加)|
|numpy.vstack((n1,n2,..)) |竖直堆叠(行增加方向)|

#### 分割数组
|函数|描述|
|:---|:---|
|numpy.split(arrray, indices_or_sections, axis)|indices_or_sections：如果是一个整数，就用该数平均切分,如果是一个数组，为沿轴切分的位置（左开右闭|
|numpy.hsplit(array,indices_or_sections)|水平分割(列方向分割)|
|numpy.vsplit(array,indices_or_sections)|竖直分割(行方向分割)|
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

#### 生成网格矩阵
|函数|说明|
|:---|:---|
|np.meshgrid(npa,npb)|生成网格矩阵，在3D图形经常使用|
```python
npa = np.arange(0,4)
npb = np.arange(4,7)
xx,yy = np.meshgrid(npa,npb)
```
>[[0,1,2,3]   [[4,4,4]
>  [0,1,2,3]     [5,5,5]
>  [01,2,3]]     [6,6,6]]

### 迭代                                                                                                                                                                                                                                                         
|函数|说明|
|:---|:---|
|np.nditer(array,op_flags,flags,order)|返回一维数组|
|op_flags|["readwrite"]模式-可修改原数组|
|order|'C'行优先'F'列优先|

| flags参数            | 描述                                           |
| :-------------- | :--------------------------------------------- |
| c_index       | 可以跟踪 C 顺序的索引                          |
| f_index       | 可以跟踪 Fortran 顺序的索引                    |
| multi-index   | 每次迭代可以跟踪一种索引类型                   |
| external_loop | 给出的值是具有多个值的一维数组，而不是零维数组 |

```python
npa = np.arange(0,60,5).reshape(3,4)
for x in np.nditer(a,flags=['external_loop'],order='F'):
	print(x,end=",")
```
>[ 0 20 40], [ 5 25 45], [10 30 50], [15 35 55],

### 函数运算
#### 位运算
|函数|说明|
|:---|:---|
|bitwise_and|对数组元素按位**与**|
|bitwise_or|对数组元素按位**或**|
|invert|按位取反|
|left_shift|向左移动二进制位|
|right_shift|向右移动二级制位|

#### 字符串运算
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

#### 数学函数

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
|np.sqrt()|方根|
|np.clip(npa,max,min)|范围|
|np.log()|log底数为自然数e|
|np.log10()|log底数为10|
|np.exp()|e\*\*x |
|np.allclose(npa,npb,atol)|判断误差范围，atol误差大小|

#### 统计函数
没有传入axis时，将展开计算
|函数|描述|
|:---|:---|
|np.amin()|最小值|
|np.amax()|最大值|
|np.ptp()|最大值 - 最小值|
|np.percentile(a,q,axis)|百分位数|
|np.median()|中位数|
|np.mean()|算术平均值|
|np.average(arr,weights,returned)|加权平均值|
|np.std()|标准差|
|np.var()|方差|
|np.diff(npa,n,axis)|与前一个元素的差，n执行次数|

#### 排序，条件函数
|函数|描述|
|:---|:---|s
|np.sort(a,axis,kind,order)|排序|
|np.argsort()|返回数值从小到大index|
|np.lexsort()|多个序列排序|
|np.argmin()|返回最小元素index|
|np.argmax()|返回最大元素index|
|np.nonzero()|返回非零值index|
|np.where()|返回满足条件index|
|np.extract()|根据条件抽取元素|

| 种类                      | 速度 | 最坏情况      | 工作空间 | 稳定性 |
| :------------------------ | :--- | :------------ | :------- | :----- |
| 'quicksort'（快速排序） | 1    | O(n^2)     | 0        | 否     |
| 'mergesort'（归并排序） | 2    | O(n\*log(n)) | ~n/2     | 是     |
| 'heapsort'（堆排序）    | 3    |O(n\*log(n)) | 0        | 否     |



#### 字节交换

|函数|描述|
|:---|:---|
|npa.bytes()|ndarray转换为bytes|
|np.frombuffer(npa,dtype=np.uint8)|将bytes转化为ndarray|
|npa.ndarray.byteswap()|将 ndarray 中每个元素中的字节进行大小端转换|

```python
npa = np.array([1, 256, 8755], dtype=np.int16)
print(npa)
print(list(map(hex, npa)))
print(npa.byteswap(inplace=True))
print(list(map(hex, npa)))
```
>[   1  256 8755]
['0x1', '0x100', '0x2233']
[  256     1 13090]
['0x100', '0x1', '0x3322']

### 副本，视图，深浅复制

|函数|描述|
|:---|:---------------------------------|
|npb = npa|无复制：ID一样，修改npb，npa也会被修改|
|npb = npa.view()|视图/浅复制：维度不会被影响，元素会被影响|
|npb = npa.copy()|副本/深复制：相互不影响|

### 矩阵图(Matrix)

一个m X n 的矩阵是一个由 m 行（row）n 列（column)元素排列成的矩形阵列。

#### 创建矩阵
|函数|描述|
|:---|:---------------------------------|
|np.matlib.empty(shape,dtype,order|   填充为随机数据|
|np.matlib.zeros((2,2))	| 填充为0|
|np.matlib.ones((2,2))	|填充为1|
|np.matlib.eye(n,M,k,dtype)|对角线元素为1，其他为0|
|np.matlib.identity(num,dtype)|返回给定大小得单位矩阵|
|np.matlib.rand(row,column)|返回给定大小得矩阵,随机填充数据(0~1)|
|np.asarray()|矩阵变数组|
|np.asmatrix()|数组变矩阵，必须是二维数组|
```python
np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float)
```
>[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]]

#### 线性代数

| 函数          | 描述                             |
| :------------ | :------------------------------- |
| np.dot()         | 两个数组的点积，即元素对应相乘。可用npa @ npb |
| np.vdot()        | 两个向量的点积                   |
| np.inner()      | 两个数组的内积                   |
| np.matmul()     | 两个数组的矩阵积 (两个二维数组返回点积，否则点积后广播) |
| np.linalg.det() | 数组的行列式                     |
| np.linalg.solve() | 求解线性矩阵方程                 |
| np.linalg.inv() | 计算矩阵的乘法逆矩阵 AB=BA=E     |

```python
npa = np.array([[1,2],
				[3,4]])
npb = np.array([[11,12],
				[13,14]])
np.dot(npa,npb)
np.vdot(npa,npb)
np.inner(npa,npb)
```
>[[37  40] 
>[85  92]]
>`[[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]`
>
>130
>`1*11 + 2*12 + 3*13 + 4*14 = 130`
>
>[[35, 41],
>[81, 95]]
>
>```python
>1*11+2*12, 1*13+2*14 
>3*11+4*12, 3*13+4*14
>```

```python
npa = [[1,0],[0,1]] 
npb = [[4,1],[2,2]] 
np.matmul(a,b)
```

>[[4, 1],
 [2, 2]]    #同维数组 返回点积
```python
npa = np.array([[6,1,1], [4, -2, 5], [2,8,7]]) 
np.linalg.det(npa)   #行列式计算
```
>[[ 6  1  1]
 [ 4 -2  5]
 [ 2  8  7]]
 `6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2)`

```python
npa = np.array([[1,1,1],[0,2,5],[2,5,-1]])
npb = np.array([[6],[-4],[27]])
np.linalg.solve(npa,npb)  #解线性方程
```
>[[ 5.],[ 3.],[-2.] ] 

### I/O
|函数|描述|
|:---|:---|
|np.save()|保存单个数组|
|np.load()|读取单个数组|
|np.savez(file,arrs,keyword)|多个数组|
|np.savetxt(FILENAME, arr, fmt="%d", delimiter=",") |改为数字保存，逗号分割|
|np.loadtxt(FILENAME, dtype=int, delimiter=' ')|读取数据 空格分割|
