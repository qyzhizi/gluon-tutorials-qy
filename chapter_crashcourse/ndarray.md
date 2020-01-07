# 使用NDArray来处理数据
对于机器学习来说，处理数据往往是万事之开头。它包含两个部分：(i)数据读取，(ii)数据已经在内存中时如何处理。本章将关注后者。
我们首先介绍`NDArray`，这是MXNet储存和变换数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点使得`NDArray`能更好地支持机器学习。
## 让我们开始
我们先介绍最基本的功能。如果你不懂我们用到的数学操作也不用担心，例如按元素加法、正态分布；我们会在之后的章节分别从数学和代码编写的角度详细介绍。
我们首先从`mxnet`导入`ndarray`这个包

矩阵相乘：
设A为mxp的矩阵，B为pxn的矩阵，那么称mxn的矩阵C为矩阵A与B的乘积，记作
C=AXB

```{.python .input  n=1}
from mxnet import ndarray as nd
```

然后我们创建一个3行和4列的2D数组（通常也叫**矩阵**），并且把每个元素初始化成0

```{.python .input  n=2}
nd.zeros((3, 4))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[[0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
for　i in 　
 print(nd[i])
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input  n=4}
x = nd.ones((3, 4))
x
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=47}
#列表打印
list1=[1, 2, 3, 4 ]
list2=[2,3,5,5]
list3[:]=[list1[:],list2[:]]
print(list3)
print(list2)
print(list1)
print(id(list1))
print(id(list3[0]))
list3[0]=[1,2,3,0]
print(list1)
print(list3[0])
print(id(list1))
print(id(list3[0]))
```

```{.json .output n=47}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[1, 2, 3, 4], [2, 3, 5, 5]]\n[2, 3, 5, 5]\n[1, 2, 3, 4]\n140548172616520\n140548112249160\n[1, 2, 3, 4]\n[1, 2, 3, 0]\n140548172616520\n140548172574664\n"
 }
]
```

或者从python的数组直接构造

```{.python .input  n=3}
F=nd.array([[1,2],[2,3]])
print(F)
print(nd.mean(F))
nd.mean(F).asscalar()
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[1. 2.]\n [2. 3.]]\n<NDArray 2x2 @cpu(0)>\n\n[2.]\n<NDArray 1 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "2.0"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=18}
print(F.flip)
for i in  F.flatten():
    print(i)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<bound method NDArray.flip of \n[[1. 2.]\n [2. 3.]]\n<NDArray 2x2 @cpu(0)>>\n\n[1. 2.]\n<NDArray 2 @cpu(0)>\n\n[2. 3.]\n<NDArray 2 @cpu(0)>\n"
 }
]
```

```{.python .input  n=5}
def relu(X):
    return nd.maximum(X, 0)
```

```{.python .input  n=7}
print(relu(F))
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  2.]\n [ 2.  3.]]\n<NDArray 2x2 @cpu(0)>\n"
 }
]
```

```{.python .input  n=18}
F*F#NDArray 相乘 每个对应元素相乘
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  4.]\n [ 4.  9.]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们经常需要创建随机数组，即每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。以下代码创建数组，`它的元素服从均值0标准差1的正态分布`。

```{.python .input  n=31}
y =.01* nd.random_normal(shape=(3, 4))#nd.random_normal(0，1，shape=(3, 4))
y
```

```{.json .output n=31}
[
 {
  "data": {
   "text/plain": "\n[[-0.00971219 -0.00582562  0.00371708  0.00930007]\n [-0.01422575 -0.0051762   0.02008832  0.00286308]\n [ 0.00560459  0.0096976  -0.00528537 -0.0188909 ]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 31,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input  n=13}
y.shape
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "(3, 4)"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input  n=14}
y.size
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "12"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input  n=15}
x + y
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "\n[[ 0.19961983  0.83117795  1.93632793  1.35744405]\n [ 1.77932847 -0.01030731  0.60842693  2.31661868]\n [ 0.56707376  1.71535993  1.92541552  0.09504914]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

乘法：

```{.python .input  n=16}
x * y
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "\n[[-0.80038017 -0.16882208  0.93632793  0.35744399]\n [ 0.77932847 -1.01030731 -0.39157307  1.31661868]\n [-0.43292624  0.71535987  0.92541558 -0.90495086]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

指数运算：

```{.python .input  n=17}
nd.exp(y)
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "\n[[ 0.44915816  0.84465915  2.55059814  1.42967045]\n [ 2.18000793  0.36410707  0.67599267  3.73078513]\n [ 0.64860833  2.04492235  2.52291656  0.40456176]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

也可以转置一个矩阵然后计算矩阵乘法：

```{.python .input  n=24}
nd.dot(x, y.T)
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "\n[[ 0.32456964  0.69406676  0.30289841]\n [ 0.32456964  0.69406676  0.30289841]\n [ 0.32456964  0.69406676  0.30289841]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们会在之后的线性代数一章讲解这些运算符。

## 广播（Broadcasting）
当二元操作符左右两边ndarray形状不一样时，系统会尝试将其复制到一个共同的形状。例如`a`的第0维是3,
`b`的第0维是1，那么`a+b`时会将`b`沿着第0维复制3遍：

```{.python .input  n=26}
a = nd.arange(3).reshape((3,1))
b = nd.arange(2).reshape((1,2))
print('a:', a)
print('b:', b)
print('a+b:', a+b)

```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "a: \n[[ 0.]\n [ 1.]\n [ 2.]]\n<NDArray 3x1 @cpu(0)>\nb: \n[[ 0.  1.]]\n<NDArray 1x2 @cpu(0)>\na+b: \n[[ 0.  1.]\n [ 1.  2.]\n [ 2.  3.]]\n<NDArray 3x2 @cpu(0)>\n"
 }
]
```

## 跟NumPy的转换

ndarray可以很方便地同numpy进行转换

```{.python .input  n=27}
import numpy as np
x = np.ones((2,3))
y = nd.array(x)  # numpy -> mxnet
z = y.asnumpy()  # mxnet -> numpy
print([z, y])
```

```{.json .output n=27}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[array([[ 1.,  1.,  1.],\n       [ 1.,  1.,  1.]], dtype=float32), \n[[ 1.  1.  1.]\n [ 1.  1.  1.]]\n<NDArray 2x3 @cpu(0)>]\n"
 }
]
```

```{.python .input  n=29}
type(y)
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "mxnet.ndarray.ndarray.NDArray"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 替换操作

在前面的样例中，我们为每个操作新开内存来存储它的结果。例如，如果我们写`y = x + y`,
我们会把`y`从现在指向的实例转到新建的实例上去。我们可以用Python的`id()`函数来看这个是怎么执行的：

```{.python .input  n=30}
x = nd.ones((3, 4))
y = nd.ones((3, 4))

before = id(y)
y = y + x
id(y) == before
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "False"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以把结果通过`[:]`写到一个之前开好的数组里：

```{.python .input  n=31}
z = nd.zeros_like(x)
before = id(z)
z[:] = x + y
id(z) == before
```

```{.json .output n=31}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 31,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

但是这里我们还是为`x+y`创建了临时空间，然后再复制到`z`。需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input  n=32}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果现有的数组不会复用，我们也可以用 `x[:] = x + y` ，或者 `x += y` 达到这个目的：

```{.python .input  n=33}
before = id(x)
x += y
id(x) == before
```

```{.json .output n=33}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 33,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 截取（Slicing）

MXNet NDArray 提供了各种截取方法。截取 x 的 index 为 1、2 的行：

```{.python .input  n=37}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[1:3]
```

```{.json .output n=37}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x:  \n[[ 0.  1.  2.]\n [ 3.  4.  5.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 3.  4.  5.]\n [ 6.  7.  8.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 37,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

以及直接`写入指定位置`：

```{.python .input  n=38}
x[1,2] = 9.0
x
```

```{.json .output n=38}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.]\n [ 3.  4.  9.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 38,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

多维截取：

```{.python .input  n=39}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[1:2,1:3]
```

```{.json .output n=39}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x:  \n[[ 0.  1.  2.]\n [ 3.  4.  5.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 4.  5.]]\n<NDArray 1x2 @cpu(0)>"
  },
  "execution_count": 39,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

多维写入：

```{.python .input  n=40}
x[1:2,1:3] = 9.0
x
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.]\n [ 3.  9.  9.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 总结

ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见[NDArray
API文档](https://mxnet.incubator.apache.org/api/python/ndarray.html)。
**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/745)

# add about reshape and transpose
reshape:

```{.python .input  n=47}
x1=nd.array([[1,2,3],
         [4,5,6],
         [7,8,9]])
```

```{.python .input  n=48}
x1.reshape(-1,3)
```

```{.json .output n=48}
[
 {
  "data": {
   "text/plain": "\n[[1. 2. 3.]\n [4. 5. 6.]\n [7. 8. 9.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 48,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=49}
x1.transpose((1,0))
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "\n[[1. 4. 7.]\n [2. 5. 8.]\n [3. 6. 9.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=50}
x1[0]
```

```{.json .output n=50}
[
 {
  "data": {
   "text/plain": "\n[1. 2. 3.]\n<NDArray 3 @cpu(0)>"
  },
  "execution_count": 50,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=51}
x2=nd.arange(3*2*3*4).reshape((3,2,3,4))
```

```{.python .input  n=52}
x2
```

```{.json .output n=52}
[
 {
  "data": {
   "text/plain": "\n[[[[ 0.  1.  2.  3.]\n   [ 4.  5.  6.  7.]\n   [ 8.  9. 10. 11.]]\n\n  [[12. 13. 14. 15.]\n   [16. 17. 18. 19.]\n   [20. 21. 22. 23.]]]\n\n\n [[[24. 25. 26. 27.]\n   [28. 29. 30. 31.]\n   [32. 33. 34. 35.]]\n\n  [[36. 37. 38. 39.]\n   [40. 41. 42. 43.]\n   [44. 45. 46. 47.]]]\n\n\n [[[48. 49. 50. 51.]\n   [52. 53. 54. 55.]\n   [56. 57. 58. 59.]]\n\n  [[60. 61. 62. 63.]\n   [64. 65. 66. 67.]\n   [68. 69. 70. 71.]]]]\n<NDArray 3x2x3x4 @cpu(0)>"
  },
  "execution_count": 52,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=57}
x2.reshape((-1,4))
```

```{.json .output n=57}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]\n [12. 13. 14. 15.]\n [16. 17. 18. 19.]\n [20. 21. 22. 23.]\n [24. 25. 26. 27.]\n [28. 29. 30. 31.]\n [32. 33. 34. 35.]\n [36. 37. 38. 39.]\n [40. 41. 42. 43.]\n [44. 45. 46. 47.]\n [48. 49. 50. 51.]\n [52. 53. 54. 55.]\n [56. 57. 58. 59.]\n [60. 61. 62. 63.]\n [64. 65. 66. 67.]\n [68. 69. 70. 71.]]\n<NDArray 18x4 @cpu(0)>"
  },
  "execution_count": 57,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=53}
x2.transpose((0,3,1,2))
```

```{.json .output n=53}
[
 {
  "data": {
   "text/plain": "\n[[[[ 0.  4.  8.]\n   [12. 16. 20.]]\n\n  [[ 1.  5.  9.]\n   [13. 17. 21.]]\n\n  [[ 2.  6. 10.]\n   [14. 18. 22.]]\n\n  [[ 3.  7. 11.]\n   [15. 19. 23.]]]\n\n\n [[[24. 28. 32.]\n   [36. 40. 44.]]\n\n  [[25. 29. 33.]\n   [37. 41. 45.]]\n\n  [[26. 30. 34.]\n   [38. 42. 46.]]\n\n  [[27. 31. 35.]\n   [39. 43. 47.]]]\n\n\n [[[48. 52. 56.]\n   [60. 64. 68.]]\n\n  [[49. 53. 57.]\n   [61. 65. 69.]]\n\n  [[50. 54. 58.]\n   [62. 66. 70.]]\n\n  [[51. 55. 59.]\n   [63. 67. 71.]]]]\n<NDArray 3x4x2x3 @cpu(0)>"
  },
  "execution_count": 53,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=54}
x2.transpose((0,1,3,2))
```

```{.json .output n=54}
[
 {
  "data": {
   "text/plain": "\n[[[[ 0.  4.  8.]\n   [ 1.  5.  9.]\n   [ 2.  6. 10.]\n   [ 3.  7. 11.]]\n\n  [[12. 16. 20.]\n   [13. 17. 21.]\n   [14. 18. 22.]\n   [15. 19. 23.]]]\n\n\n [[[24. 28. 32.]\n   [25. 29. 33.]\n   [26. 30. 34.]\n   [27. 31. 35.]]\n\n  [[36. 40. 44.]\n   [37. 41. 45.]\n   [38. 42. 46.]\n   [39. 43. 47.]]]\n\n\n [[[48. 52. 56.]\n   [49. 53. 57.]\n   [50. 54. 58.]\n   [51. 55. 59.]]\n\n  [[60. 64. 68.]\n   [61. 65. 69.]\n   [62. 66. 70.]\n   [63. 67. 71.]]]]\n<NDArray 3x2x4x3 @cpu(0)>"
  },
  "execution_count": 54,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=55}
3*5*5
```

```{.json .output n=55}
[
 {
  "data": {
   "text/plain": "75"
  },
  "execution_count": 55,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
