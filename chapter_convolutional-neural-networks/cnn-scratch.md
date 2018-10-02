# 卷积神经网络 --- 从0开始

之前的教程里，在输入神经网络前我们将输入图片直接转成了向量。这样做有两个不好的地方：

- 在图片里相近的像素在向量表示里可能很远，从而模型很难捕获他们的空间关系。
- 对于大图片输入，模型可能会很大。例如输入是$256\times 256\times3$的照片（仍然远比手机拍的小），输出层是1000，那么这一层的模型大小是将近1GB.

这一节我们介绍卷积神经网络，其有效了解决了上述两个问题。

## 卷积神经网络

卷积神经网络是指主要由卷积层构成的神经网络。

### 卷积层

卷积层跟前面的全连接层类似，但输入和权重不是做简单的矩阵乘法，而是使用每次作用在一个窗口上的卷积。下图演示了输入是一个$4\times 4$矩阵，使用一个$3\times 3$的权重，计算得到$2\times 2$结果的过程。每次我们采样一个跟权重一样大小的窗口，让它跟权重做按元素的乘法然后相加。通常我们也是用卷积的术语把这个权重叫kernel或者filter。

![](../img/no_padding_no_strides.gif)

（图片版权属于vdumoulin@github）

我们使用`nd.Convolution`来演示这个。

```{.python .input  n=3}
from mxnet import nd

# 输入输出数据格式是 batch x channel x height x width，这里batch和channel都是1
# 权重格式是 output_channels x in_channels x height x width，这里input_filter和output_filter都是1。
w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1]) #num_filter 表示输出通道 

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
print(w.shape[2:])
print(w.shape[1])
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "input: \n[[[[ 0.  1.  2.]\n   [ 3.  4.  5.]\n   [ 6.  7.  8.]]]]\n<NDArray 1x1x3x3 @cpu(0)> \n\nweight: \n[[[[ 0.  1.]\n   [ 2.  3.]]]]\n<NDArray 1x1x2x2 @cpu(0)> \n\nbias: \n[ 1.]\n<NDArray 1 @cpu(0)> \n\noutput: \n[[[[ 20.  26.]\n   [ 38.  44.]]]]\n<NDArray 1x1x2x2 @cpu(0)>\n(2, 2)\n1\n"
 }
]
```

```{.python .input  n=19}
nd.Convolution??
```

我们可以控制如何移动窗口，和在边缘的时候如何填充窗口。下图演示了`stride=2`和`pad=1`。

![](../img/padding_strides.gif)

```{.python .input  n=2}
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[1],
                     stride=(2,2), pad=(1,1))#stride:移动窗口行和列都跳2 pad 行和列都补1

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "input: \n[[[[ 0.  1.  2.]\n   [ 3.  4.  5.]\n   [ 6.  7.  8.]]]]\n<NDArray 1x1x3x3 @cpu(0)> \n\nweight: \n[[[[ 0.  1.]\n   [ 2.  3.]]]]\n<NDArray 1x1x2x2 @cpu(0)> \n\nbias: \n[ 1.]\n<NDArray 1 @cpu(0)> \n\noutput: \n[[[[  1.   9.]\n   [ 22.  44.]]]]\n<NDArray 1x1x2x2 @cpu(0)>\n"
 }
]
```

当输入数据有多个通道的时候，每个通道会有对应的权重，然后会对每个通道做卷积之后在通道之间求和

$$conv(data, w, b) = \sum_i conv(data[:,i,:,:], w[:,i,:,:], b)$$

```{.python .input  n=4}
w = nd.arange(8).reshape((1,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
print(w.shape[0])
print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "1\ninput: \n[[[[  0.   1.   2.]\n   [  3.   4.   5.]\n   [  6.   7.   8.]]\n\n  [[  9.  10.  11.]\n   [ 12.  13.  14.]\n   [ 15.  16.  17.]]]]\n<NDArray 1x2x3x3 @cpu(0)> \n\nweight: \n[[[[ 0.  1.]\n   [ 2.  3.]]\n\n  [[ 4.  5.]\n   [ 6.  7.]]]]\n<NDArray 1x2x2x2 @cpu(0)> \n\nbias: \n[ 1.]\n<NDArray 1 @cpu(0)> \n\noutput: \n[[[[ 269.  297.]\n   [ 353.  381.]]]]\n<NDArray 1x1x2x2 @cpu(0)>\n"
 }
]
```

```{.python .input  n=5}
data2 = nd.arange(36).reshape((2,2,3,3))
data2
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "\n[[[[  0.   1.   2.]\n   [  3.   4.   5.]\n   [  6.   7.   8.]]\n\n  [[  9.  10.  11.]\n   [ 12.  13.  14.]\n   [ 15.  16.  17.]]]\n\n\n [[[ 18.  19.  20.]\n   [ 21.  22.  23.]\n   [ 24.  25.  26.]]\n\n  [[ 27.  28.  29.]\n   [ 30.  31.  32.]\n   [ 33.  34.  35.]]]]\n<NDArray 2x2x3x3 @cpu(0)>"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当输出需要多通道时，每个输出通道有对应权重，然后每个通道上做卷积。

$$conv(data, w, b)[:,i,:,:] = conv(data, w[i,:,:,:], b[i])$$

```{.python .input  n=4}
w = nd.arange(16).reshape((2,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))
b = nd.array([1,2])

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "input: \n[[[[  0.   1.   2.]\n   [  3.   4.   5.]\n   [  6.   7.   8.]]\n\n  [[  9.  10.  11.]\n   [ 12.  13.  14.]\n   [ 15.  16.  17.]]]]\n<NDArray 1x2x3x3 @cpu(0)> \n\nweight: \n[[[[  0.   1.]\n   [  2.   3.]]\n\n  [[  4.   5.]\n   [  6.   7.]]]\n\n\n [[[  8.   9.]\n   [ 10.  11.]]\n\n  [[ 12.  13.]\n   [ 14.  15.]]]]\n<NDArray 2x2x2x2 @cpu(0)> \n\nbias: \n[ 1.  2.]\n<NDArray 2 @cpu(0)> \n\noutput: \n[[[[  269.   297.]\n   [  353.   381.]]\n\n  [[  686.   778.]\n   [  962.  1054.]]]]\n<NDArray 1x2x2x2 @cpu(0)>\n"
 }
]
```

### 池化层（pooling）

因为卷积层每次作用在一个窗口，它对位置很敏感。池化层能够很好的缓解这个问题。它跟卷积类似每次看一个小窗口，然后选出窗口里面最大的元素，或者平均元素作为输出。

```{.python .input  n=5}
data = nd.arange(18).reshape((1,2,3,3))

max_pool = nd.Pooling(data=data, pool_type="max", kernel=(2,2))
avg_pool = nd.Pooling(data=data, pool_type="avg", kernel=(2,2))

print('data:', data, '\n\nmax pooling:', max_pool, '\n\navg pooling:', avg_pool)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "data: \n[[[[  0.   1.   2.]\n   [  3.   4.   5.]\n   [  6.   7.   8.]]\n\n  [[  9.  10.  11.]\n   [ 12.  13.  14.]\n   [ 15.  16.  17.]]]]\n<NDArray 1x2x3x3 @cpu(0)> \n\nmax pooling: \n[[[[  4.   5.]\n   [  7.   8.]]\n\n  [[ 13.  14.]\n   [ 16.  17.]]]]\n<NDArray 1x2x2x2 @cpu(0)> \n\navg pooling: \n[[[[  2.   3.]\n   [  5.   6.]]\n\n  [[ 11.  12.]\n   [ 14.  15.]]]]\n<NDArray 1x2x2x2 @cpu(0)>\n"
 }
]
```

下面我们可以开始使用这些层构建模型了。


## 获取数据

我们继续使用FashionMNIST（希望你还没有彻底厌烦这个数据）

```{.python .input  n=6}
import sys
sys.path.append('..')
from utils import load_data_fashion_mnist

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)
```

```{.python .input  n=30}
load_data_fashion_mnist??

```

```{.python .input  n=29}
train_data??
```

## 定义模型

因为卷积网络计算比全连接要复杂，这里我们默认使用GPU来计算。如果GPU不能用，默认使用CPU。（下面这段代码会保存在`utils.py`里可以下次重复使用）。

```{.python .input  n=7}
import mxnet as mx

try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()
ctx
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=32}
mx??
```

我们使用MNIST常用的LeNet，它有两个卷积层，之后是两个全连接层。注意到我们将权重全部创建在`ctx`上：

```{.python .input  n=8}
weight_scale = .01

# output channels = 20, kernel = (5,5)
W1 = nd.random_normal(shape=(20,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

# output channels = 50, kernel = (3,3)
W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(W2.shape[0], ctx=ctx)

# output dim = 128
W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

# output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
    param.attach_grad()
```

卷积模块通常是“卷积层-激活层-池化层”。然后转成2D矩阵输出给后面的全连接层。

```{.python .input  n=15}
def net(X, verbose=False):
    X = X.as_in_context(W1.context)
    #print(X.shape)#输入数据的格式
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv) #激活函数relu 输出是个ndarray
    #print(h1_activation.shape)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))#max pooling 输出是个ndarray
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    #print(h2_activation.shape)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    #print(h2.shape)
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    #print(h3_linear.shape)
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape) #第一层卷积块
        print('2nd conv block:', h2.shape)#
        print('1st dense:', h3.shape)#第一层全连接层
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear
```

```{.python .input  n=46}
train_data??
```

测试一下，输出中间结果形状（当然可以直接打印结果)和最终结果。

```{.python .input  n=16}
for data, _ in train_data:
    net(data, verbose=True)
    break
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(256, 50, 10, 10)\n(256, 50, 5, 5)\n1st conv block: (256, 20, 12, 12)\n2nd conv block: (256, 1250)\n1st dense: (256, 128)\n2nd dense: (256, 10)\noutput: \n[[  7.10813692e-05  -2.70309883e-05   1.84706645e-04 ...,   8.44229653e-05\n   -7.47105805e-05   2.87702878e-05]\n [  8.51226869e-05  -2.00891463e-05   1.13181348e-04 ...,   4.20196848e-05\n   -5.71249220e-05   9.24917185e-05]\n [  1.11248492e-05  -2.27673572e-05   8.17320033e-05 ...,   1.96618748e-05\n   -1.01790704e-06   2.29034613e-05]\n ..., \n [  6.13950397e-05   1.76016856e-05   1.34257396e-04 ...,   7.29362655e-05\n   -1.33139343e-04   1.13031565e-05]\n [  2.48712058e-05  -3.06258262e-05   1.12566820e-04 ...,   5.34600513e-05\n   -4.92884246e-05   8.36810796e-06]\n [  6.56771226e-05  -2.00740906e-05   1.61597287e-04 ...,   7.42339544e-05\n   -7.90345366e-05  -1.45943304e-05]]\n<NDArray 256x10 @cpu(0)>\n"
 }
]
```

## 训练

跟前面没有什么不同的，除了这里我们使用`as_in_context`将`data`和`label`都放置在需要的设备上。（下面这段代码也将保存在`utils.py`里方便之后使用）。

```{.python .input  n=56}
from mxnet import autograd as autograd
from utils import SGD, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .0001

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)#将label转换成ctx类型
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
```

```{.json .output n=56}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.242031, Train acc 0.911892, Test acc 0.902043\nEpoch 1. Loss: 0.242145, Train acc 0.911859, Test acc 0.901843\nEpoch 2. Loss: 0.241940, Train acc 0.911976, Test acc 0.901743\n"
 },
 {
  "ename": "KeyboardInterrupt",
  "evalue": "",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-56-a1191cc53243>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     18\u001b[0m         \u001b[0mSGD\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mparams\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m/\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m         \u001b[0mtrain_loss\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0mnd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masscalar\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m         \u001b[0mtrain_acc\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0maccuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masscalar\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1842\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m \u001b[0;34m!=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1843\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"The current array is not a scalar\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1844\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masnumpy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1845\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1846\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mastype\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/mxnet/ndarray/ndarray.py\u001b[0m in \u001b[0;36masnumpy\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m   1824\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1825\u001b[0m             \u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata_as\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_void_p\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1826\u001b[0;31m             ctypes.c_size_t(data.size)))\n\u001b[0m\u001b[1;32m   1827\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1828\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
  ]
 }
]
```

```{.python .input  n=52}
SGD??
```

## 结论

可以看到卷积神经网络比前面的多层感知的分类精度更好。事实上，如果你看懂了这一章，那你基本知道了计算视觉里最重要的几个想法。LeNet早在90年代就提出来了。不管你相信不相信，如果你5年前懂了这个而且开了家公司，那么你很可能现在已经把公司作价几千万卖个某大公司了。幸运的是，或者不幸的是，现在的算法已经更加高级些了，接下来我们会看到一些更加新的想法。

## 练习

- 试试改改卷积层设定，例如filter数量，kernel大小
- 试试把池化层从`max`改到`avg`
- 如果你有GPU，那么尝试用CPU来跑一下看看
- 你可能注意到比前面的多层感知机慢了很多，那么尝试计算下这两个模型分别需要多少浮点计算。例如$n\times m$和$m \times k$的矩阵乘法需要浮点运算 $2nmk$。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/736)
