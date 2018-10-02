# 创建神经网络

前面的教程我们教了大家如何实现线性回归，多类Logistic回归和多层感知机。我们既展示了如何从0开始实现，也提供使用`gluon`的更紧凑的实现。因为前面我们主要关注在模型本身，所以只解释了如何使用`gluon`，但没说明他们是如何工作的。我们使用了`nn.Sequential`，它是`nn.Block`的一个简单形式，但没有深入了解它们。

本教程和接下来几个教程，我们将详细解释如何使用这两个类来定义神经网络、初始化参数、以及保存和读取模型。

我们重新把[多层感知机 --- 使用Gluon](../chapter_supervised-learning/mlp-gluon.md)里的网络定义搬到这里作为开始的例子（为了简单起见，这里我们丢掉了Flatten层）。

```{.python .input  n=4}
from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation="relu"))
    net.add(nn.Dense(10))

print(net)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 256, Activation(relu))\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

```{.python .input  n=3}
mxnet.gluon.Dense??
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Object `mxnet.gluon.Dense` not found.\n"
 }
]
```

## 使用 `nn.Block` 来定义

事实上，`nn.Sequential`是`nn.Block`的简单形式。我们先来看下如何使用`nn.Block`来实现同样的网络。

```{.python .input  n=13}
class MLP(nn.Block):#创建nn.Block的子类 MLP ，继承了nn.Block的属性和方法
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))
    
   
    
```

可以看到`nn.Block`的使用是通过创建一个它子类的类，其中至少包含了两个函数。

- `__init__`：创建参数。上面例子我们使用了包含了参数的`dense`层
- `forward()`：定义网络的计算

我们所创建的类的使用跟前面`net`没有太多不一样。

```{.python .input  n=15}
net2 = MLP()
print(net2)
net2.initialize()#调用的是nn.Block的方法
x = nd.random.uniform(shape=(4,20))
y = net2(x) #用初始化的参数和x 运算
y
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "MLP(\n  (dense0): Dense(None -> 256, linear)\n  (dense1): Dense(None -> 10, linear)\n)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 0.02261425 -0.01928135 -0.04386696 -0.05276278 -0.02026141  0.03861122\n   0.02406352 -0.03705024 -0.06364935 -0.06602409]\n [ 0.02106983 -0.01929314 -0.00597952 -0.03301905 -0.00101984  0.04826394\n   0.00532017 -0.02418715 -0.0364222  -0.04529519]\n [ 0.0325794  -0.02929417 -0.01193233 -0.03553995  0.0098096   0.05884267\n  -0.02304009 -0.07396501 -0.06144218 -0.05524419]\n [ 0.0132762  -0.02252256 -0.02241572 -0.04267278 -0.00365663  0.05374448\n   0.00203513 -0.02124541 -0.03196853 -0.04970155]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=17}
nn.Block
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "mxnet.gluon.block.Block"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=16}
nn.Dense
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "mxnet.gluon.nn.basic_layers.Dense"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如何定义创建和使用`nn.Dense`比较好理解。接下来我们仔细看下`MLP`里面用的其他命令：

- `super(MLP, self).__init__(**kwargs)`：这句话调用`nn.Block`的`__init__`函数，它提供了`prefix`（指定名字）和`params`（指定模型参数）两个参数。我们会之后详细解释如何使用。

- `self.name_scope()`：调用`nn.Block`提供的`name_scope()`函数。`nn.Dense`的定义放在这个`scope`里面。它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。默认自动会自动生成前缀，我们也可以在创建的时候手动指定。推荐在构建网络时，每个层至少在一个`name_scope()`里。

```{.python .input  n=18}
print('default prefix:', net2.dense0.name)

net3 = MLP(prefix='another_mlp_')
print('customized prefix:', net3.dense0.name)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "default prefix: mlp6_dense0\ncustomized prefix: another_mlp_dense0\n"
 }
]
```

大家会发现这里并没有定义如何求导，或者是`backward()`函数。事实上，系统会使用`autograd`对`forward()`自动生成对应的`backward()`函数。

## `nn.Block`到底是什么东西？

在`gluon`里，`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们可以（近似）无限地嵌套`nn.Block`来构建新的`nn.Block`。

`nn.Block`主要提供这个东西

1. 存储参数
2. 描述`forward`如何执行
3. 自动求导

## 那么现在可以解释`nn.Sequential`了吧

`nn.Sequential`是一个`nn.Block`容器，它通过`add`来添加`nn.Block`。它自动生成`forward()`函数，其就是把加进来的`nn.Block`逐一运行。

一个简单的实现是这样的：

```{.python .input  n=19}
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

可以跟`nn.Sequential`一样的使用这个自定义的类：

```{.python .input  n=20}
net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
y
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "\n[[ 0.03204657  0.02124143  0.04759127 -0.01789677 -0.00348268  0.00399428\n  -0.00574583  0.05751251  0.01061102  0.07751886]\n [ 0.03958406 -0.01001829  0.05205595 -0.00073354 -0.01654139 -0.01459009\n   0.01740491  0.02927153  0.0145517   0.03818714]\n [ 0.01666046  0.03472541  0.02826045 -0.02640499 -0.01061744  0.01593906\n  -0.01907374 -0.00226934  0.00347918  0.06857711]\n [ 0.0382334   0.01357234  0.01822614 -0.01709829 -0.02348557 -0.01937572\n   0.01163109  0.01251305  0.00522111  0.04860325]]\n<NDArray 4x10 @cpu(0)>"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到，`nn.Sequential`的主要好处是定义网络起来更加简单。但`nn.Block`可以提供更加灵活的网络定义。考虑下面这个例子

```{.python .input  n=21}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))

    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x
```

看到这里我们直接手动创建和初始了权重`weight`，并重复用了`dense`的层。测试一下：

```{.python .input  n=22}
fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)
```

```{.json .output n=22}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(4, 256)\n"
 }
]
```

## `nn.Block`和`nn.Sequential`的嵌套使用

现在我们知道了`nn`下面的类基本都是`nn.Block`的子类，他们可以很方便地嵌套使用。

```{.python .input  n=23}
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)

    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): RecMLP(\n    (net): Sequential(\n      (0): Dense(None -> 256, Activation(relu))\n      (1): Dense(None -> 128, Activation(relu))\n    )\n    (dense): Dense(None -> 64, linear)\n  )\n  (1): Dense(None -> 10, linear)\n)\n"
 }
]
```

## 总结

不知道你同不同意，通过`nn.Block`来定义神经网络跟玩积木很类似。

## 练习

如果把`RecMLP`改成`self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]`，`forward`就用for loop来实现，会有什么问题吗？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/986)
