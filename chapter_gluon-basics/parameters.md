# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=1}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=2}
import sys
try:
    net = get_net()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Parameter sequential0_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
 }
]
```

正确的打开方式是这样

```{.python .input  n=3}
net.initialize()
net(x)
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "\n[[ 0.00212593  0.00365805]\n [ 0.00161272  0.00441845]\n [ 0.00204872  0.00352518]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：

```{.python .input  n=4}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "name:  sequential0_dense0 \nweight:  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>) \nbias:  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n"
 }
]
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=5}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "weight: \n[[-0.06206018  0.06491279 -0.03182812 -0.01631819 -0.00312688]\n [ 0.0408415   0.04370362  0.00404529 -0.0028032   0.00952624]\n [-0.01501013  0.05958354  0.04705103 -0.06005495 -0.02276454]\n [-0.0578019   0.02074406 -0.06716943 -0.01844618  0.04656678]]\n<NDArray 4x5 @cpu(0)>\nweight gradient \n[[ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]]\n<NDArray 4x5 @cpu(0)>\nbias: \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\nbias gradient \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

我们也可以通过`collect_params`来访问Block里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应Parameter的dict。既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=6}
params = net.collect_params()
print(params)
print(params['sequential0_dense0_bias'].data())
print(params.get('dense0_weight').data())
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential0_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n\n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n\n[[-0.06206018  0.06491279 -0.03182812 -0.01631819 -0.00312688]\n [ 0.0408415   0.04370362  0.00404529 -0.0028032   0.00952624]\n [-0.01501013  0.05958354  0.04705103 -0.06005495 -0.02276454]\n [-0.0578019   0.02074406 -0.06716943 -0.01844618  0.04656678]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

## 使用不同的初始函数来初始化

我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布

```{.python .input  n=7}
from mxnet import init
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.00359026  0.0302582  -0.01496244  0.01725933 -0.02177767]\n [ 0.01344385  0.00272668 -0.00392631 -0.03435376  0.01124353]\n [-0.00622001  0.00689361  0.02062465  0.00675439  0.01104854]\n [ 0.01147354  0.00579418 -0.04144352 -0.02262641  0.00582818]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

看得更加清楚点：

```{.python .input  n=9}
params.initialize(init=init.One(), force_reinit=True)
print(net[1].weight.data(), net[1].bias.data())
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 2x4 @cpu(0)> \n[ 0.  0.]\n<NDArray 2 @cpu(0)>\n"
 }
]
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 

## 延后的初始化

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input  n=10}
net = get_net()
net.collect_params()
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

然后我们初始化

```{.python .input  n=11}
net.initialize()
net.collect_params()
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

你会看到我们形状并没有发生变化，这是因为我们仍然不能确定权重形状。真正的初始化发生在我们看到数据时。

```{.python .input  n=15}
net(x)
print(x)
net.collect_params()
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.54881352  0.59284461  0.71518934  0.84426576  0.60276335]\n [ 0.85794562  0.54488319  0.84725171  0.42365479  0.62356371]\n [ 0.64589411  0.38438171  0.4375872   0.29753461  0.89177299]]\n<NDArray 3x5 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "sequential1_ (\n  Parameter sequential1_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential1_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这时候我们看到shape里面的0被填上正确的值了。

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input  n=16}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu"))
    net.add(nn.Dense(4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2))
```

初始化然后打印

```{.python .input  n=17}
net.initialize()
net(x)
print(net[1].weight.data())
print(net[2].weight.data())
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.00816047 -0.03040703  0.06714214 -0.05317248]\n [-0.01967777 -0.02854037 -0.00267491 -0.05337812]\n [ 0.02641256 -0.02548236  0.05326662 -0.01200318]\n [ 0.05855297 -0.06101935 -0.0396449   0.0269461 ]]\n<NDArray 4x4 @cpu(0)>\n\n[[-0.00816047 -0.03040703  0.06714214 -0.05317248]\n [-0.01967777 -0.02854037 -0.00267491 -0.05337812]\n [ 0.02641256 -0.02548236  0.05326662 -0.01200318]\n [ 0.05855297 -0.06101935 -0.0396449   0.0269461 ]]\n<NDArray 4x4 @cpu(0)>\n"
 }
]
```

## 自定义初始化方法

下面我们自定义一个初始化方法。它通过重载`_init_weight`来实现不同的初始化方法。（注意到Gluon里面`bias`都是默认初始化成0）

```{.python .input  n=20}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)

net = get_net()
net.initialize(MyInit())
net(x)
net[0].weight.data()
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 8.51868629  8.67597008  6.44238234  9.81094265  7.16644001]\n [ 6.24376583  8.78053284  7.8807869   6.98049164  7.96020985]\n [ 9.48019218  7.86125946  8.19460487  6.11540794  9.45777225]\n [ 9.76374531  8.40027809  7.23562717  7.24598885  9.23204327]]\n<NDArray 4x5 @cpu(0)>"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当然我们也可以通过`Parameter.set_data`来直接改写权重。注意到由于有延后初始化，所以我们通常可以通过调用一次`net(x)`来确定权重的形状先。

```{.python .input  n=26}
net = get_net()
net.initialize()
net(x)

print('default weight:', net[1].weight.data())

w = net[1].weight
print(w)
print(id(w)==id(net[1].weight))#查看两者的id是否相等
w.set_data(nd.ones(w.shape))

print('init to all 1s:', net[1].weight.data())
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "default weight: \n[[ 0.06583313 -0.03816195  0.02527625 -0.0343901 ]\n [-0.05805862 -0.06187592 -0.06210143 -0.00918167]]\n<NDArray 2x4 @cpu(0)>\nParameter sequential11_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\nTrue\ninit to all 1s: \n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 2x4 @cpu(0)>\n"
 }
]
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)
