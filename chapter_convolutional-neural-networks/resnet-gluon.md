# ResNet：深度残差网络

当大家还在惊叹GoogLeNet用结构化的连接纳入了大量卷积层的时候，微软亚洲研究院的研究员已经在设计更深但结构更简单的网络[ResNet](https://arxiv.org/abs/1512.03385)。他们凭借这个网络在2015年的Imagenet竞赛中大获全胜。

ResNet有效的解决了深度卷积神经网络难训练的问题。这是因为在误差反传的过程中，梯度通常变得越来越小，从而权重的更新量也变小。这个导致远离损失函数的层训练缓慢，随着层数的增加这个现象更加明显。之前有两种常用方案来尝试解决这个问题：

1. 按层训练。先训练靠近数据的层，然后慢慢的增加后面的层。但效果不是特别好，而且比较麻烦。
2. 使用更宽的层（增加输出通道）而不是更深来增加模型复杂度。但更宽的模型经常不如更深的效果好。

ResNet通过增加跨层的连接来解决梯度逐层回传时变小的问题。虽然这个想法之前就提出过了，但ResNet真正的把效果做好了。

下图演示了一个跨层的连接。

![](../img/residual.svg)


最底下那层的输入不仅仅是输出给了中间层，而且其与中间层结果相加进入最上层。这样在梯度反传时，最上层梯度可以直接跳过中间层传到最下层，从而避免最下层梯度过小情况。

为什么叫做残差网络呢？我们可以将上面示意图里的结构拆成两个网络的和，一个一层，一个两层，最下面层是共享的。

![](../img/residual2.svg)

在训练过程中，左边的网络因为更简单所以更容易训练。这个小网络没有拟合到的部分，或者说残差，则被右边的网络抓取住。所以直观上来说，即使加深网络，跨层连接仍然可以使得底层网络可以充分的训练，从而不会让训练更难。

## Residual块

ResNet沿用了VGG的那种全用$3\times 3$卷积，但在卷积和池化层之间加入了批量归一层来加速训练。每次跨层连接跨过两层卷积。这里我们定义一个这样的残差块。注意到如果输入的通道数和输出不一样时（`same_shape=False`），我们使用一个额外的$1\times 1$卷积来做通道变化，同时使用`strides=2`来把长宽减半。

```{.python .input  n=1}
from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                              strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                  strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)
```

```{.python .input  n=2}
channels=3
conv1=nn.Conv2D(channels,kernel_size=3,padding=1,strides=2)
bn1=nn.BatchNorm()
conv2=nn.Conv2D(channels,kernel_size=3,padding=1)
bn2=nn.BatchNorm()


conv1.initialize()
bn1.initialize()
conv2.initialize()
bn2.initialize()
```

```{.python .input  n=3}
padding=1;strides=2;kernel_size=3
(6+2*padding-kernel_size)/strides+1
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "3.5"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=4}
x=nd.random.uniform(shape=(2,1,6,6));print("x=",x)
out1=conv1(x)
print(out1.shape);print(out1[0][0][0]);print(out1)
print("params=",conv1.params,'type-params=',type(conv1.params))
out2=bn1(out1)
out2_relu=nd.relu(out2)
print(out2.shape);print(out2[0][0][0])
print(out2_relu.shape);print(out2_relu[0][0][0])
out3=conv2(out2_relu)
print("out3",out3.shape);print(out3[0][0][0])
out4=bn2(out3)
print(out4.shape);print(out4[0][0][0])
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x= \n[[[[5.4881352e-01 5.9284461e-01 7.1518934e-01 8.4426576e-01\n    6.0276335e-01 8.5794562e-01]\n   [5.4488319e-01 8.4725171e-01 4.2365479e-01 6.2356371e-01\n    6.4589411e-01 3.8438171e-01]\n   [4.3758720e-01 2.9753461e-01 8.9177299e-01 5.6712978e-02\n    9.6366274e-01 2.7265629e-01]\n   [3.8344151e-01 4.7766513e-01 7.9172504e-01 8.1216872e-01\n    5.2889490e-01 4.7997716e-01]\n   [5.6804454e-01 3.9278480e-01 9.2559665e-01 8.3607876e-01\n    7.1036056e-02 3.3739617e-01]\n   [8.7129295e-02 6.4817190e-01 2.0218398e-02 3.6824155e-01\n    8.3261985e-01 9.5715517e-01]]]\n\n\n [[[4.1702199e-01 9.9718481e-01 7.2032452e-01 9.3255734e-01\n    1.1438108e-04 1.2812445e-01]\n   [3.0233258e-01 9.9904054e-01 1.4675589e-01 2.3608898e-01\n    9.2338592e-02 3.9658073e-01]\n   [1.8626021e-01 3.8791075e-01 3.4556073e-01 6.6974604e-01\n    3.9676747e-01 9.3553907e-01]\n   [5.3881675e-01 8.4631091e-01 4.1919452e-01 3.1327352e-01\n    6.8521953e-01 5.2454817e-01]\n   [2.0445225e-01 4.4345289e-01 8.7811744e-01 2.2957721e-01\n    2.7387597e-02 5.3441393e-01]\n   [6.7046750e-01 9.1396201e-01 4.1730481e-01 4.5720482e-01\n    5.5868983e-01 4.3069857e-01]]]]\n<NDArray 2x1x6x6 @cpu(0)>\n(2, 3, 3, 3)\n\n[0.05296906 0.125489   0.13221557]\n<NDArray 3 @cpu(0)>\n\n[[[[ 5.2969057e-02  1.2548900e-01  1.3221557e-01]\n   [ 5.8253027e-02  1.6790406e-01  1.3188377e-01]\n   [ 4.2223692e-02  1.2610450e-01  1.2203022e-01]]\n\n  [[ 5.7919845e-03 -1.2149776e-03 -3.2772571e-02]\n   [ 4.5316283e-02  1.7642543e-02  5.4286368e-02]\n   [ 5.1302858e-02  8.5140169e-02 -3.3094224e-02]]\n\n  [[-1.4739353e-04 -6.6965081e-02 -9.1112986e-02]\n   [ 1.7492630e-02  2.3380067e-02  1.6161414e-02]\n   [ 2.8164623e-02  1.6951127e-02  1.0999765e-02]]]\n\n\n [[[ 3.1607538e-02  1.4323843e-01  5.9491329e-02]\n   [ 6.5711603e-02  1.3593936e-01  1.2207668e-01]\n   [ 5.2087810e-02  1.6243905e-01  5.6988262e-02]]\n\n  [[ 2.2546608e-02 -7.3940954e-03 -4.7148541e-02]\n   [ 2.7769720e-02  8.0296518e-03 -1.8847086e-02]\n   [ 2.8663103e-02  2.6707999e-02  1.9599225e-02]]\n\n  [[ 1.6411571e-02 -9.7381294e-02 -4.8537951e-02]\n   [ 1.9173989e-02  5.9463321e-03 -4.6173915e-02]\n   [ 2.3662455e-02  2.1171637e-03  7.8784470e-03]]]]\n<NDArray 2x3x3x3 @cpu(0)>\nparams= conv0_ (\n  Parameter conv0_weight (shape=(3, 1, 3, 3), dtype=<class 'numpy.float32'>)\n  Parameter conv0_bias (shape=(3,), dtype=<class 'numpy.float32'>)\n) type-params= <class 'mxnet.gluon.parameter.ParameterDict'>\n(2, 3, 3, 3)\n\n[0.05296879 0.12548837 0.13221492]\n<NDArray 3 @cpu(0)>\n(2, 3, 3, 3)\n\n[0.05296879 0.12548837 0.13221492]\n<NDArray 3 @cpu(0)>\nout3 (2, 3, 3, 3)\n\n[-0.01782976 -0.01803267 -0.00552791]\n<NDArray 3 @cpu(0)>\n(2, 3, 3, 3)\n\n[-0.01782967 -0.01803258 -0.00552788]\n<NDArray 3 @cpu(0)>\n"
 }
]
```

输入输出通道相同：

```{.python .input  n=23}
blk = Residual(3)
blk.initialize()
```

```{.python .input  n=25}
x = nd.random.uniform(shape=(4, 3, 6, 6))
blk(x).shape
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "(4, 3, 6, 6)"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=26}
print(blk)
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Residual(\n  (conv1): Conv2D(3 -> 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=3)\n  (conv2): Conv2D(3 -> 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=3)\n)\n"
 }
]
```

输入输出通道不同：

```{.python .input  n=16}
blk2 = Residual(4, same_shape=False)
blk2.initialize()
blk2(x).shape
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "(4, 4, 3, 3)"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=17}
print(blk2)
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Residual(\n  (conv1): Conv2D(3 -> 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n  (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=4)\n  (conv2): Conv2D(4 -> 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=4)\n  (conv3): Conv2D(3 -> 4, kernel_size=(1, 1), stride=(2, 2))\n)\n"
 }
]
```

## 构建ResNet

类似GoogLeNet主体是由Inception块串联而成，ResNet的主体部分串联多个Residual块。下面我们定义18层的ResNet。同样为了阅读更加容易，我们这里使用了多个`nn.Sequential`。另外注意到一点是，这里我们没用池化层来减小数据长宽，而是通过有通道变化的Residual块里面的使用`strides=2`的卷积层。

```{.python .input  n=7}
class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64)
            )
            # block 3
            b3 = nn.Sequential()
            b3.add(
                Residual(128, same_shape=False),
                Residual(128)
            )
            # block 4
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape=False),
                Residual(256)
            )
            # block 5
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape=False),
                Residual(512)
            )
            # block 6
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out
```

这里演示数据在块之间的形状变化：

```{.python .input  n=8}
net = ResNet(10, verbose=True)
net.initialize()

x = nd.random.uniform(shape=(4, 3, 96, 96))
y = net(x)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Block 1 output: (4, 64, 45, 45)\nBlock 2 output: (4, 64, 22, 22)\nBlock 3 output: (4, 128, 11, 11)\nBlock 4 output: (4, 256, 6, 6)\nBlock 5 output: (4, 512, 3, 3)\nBlock 6 output: (4, 10)\n"
 }
]
```

```{.python .input  n=10}
print(net)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "ResNet(\n  (net): Sequential(\n    (0): Conv2D(3 -> 64, kernel_size=(7, 7), stride=(2, 2))\n    (1): Sequential(\n      (0): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n      (1): Residual(\n        (conv1): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)\n        (conv2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)\n      )\n      (2): Residual(\n        (conv1): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)\n        (conv2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=64)\n      )\n    )\n    (2): Sequential(\n      (0): Residual(\n        (conv1): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)\n        (conv2): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)\n        (conv3): Conv2D(64 -> 128, kernel_size=(1, 1), stride=(2, 2))\n      )\n      (1): Residual(\n        (conv1): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)\n        (conv2): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=128)\n      )\n    )\n    (3): Sequential(\n      (0): Residual(\n        (conv1): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)\n        (conv2): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)\n        (conv3): Conv2D(128 -> 256, kernel_size=(1, 1), stride=(2, 2))\n      )\n      (1): Residual(\n        (conv1): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)\n        (conv2): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=256)\n      )\n    )\n    (4): Sequential(\n      (0): Residual(\n        (conv1): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)\n        (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)\n        (conv3): Conv2D(256 -> 512, kernel_size=(1, 1), stride=(2, 2))\n      )\n      (1): Residual(\n        (conv1): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)\n        (conv2): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n        (bn2): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=512)\n      )\n    )\n    (5): Sequential(\n      (0): AvgPool2D(size=(3, 3), stride=(3, 3), padding=(0, 0), ceil_mode=False)\n      (1): Dense(512 -> 10, linear)\n    )\n  )\n)\n"
 }
]
```

## 获取数据并训练

跟前面类似，但因为有批量归一化，所以使用了较大的学习率。

```{.python .input  n=48}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=96)

ctx = utils.try_gpu()
net = ResNet(10)
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

```{.json .output n=48}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Start training on  cpu(0)\nEpoch 0. Loss: 0.430, Train acc 0.85, Test acc 0.90, Time 4343.9 sec\n"
 }
]
```

## 结论

ResNet使用跨层通道使得训练非常深的卷积神经网络成为可能。同样它使用很简单的卷积层配置，使得其拓展更加简单。

## 练习

- 这里我们实现了ResNet 18，原论文中还讨论了更深的配置。尝试实现它们。（提示：参考论文中的表1）
- 原论文中还介绍了一个“bottleneck”架构，尝试实现它
- ResNet作者在[接下来的一篇论文](https://arxiv.org/abs/1603.05027)讨论了将Residual块里面的`Conv->BN->Relu`结构改成了`BN->Relu->Conv`（参考论文图1），尝试实现它


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1663)
