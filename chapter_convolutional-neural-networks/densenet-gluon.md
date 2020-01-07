# DenseNet：稠密连接的卷积神经网络

ResNet的跨层连接思想影响了接下来的众多工作。这里我们介绍其中的一个：[DenseNet](https://arxiv.org/pdf/1608.06993.pdf)。下图展示了这两个的主要区别：

![](../img/densenet.svg)

可以看到DenseNet里来自跳层的输出不是通过加法（`+`）而是拼接（`concat`）来跟目前层的输出合并。因为是拼接，所以底层的输出会保留的进入上面所有层。这是为什么叫“稠密连接”的原因

## 稠密块（Dense Block）

我们先来定义一个稠密连接块。DenseNet的卷积块使用ResNet改进版本的`BN->Relu->Conv`。每个卷积的输出通道数被称之为`growth_rate`，这是因为假设输出为`in_channels`，而且有`layers`层，那么输出的通道数就是`in_channels+growth_rate*layers`。

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

def conv_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )
    return out

class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x
```

我们验证下输出通道数是不是符合预期。

```{.python .input  n=2}
dblk = DenseBlock(2, 10)
dblk.initialize()

x = nd.random.uniform(shape=(4,3,8,8))
dblk(x).shape
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "(4, 23, 8, 8)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 过渡块（Transition Block）
因为使用拼接的缘故，每经过一次拼接输出通道数可能会激增。为了控制模型复杂度，这里引入一个过渡块，它不仅把输入的长宽减半，同时也使用$1\times1$卷积来改变通道数。

```{.python .input  n=4}
def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out
```

```{.python .input  n=7}
x.shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(4, 3, 8, 8)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

验证一下结果：

```{.python .input  n=5}
tblk = transition_block(10)
tblk.initialize()

tblk(x).shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(4, 10, 4, 4)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## DenseNet

    DenseNet的主体就是`交替串联稠密块和过渡块`。它使用全局的`growth_rate`使得配置更加简单。过渡层每次都将通道数减半。下面定义一个121层的DenseNet。

```{.python .input}
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10

def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net

```

```{.python .input  n=15}
net_2=dense_net()
print(net_2)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Conv2D(None -> 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))\n  (1): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n  (2): Activation(relu)\n  (3): MaxPool2D(size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)\n  (4): DenseBlock(\n    (net): Sequential(\n      (0): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (1): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (2): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (3): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (4): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (5): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n    )\n  )\n  (5): Sequential(\n    (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n    (1): Activation(relu)\n    (2): Conv2D(None -> 128, kernel_size=(1, 1), stride=(1, 1))\n    (3): AvgPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n  )\n  (6): DenseBlock(\n    (net): Sequential(\n      (0): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (1): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (2): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (3): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (4): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (5): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (6): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (7): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (8): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (9): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (10): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (11): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n    )\n  )\n  (7): Sequential(\n    (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n    (1): Activation(relu)\n    (2): Conv2D(None -> 320, kernel_size=(1, 1), stride=(1, 1))\n    (3): AvgPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n  )\n  (8): DenseBlock(\n    (net): Sequential(\n      (0): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (1): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (2): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (3): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (4): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (5): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (6): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (7): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (8): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (9): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (10): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (11): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (12): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (13): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (14): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (15): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (16): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (17): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (18): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (19): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (20): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (21): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (22): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (23): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n    )\n  )\n  (9): Sequential(\n    (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n    (1): Activation(relu)\n    (2): Conv2D(None -> 704, kernel_size=(1, 1), stride=(1, 1))\n    (3): AvgPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n  )\n  (10): DenseBlock(\n    (net): Sequential(\n      (0): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (1): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (2): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (3): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (4): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (5): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (6): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (7): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (8): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (9): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (10): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (11): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (12): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (13): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (14): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n      (15): Sequential(\n        (0): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n        (1): Activation(relu)\n        (2): Conv2D(None -> 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n      )\n    )\n  )\n  (11): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)\n  (12): Activation(relu)\n  (13): AvgPool2D(size=(1, 1), stride=(1, 1), padding=(0, 0), ceil_mode=False)\n  (14): Flatten\n  (15): Dense(None -> 10, linear)\n)\n"
 }
]
```

```{.python .input  n=12}
channels=5;channels//2;print(channels)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "5\n"
 }
]
```

## 获取数据并训练

因为这里我们使用了比较深的网络，所以我们进一步把输入减少到$32\times 32$来训练。

```{.python .input  n=13}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=32)

ctx = utils.try_gpu()
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

```{.json .output n=13}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Start training on  cpu(0)\nEpoch 0. Loss: 0.485, Train acc 0.83, Test acc 0.87, Time 3143.8 sec\n"
 }
]
```

## 结论

Desnet通过将ResNet里的`+`替换成`concat`从而获得更稠密的连接。

## 练习

- DesNet论文中提交的一个优点是其模型参数比ResNet更小，想想为什么？
- DesNet被人诟病的一个问题是内存消耗过多。真的会这样吗？可以把输入换成$224\times 224$（需要改最后的`AvgPool2D`大小），来看看实际（GPU）内存消耗。
- 这里的FashionMNIST有必要用100+层的网络吗？尝试将其改简单看看效果。


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1664)
