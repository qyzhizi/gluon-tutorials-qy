# 卷积神经网络 --- 使用Gluon

现在我们使用Gluon来实现[上一章的卷积神经网络](cnn-scratch.md)。

## 定义模型

下面是LeNet在Gluon里的实现，注意到我们不再需要实现去计算每层的输入大小，尤其是接在卷积后面的那个全连接层。

```{.python .input  n=1}
from mxnet.gluon import nn

net = nn.Sequential() #定义了一个Sequential的对象
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
```

```{.python .input  n=4}
net.add??#函数
```

```{.python .input  n=3}
nn.Conv2D?? #对象
```

```{.python .input  n=5}
print(net)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Conv2D(None -> 20, kernel_size=(5, 5), stride=(1, 1))\n  (1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n  (2): Conv2D(None -> 50, kernel_size=(3, 3), stride=(1, 1))\n  (3): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)\n  (4): Flatten\n  (5): Dense(None -> 128, Activation(relu))\n  (6): Dense(None -> 10, linear)\n)\n"
 }
]
```

## 获取数据和训练

剩下的跟上一章没什么不同，我们重用`utils.py`里定义的函数。

```{.python .input  n=2}
from mxnet import gluon
import sys
sys.path.append('..')
import utils

# 初始化
ctx = utils.try_gpu()
net.initialize(ctx=ctx)
print('initialize weight on', ctx)

# 获取数据
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 训练
loss = gluon.loss.SoftmaxCrossEntropyLoss()#实例化了一个对象 未初始化
#实例化trainer Applies an Optimizer on a set of Parameters.
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=10)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "initialize weight on cpu(0)\nStart training on  cpu(0)\nEpoch 0. Loss: 1.095, Train acc 0.59, Test acc 0.77, Time 30.8 sec\nEpoch 1. Loss: 0.446, Train acc 0.83, Test acc 0.86, Time 30.2 sec\nEpoch 2. Loss: 0.368, Train acc 0.86, Test acc 0.87, Time 31.3 sec\nEpoch 3. Loss: 0.332, Train acc 0.88, Test acc 0.88, Time 31.0 sec\nEpoch 4. Loss: 0.310, Train acc 0.88, Test acc 0.88, Time 31.7 sec\nEpoch 5. Loss: 0.290, Train acc 0.89, Test acc 0.88, Time 31.1 sec\nEpoch 6. Loss: 0.273, Train acc 0.90, Test acc 0.89, Time 38.7 sec\nEpoch 7. Loss: 0.260, Train acc 0.90, Test acc 0.90, Time 42.3 sec\nEpoch 8. Loss: 0.245, Train acc 0.91, Test acc 0.90, Time 35.7 sec\nEpoch 9. Loss: 0.237, Train acc 0.91, Test acc 0.90, Time 29.0 sec\n"
 }
]
```

```{.python .input  n=6}
gluon.Trainer??
```

```{.python .input  n=32}
utils.train??
```

```{.python .input  n=8}
loss??
```

```{.python .input  n=31}
print([ctx])
print(ctx)
ctx??
print(type([ctx]))

```

```{.json .output n=31}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[cpu(0)]\ncpu(0)\n<class 'list'>\n"
 }
]
```

```{.python .input  n=33}

```

```{.json .output n=33}
[
 {
  "ename": "AttributeError",
  "evalue": "'DataLoader' object has no attribute 'reset'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-33-ece6581ee8a3>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtrain_data\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreset\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m: 'DataLoader' object has no attribute 'reset'"
  ]
 }
]
```

## 结论

使用Gluon来实现卷积网络轻松加随意。

## 练习

再试试改改卷积层设定，是不是会比上一章容易很多？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/737)
