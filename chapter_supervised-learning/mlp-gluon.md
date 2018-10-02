# 多层感知机 --- 使用Gluon

我们只需要稍微改动[多类Logistic回归](../chapter_crashcourse/softmax-regression-gluon.md)来实现多层感知机。

## 定义模型

唯一的区别在这里，我们加了一行进来。

```{.python .input  n=1}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## 读取数据并训练

```{.python .input  n=2}
import sys
sys.path.append('..')
from mxnet import ndarray as nd
from mxnet import autograd
import utils


batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:84: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)\n/home/qy/software/anaconda3/envs/gluon/lib/python3.6/site-packages/mxnet/gluon/data/vision/datasets.py:88: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead\n  data = np.fromstring(fin.read(), dtype=np.uint8)\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.706634, Train acc 0.741837, Test acc 0.836338\nEpoch 1. Loss: 0.461611, Train acc 0.829127, Test acc 0.850661\nEpoch 2. Loss: 0.408988, Train acc 0.849175, Test acc 0.834936\nEpoch 3. Loss: 0.384444, Train acc 0.858858, Test acc 0.852464\nEpoch 4. Loss: 0.357179, Train acc 0.868022, Test acc 0.873898\n"
 }
]
```

```{.python .input  n=3}
net.export('../params1')
```

```{.json .output n=3}
[
 {
  "ename": "AttributeError",
  "evalue": "'Sequential' object has no attribute 'export'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-3-64f6f96eaf07>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexport\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'../params1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m: 'Sequential' object has no attribute 'export'"
  ]
 }
]
```

## 结论

通过Gluon我们可以更方便地构造多层神经网络。

## 练习

- 尝试多加入几个隐含层，对比从0开始的实现。
- 尝试使用一个另外的激活函数，可以使用`help(nd.Activation)`或者[线上文档](https://mxnet.apache.org/api/python/ndarray.html#mxnet.ndarray.Activation)查看提供的选项。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/738)
