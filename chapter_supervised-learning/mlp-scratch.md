# 多层感知机 --- 从0开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

```

```{.python .input  n=2}
data, label=test_data.dataset[0]
('example shape: ', data.shape, 'label:', label)
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "('example shape: ', (28, 28, 1), 'label:', 0)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一个到多个隐含层。

![](../img/multilayer-perceptron.png)

这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=3}
from mxnet import ndarray as nd

num_inputs = 28*28
num_outputs = 10

num_hidden = 256
weight_scale = .01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 激活函数

如果我们就用线性操作符来构造多层神经网络，那么整个模型仍然只是一个线性函数。这是因为

$$\hat{y} = X \cdot W_1 \cdot W_2 = X \cdot W_3 $$

这里$W_3 = W_1 \cdot W_2$。为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。这里我们使用ReLU

$$\textrm{rel}u(x)=\max(x, 0)$$

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0)
```

```{.python .input  n=5}
shuzu=nd.array([[1,0,-1,2],[2,-1,-1,3]])
print(shuzu)
relu(shuzu)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  0. -1.  2.]\n [ 2. -1. -1.  3.]]\n<NDArray 2x4 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 1.  0.  0.  2.]\n [ 2.  0.  0.  3.]]\n<NDArray 2x4 @cpu(0)>"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义模型

我们的模型就是将层（全连接）和激活函数（Relu）串起来：

```{.python .input  n=6}
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=7}
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=12}
from mxnet import autograd as autograd

learning_rate = .5

for epoch in range(9):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.307975, Train acc 0.885350, Test acc 0.876703\nEpoch 1. Loss: 0.299738, Train acc 0.888655, Test acc 0.879006\nEpoch 2. Loss: 0.289032, Train acc 0.893379, Test acc 0.884716\nEpoch 3. Loss: 0.282556, Train acc 0.895950, Test acc 0.883213\nEpoch 4. Loss: 0.274688, Train acc 0.897887, Test acc 0.878806\nEpoch 5. Loss: 0.271874, Train acc 0.898621, Test acc 0.887420\nEpoch 6. Loss: 0.264716, Train acc 0.901476, Test acc 0.884916\nEpoch 7. Loss: 0.260967, Train acc 0.902611, Test acc 0.886418\nEpoch 8. Loss: 0.254603, Train acc 0.905699, Test acc 0.890825\n"
 }
]
```

```{.python .input  n=13}
import matplotlib.pyplot as plt
def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]


def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
```

```{.python .input  n=16}
data, label=test_data.dataset[0:9]
show_images(data)
print（data[0]as np)
data, label=transform(data, label)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
```

```{.json .output n=16}
[
 {
  "ename": "SyntaxError",
  "evalue": "invalid character in identifier (<ipython-input-16-01ff5d7db2bd>, line 3)",
  "output_type": "error",
  "traceback": [
   "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-16-01ff5d7db2bd>\"\u001b[0;36m, line \u001b[0;32m3\u001b[0m\n\u001b[0;31m    print\uff08data[0])\u001b[0m\n\u001b[0m             ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid character in identifier\n"
  ]
 }
]
```

## 总结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 我们使用了 `weight_scale` 来控制权重的初始化值大小，增大或者变小这个值会怎么样？
- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/739)
