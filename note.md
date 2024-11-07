# Notes for Let's reproduce GPT-2

## 权重复用

* 在输入的token_embedding和输出的linear层，其大小是相同的，都是[Vocab_size,embed_size]
只是前者是一个查找表，后者是一个全连接。但是在embed空间内希望语义相近的两个词在一起的话，都是类似的。
* 所以在反传的时候梯度也是累加的。而且参数共享压缩了模型

## 训练加速

* torch 默认的所有参数以及激活值都是fp32，太大了
* most of the bottleneck in the training is the limitation of GPU bandwidth.The tensor core is always waiting fot data
* ![alt text](image.png)
* 混合精度：权重是fp32，而激活值和loss是bf16
* 使用torch.compile可以加速的原因：
* 知道整体网络架构所以可以优化
* 知道整个计算流程避免了反复移动并且保存中间结果，kernel fusion
* 使用flash attention加速,将q,k在seq维度上进行切分，然后计算。

## 优化cuda kernel的实现

* odd and even 用更大的，且是2的幂的vocab_size ,padding the input
