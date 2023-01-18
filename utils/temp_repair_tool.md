## 一个模型

1. 模型结构: 定义了模型的组成, 例如神经网络的层数和单元数.

2. 模型参数: 是在训练过程中学习到的值, 例如权重和偏差.

3. 训练数据: 用于训练模型的数据, 例如图像, 文本等.

4. 损失函数: 用于评估模型在训练数据上的表现, 并用于优化模型参数.

5. 优化算法: 用于优化模型参数, 使得损失函数最小.

6. 测试数据: 用于评估模型在未知数据上的表现.

##  权重参数保存的两种方式
- 第一种：将网络模型和对应的参数保存在一起；
- 第二种：模型和参数分离， 单独的保存模型的权重参数， 方便于网络模型修改后， 提取出对应层的参数权重；

**接下来下文主要是对于第一种保存的权重文件，如何转换做介绍。**

### 权重文件的类型
首先了解权重文件类型： 我们在使用pytorch进行模型训练的时候，
最后的权重文件实际上是一个字典，一个有序字典OrderedDict类 ，
关于这个类的一些常见操作，在下文拓展中有介绍。
### 权重转换
在明确权重文件其实就是一个字典类的时候，那么我们就能了解，
权重文件其实就是key+value,所谓key就是每一层的关键字，而value就是每一层的矩阵数据。

**权重转换核心思想: 就是把权重文件看作是一个字典，在我们新的模型中添加原来权重文件中存在的key以及value。**

## 拓展

### 字典OrderedDict类 
> 官网链接： https://docs.python.org/3/library/collections.html?highlight=ordereddict#collections.OrderedDict

>  **下面是小编总结的一些常见操作。**

#### 增
```
import collections

dic = collections.OrderedDict()
dic['k1'] = 'v1'
dic['k2'] = 'v2'
dic['k3'] = 'v3'
print(f'{dic=}')
# dic=OrderedDict([('k1', 'v1'), ('k2', 'v2'), ('k3', 'v3')])
```
dic.setdefault # 获取指定key的value，如果key不存在，则创建

#### 删
- dic.clear() 删除全部 
- dic.pop('key') 获取指定key的value，并在字典中删除 
- dic.popitem()  按照后进先出原则，删除最后加入的元素，返回key-value
#### 改
- dic.move_to_end('key') 指定一个key，把对应的key-value移到最后)
- dic.items() 返回由 "键值对组成元素" 的列表
#### 查
- dic.values 获取字典所有的value，返回一个列表




