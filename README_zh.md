# 22种动物图片分类

本项目主要是对动物图片进行分类。文件分为训练集、测试集。训练集的图片标签来源于其所属的文件夹名，例如，ape文件夹内的所有图片的标签都是ape。

## 主要流程
### 主要流程和需要保留的文件：
1. 数据处理代码dataset.py，使得图片和标签可以一一对应便于后续训练
2. 训练代码train.py，利用训练集train保存多轮训练后的模型参数model.path，训练轮数自行选择。
3. 测试文件inference.py，加载模型参数model.path，并对测试集test的每一张图片输出结其属于的标签，以output.json文件形式整理。Json文件的样例如下：

{
    "0": "cat",
    "1": "ape",
    "2": "fish"
}
4. model.py中构建了三种模型可供使用，你也可以在这里接入自己喜欢的模型和加载预训练模型
