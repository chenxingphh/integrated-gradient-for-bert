# integrated-gradient-for-bert
[积分梯度(Integrated gradient,IG)](https://arxiv.org/pdf/1703.01365.pdf)是一种归因方法用于解释神经网络的预测过程，即将预测结果归因为输入的重要性。例如，在图像分类中，能够对输入的每一个像素给出对应的重要性；在文本分类中，对输入的每一个词汇给出重要性。

## 模型&数据集
    dataset: IMDB
    model: bert-tiny

## 积分梯度效果
* 其中红色代表的是对预测结果起到正向的word，蓝色表示对预测结果起到负向的结果；颜色越深就表示对结果影响越大。

![Alt text](https://github.com/chenxingphh/integrated-gradient-for-bert/blob/main/ig_result_temp.png)
