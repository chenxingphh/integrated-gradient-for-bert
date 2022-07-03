# integrated-gradient-for-bert
Integrated gradient (IG) is used to interpret Bert's classification results.

## 模型&数据集
    dataset: IMDB
    model: bert-tiny

## 积分梯度效果
* 其中红色代表的是对预测结果起到正向的word，蓝色表示对预测结果起到负向的结果；颜色越深就表示对结果影响越大。
![Alt text](https://github.com/chenxingphh/integrated-gradient-for-bert/blob/main/ig_result_temp.png)
