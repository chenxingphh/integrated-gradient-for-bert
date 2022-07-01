'''
@Author: Haihui Pan
@Date: 2022-07-01
@Desc: 构建用于训练的句子对数据集
'''
import torch
from torch.utils.data import Dataset, TensorDataset
import jieba
import re
from tqdm import tqdm
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from transformers import AutoTokenizer, ElectraTokenizerFast
from torchtext.datasets import IMDB


class IMDBPretrainDataset():
    '''用于预训练模型的数据集: e.g:<cls>A<sep>
    '''

    def __init__(self, mode):
        # 数据集模式
        self.mode = mode
        # 标签-数值映射
        self.label_dict = {'neg': 0, 'pos': 1}
        # 句子最大长度
        self.MAX_LEN = 300

    def load_raw_data(self, mode):
        '''
        加载原始文本数据集,只完成分词的部分
        :param mode:
        :return:
        '''
        if mode not in ['train', 'test']:
            raise ValueError('The value of mode can only be: train or test!')

        # 加载原始数据
        data_list = list(IMDB(split=mode))

        text_list, label_list = [], []
        for label, text in tqdm(data_list):
            # 对于文本从反向进行截取
            text_list.append(text)
            label_list.append(label)

        # 获取平均句子长度
        # s1_len_list = [len(s1) for s1 in text_list]
        # print('s1_len:', np.percentile(s1_len_list, q=(50, 75, 90, 99, 100)))
        # s1_len: [ 202.  329.  530. 1049. 2752.]

        return text_list, label_list

    def __len__(self):
        # 获取数据集数目
        return len(self.label_list)

    def get_dataset(self, tokenizer):
        # 将数据进行数字化，并使用TensorDataset进行包装返回
        text_list, label_list = self.load_raw_data(self.mode)

        # 获取数字化后的结果
        input_ids, attention_mask = [], []

        for text, label in tqdm(zip(text_list, label_list)):
            encoded_input = tokenizer(text,
                                      padding='max_length',  # True:以batch的最长当做最长; max_length: 以指定的当做最长
                                      max_length=self.MAX_LEN,
                                      truncation=True,  # padding='max_length',注释掉 truncation
                                      )
            input_ids.append(encoded_input.input_ids)
            attention_mask.append(encoded_input.attention_mask)

        # 转换为TensorDataset
        input_ids = torch.tensor([i for i in input_ids], dtype=torch.long)
        attention_mask = torch.tensor([a for a in attention_mask], dtype=torch.long)
        label_list = torch.tensor([int(self.label_dict[l]) for l in label_list], dtype=torch.long)

        # 转TensorDataset
        return TensorDataset(input_ids, attention_mask, label_list)


if __name__ == '__main__':
    # -----------------
    # 预训练模型数据集
    # -----------------
    model_path = r'E:\1 深度学习\5 Coding\Model\pretrain-en\bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_data = IMDBPretrainDataset(mode='train').get_dataset(tokenizer=tokenizer)
    test_data = IMDBPretrainDataset(mode='test').get_dataset(tokenizer=tokenizer)
