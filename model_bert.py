'''
@Author: Haihui Pan
@Date: 2022-07-01
@Desc: 预训练模型
'''

import torch
import torch.nn as nn
from transformers import AutoModel


class Bert(nn.Module):

    def __init__(self, model_path, pooling_type='cls'):
        super().__init__()

        # 加载预训练模型
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(model_path)

        # 获取Embedding层参数
        # model.embeddings.word_embeddings.weight

        # hidden dim
        # print(self.model.config.to_dict())
        self.hidden_size = self.model.config.to_dict()['hidden_size']

        # 池化类型
        self.pooling_type = pooling_type
        if self.pooling_type not in ['cls', 'last_avg', 'first_last_avg']:
            raise ValueError('The value of pooling_type can only be:cls, last_avg or first_last_avg]')

    def forward(self, input_ids, attention_mask):
        # pooling_type: ['cls','last_avg', 'first_last_avg']

        # 前馈计算
        # output: batch, seq_len, dim
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)

        # 对输出进行池化
        if self.pooling_type == 'cls':
            #  'cls': 取输出层一个token的dim
            final = output.last_hidden_state[:, 0]

        elif self.pooling_type == 'last_avg':
            # 'last_avg':对最终层进行平均池化
            # final: batch ,dim
            final = nn.AdaptiveAvgPool2d((1, self.hidden_size))(output.last_hidden_state).squeeze(1)

        elif self.pooling_type == 'first_last_avg':
            # 'first_last_avg': 取第一层和最后一层进行平均池化，之后将2者的结果再进行池化

            # 第一层和最后一层输出
            first_output = output.hidden_states[1]
            last_output = output.hidden_states[-1]

            # 进行池化
            # first_avg, last_avg: batch, dim
            first_avg = nn.AdaptiveAvgPool2d((1, self.hidden_size))(first_output).squeeze(1)
            last_avg = nn.AdaptiveAvgPool2d((1, self.hidden_size))(last_output).squeeze(1)

            # 将first_avg, last_avg进行拼接: batch, 2, dim
            concat_avg = torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1)

            # 再次平均池化
            final = nn.AdaptiveAvgPool2d((1, self.hidden_size))(concat_avg).squeeze(1)

        return final


if __name__ == '__main__':
    model_path = r'E:\1 深度学习\5 Coding\Model\pretrain-en\bert-tiny'
    model = Bert(model_path, pooling_type='cls')
