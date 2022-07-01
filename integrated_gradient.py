'''
@Author: Haihui Pan
@Date: 2022-07-01
@Desc: 构建用于计算Bert的积分梯度
'''
import torch
import numpy as np
from text_visualization import plot_text
from dataset_IMDB import IMDBPretrainDataset
from transformers import AutoTokenizer, ElectraTokenizerFast


def integrated_gradients(model, text_token, token_mask, y):
    '''
    用于计算NLP模型中的积分梯度；
    1、由于NLP是离散型输入，因此只能通过对embedding layer的权重进行线性插值来实现输入的线性插值
    2、计算之后得到的结果是（input_len,dim），计算每一个词向量累加和当做词的重要性
    :return:
    '''
    # 除embedding层外，固定住所有的模型参数
    for name, weight in model.named_parameters():
        if 'embedding' not in name:
            weight.requires_grad = False

    # 获取原始的embedding权重
    # init_embed_weight = model.word_attn.embedding.weight.data
    init_embed_weight = model.model.embeddings.word_embeddings.weight.data

    x = text_token

    # 获取输入之后的embedding
    init_word_embedding = init_embed_weight[x[0]]
    # print(init_word_embedding.size())

    # 获取baseline
    baseline = 0 * init_embed_weight
    baseline_word_embedding = baseline[x[0]]

    # 计算线性路径积分
    steps = 50
    # 对目标权重进行线性缩放计算的路径
    gradient_list = []

    for i in range(steps + 1):
        # 进行线性缩放
        scale_weight = baseline + float(i / steps) * (init_embed_weight - baseline)

        # 更换模型embedding的权重
        model.model.embeddings.word_embeddings.weight.data = scale_weight

        # 前馈计算
        pred = model(x, token_mask)

        # 直接取对应维度的输出(没经过softmax)
        target_pred = pred[:, y]
        # print(target_pred)

        # 计算梯度
        target_pred.backward()

        # 获取输入变量的梯度
        gradient_list.append(model.model.embeddings.word_embeddings.weight.grad[x[0]].numpy())
        # print(gradient_list[-1])
        # 梯度清零，防止累加
        model.zero_grad()

    # steps,input_len,dim
    gradient_list = np.asarray(gradient_list)
    # input_len,dim
    avg_gradient = np.average(gradient_list, axis=0)

    # x-baseline
    delta_x = init_word_embedding - baseline_word_embedding
    delta_x = delta_x.numpy()
    # print(delta_x.shape)

    # 获取积分梯度
    ig = avg_gradient * delta_x

    # 对每一行进行相加得到(input_len,)
    word_ig = np.sum(ig, axis=1)
    return word_ig


if __name__ == '__main__':
    # 加载模型
    model = torch.load(r'model/bert.pth')

    # 准备数据样例
    train_data = IMDBPretrainDataset('train')
    text_list, label_list = train_data.load_raw_data('train')

    # 找一个短一些的index
    len_list = [len(text.split(' ')) for text in text_list]
    # print(np.argmin(len_list))

    for i in range(len(text_list)):
        if len(text_list[i].split(' ')) <= 50:
            print(i, len(text_list[i].split(' ')), text_list[i])
            break

    text, label = text_list[i], label_list[i]
    # print(text, label)

    # 加载分词器
    model_path = r'E:\1 深度学习\5 Coding\Model\pretrain-en\bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded_input = tokenizer(text,
                              padding='max_length',  # True:以batch的最长当做最长; max_length: 以指定的当做最长
                              max_length=train_data.MAX_LEN,
                              truncation=True,  # padding='max_length',注释掉 truncation
                              )
    input_ids, attention_mask, labels = [], [], []
    input_ids.append(encoded_input.input_ids)
    attention_mask.append(encoded_input.attention_mask)
    labels.append(label)

    # 转Tensor
    input_ids = torch.tensor([i for i in input_ids], dtype=torch.long)
    attention_mask = torch.tensor([a for a in attention_mask], dtype=torch.long)
    labels = torch.tensor([int(train_data.label_dict[l]) for l in labels], dtype=torch.long)

    # 计算积分梯度
    word_ig = integrated_gradients(model, input_ids, attention_mask, labels)

    # 英文由于存在subword的情况，因此会比较麻烦
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 输出积分梯度
    last_index = tokens.index('[PAD]')
    tokens = tokens[:last_index]
    word_ig = word_ig[:last_index]

    score_tag = [1 if w > 0 else -1 for w in word_ig]

    # 最终进行可视化
    plot_text(tokens, abs(word_ig), score_tag)
