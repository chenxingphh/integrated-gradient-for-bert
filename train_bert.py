'''
@Author: Haihui Pan
@Date: 2021-07-01
@Desc: Bert的训练
'''
import torch
import random
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from dataset_IMDB import IMDBPretrainDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model_bert import Bert
from seed_util import seed_torch

# 固定随机种子
seed_torch(512)


# 单个epoch的训练过程
def train(model, dataloader, optimizer, epoch):
    model.train()

    for idx, (token, mask, label) in enumerate(dataloader):
        optimizer.zero_grad()
        start_time = time.time()

        # 前馈传播
        predicted_label = model(input_ids=token.to(device), attention_mask=mask.to(device))
        # 使用label smooth
        loss = criterion(predicted_label, label.to(device))
        # 计算梯度
        loss.backward()

        # 参数梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # 更新参数
        optimizer.step()
        scheduler.step()

        # 预测结果
        y_pred = predicted_label.cpu().argmax(1).numpy()
        y_true = label.cpu().numpy()

        # loss
        loss_value = loss.data

        # 调用sklearn的接口来进行预测
        acc = accuracy_score(y_true, y_pred)
        p_value = precision_score(y_true, y_pred, average='macro')
        r_value = recall_score(y_true, y_pred, average='macro')
        f1 = (2 * p_value * r_value) / (p_value + r_value)

        # 计算耗时
        cost_time = time.time() - start_time

        print(
            '| epoch:{:3d} | batch:{:5d}/{:5d} | train_loss:{:8.2f} | train_acc:{:8.3f} |train_f1:{:8.3f}|train_P:{:8.3f}|train_R:{:8.3f}| time: {:5.2f}s'.format(
                epoch, idx, len(dataloader), loss_value, acc, f1, p_value, r_value, cost_time))


best_eval_acc = 0


def evaluate_dev(model, dataloader):
    model.eval()
    y_true_list = []
    y_pred_list = []

    for idx, (token, mask, label) in enumerate(dataloader):
        # 前馈传播
        predicted_label = model(token.to(device), mask.to(device))

        # 预测结果
        y_pred = predicted_label.cpu().argmax(1).numpy()
        y_true = label.cpu().numpy()

        # 添加预测结果
        y_true_list.extend(y_true.tolist())
        y_pred_list.extend(y_pred.tolist())

    # 调用sklearn的接口来进行预测
    acc = accuracy_score(y_true_list, y_pred_list)
    p_value = precision_score(y_true_list, y_pred_list, average='macro')
    r_value = recall_score(y_true_list, y_pred_list, average='macro')
    f1 = (2 * p_value * r_value) / (p_value + r_value)

    global best_eval_acc
    best_eval_acc = max(best_eval_acc, acc)

    # 保存测试集上acc最高的模型
    if best_eval_acc == acc:
        torch.save(model, 'model/bert.pth')

    print('-' * 100)
    print("| epoch:{:3d}| dev_acc:{:6.4f} | dev_f1:{:6.4f} | dev_P:{:6.4f} | dev_R:{:6.4f} | best_acc:{:6.4f}".format(
        epoch, acc, f1, p_value, r_value, best_eval_acc))
    print('-' * 100)

    print(classification_report(y_true=y_true_list, y_pred=y_pred_list))
    print(confusion_matrix(y_true_list, y_pred_list))


if __name__ == '__main__':
    # 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    warm_up_proportion = 0.1
    epochs = 3

    # 加载数据
    model_path = r'E:\1 深度学习\5 Coding\Model\pretrain-en\bert-tiny'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载训练集
    train_data = IMDBPretrainDataset(mode='train').get_dataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, )

    # 加载测试集
    test_data = IMDBPretrainDataset(mode='test').get_dataset(tokenizer=tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, )

    # 搭建模型
    model = Bert(model_path, pooling_type='cls')

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00003)
    # 更新次数
    total_step = len(train_data) * epochs // batch_size
    # 学习率计划
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_step * warm_up_proportion,
                                                num_training_steps=total_step)

    for epoch in range(1, epochs + 1):
        print("| epoch:{:3d}| learning_rate:{:6.5f}|".format(epoch, optimizer.param_groups[0]['lr']))

        train(model=model, dataloader=train_dataloader, optimizer=optimizer, epoch=epoch)

        # 设置动态学习率
        scheduler.step()

        # 评估测试集
        evaluate_dev(model, test_dataloader)
