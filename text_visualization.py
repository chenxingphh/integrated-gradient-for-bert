'''
@Author: Haihui Pan
@Date: 2022-07-01
@Desc: 使用matplotlib来对微聊会话积分梯度结果进行文本可视化
'''

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from os import path
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_text(word_list=None, word_score_list=None, score_tag=None):
    # 长度检验
    assert len(word_list) == len(word_score_list)

    # 子图初始化
    fig, ax = plt.subplots(figsize=(9, 6))

    # 去掉坐标轴
    plt.axis('off')

    # 当前所在的行列位置
    cur_x = -0.05
    cur_y = 0.99
    row_word_num = 0

    # 一行只显示20个字，超出的词汇就进行换行
    for i in range(len(word_list)):

        # 文字方框
        neg_box_style = dict(boxstyle="square", ec='lightskyblue', fc='lightblue',
                             alpha=word_score_list[i])
        pos_box_style = dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1, 0.8, 0.8),
                             alpha=word_score_list[i])

        # 根据分数正负选择不同的box style
        box_style = pos_box_style if score_tag[i] > 0 else neg_box_style

        ax.text(cur_x + row_word_num * 0.05, cur_y, word_list[i], fontsize=15, color="black",
                weight="light",
                horizontalalignment='left', bbox=box_style,
                )  # alpha=word_score_list[i]
        row_word_num += len(word_list[i])

        # 如果当前行满20个字
        if row_word_num >= 20:
            # 更新当前字
            row_word_num = 0
            # 换下一行
            cur_y -= 0.08

    plt.show()
    plt.close()


def plot_word_cloud(text, save_path, show_img=True):
    # 中文需要设置字体
    font = r'C:\Windows\Fonts\SimHei.ttf'

    # 显示的文本（会根据词频来调整大小）
    # text = ' '.join(['求职者 ', '你好', '求职者 ', '不', '考虑', '不好意思', '谢谢'])

    # 词云背景形状
    alice_mask = np.array(Image.open('data/皮卡丘.png'))

    # 用于过滤的停词
    stopwords = set(STOPWORDS)
    stopwords.add("said")

    wc = WordCloud(scale=4, background_color="white", mask=alice_mask, repeat=True, stopwords=stopwords, font_path=font)

    # generate word cloud
    wc.generate(text)

    # # store to file
    wc.to_file(save_path)

    if show_img:
        # 将x轴和y轴坐标隐藏
        plt.axis("off")
        plt.imshow(wc, interpolation="bilinear")
        plt.show()


def single_wordColud():
    '''
    绘制单个词一个圆形的词云
    '''
    # 中文需要设置字体
    font = r'C:\Windows\Fonts\SimHei.ttf'

    text = "第一 第二 第三 第四"
    # 产生一个以(150,150)为圆心,半径为130的圆形mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    wc = WordCloud(background_color="white", repeat=True, mask=mask, font_path=font)
    wc.generate(text)

    # 将x轴和y轴坐标隐藏
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


if __name__ == '__main__':
    word_score_list = [0.19, 0.11, 0.19, 0.40, 0.41, 0.89, 0.12]
    word_list = ['求职者 ', '你好', '求职者 ', '不', '考虑', '不好意思', '谢谢']
    score_tag = [1, -1, 1, 1, 1, 1, 1]
    plot_text(word_list, word_score_list, score_tag)

    # plot_word_cloud()
