import os, re
from os.path import exists, join
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Recorder():
    def __init__(self, id, log=True):
        self.log = log
        now = datetime.now()
        date = now.strftime("%y-%m-%d")
        self.dir = f"./cache/{date}-{id}"
        if self.log:
            os.mkdir(self.dir)
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
        
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        print(args)
        print()
        for (i, x) in enumerate(models):
            print(x)
            print()

    def print(self, x=None):
        if x is not None:
            print(x, flush=True)
        else:
            print(flush=True)
        if self.log:
            if x is not None:
                print(x, file=self.f, flush=True)
            else:
                print(file=self.f, flush=True)

    def plot(self, tag, values, step):
        if self.log:
            self.writer.add_scalars(tag, values, step)


    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))



def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")



def split_paragraph(para):
    # 按完整句子分割
    # para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    # para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    # para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    # para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    # para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    # paralis = para.split("\n")

    # 按照标点符号分割
    paralis = re.split(r'([，。；：！？,.;:?!])', para)

    if len(paralis) ==1:
        return paralis[0][:255]+paralis[0][-255:]


    content_l = ''
    content_r = ''
    for i in paralis:
        if len(content_l)+len(i) <=255:
            content_l += i
        else:
            break
    for j in paralis[::-1]:
        if len(content_r)+len(j) <=255:
            content_r = j+content_r
        else:
            break

    if len(content_l+content_r) >0:
        return content_l+content_r
    else:
        return para[:255] + para[-255:]

