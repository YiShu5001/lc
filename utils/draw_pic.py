# 绘制图片的工具函数

import matplotlib.pyplot as plt
import numpy as np

def draw_pic(data, title, xlabel, ylabel):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 几种常用的绘图函数
def draw_line(data, title, xlabel, ylabel):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def draw_scatter(data, title, xlabel, ylabel):
    plt.scatter(data)
    plt.title(title)
    plt.xlabel(xlabel)