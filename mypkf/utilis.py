# mypkg/utils.py の内容例
import matplotlib.pyplot as plt
import numpy as np

def plot_SP(np_WN, np1, title):
    # plot with np1
    x = np_WN       # Wave Numbers
    y = np1         # spectrum intensity
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, linestyle="-")
    plt.title(title)
    plt.xlabel("Wavenumvers ($cm^-$$^1$)")
    plt.ylabel("Counts")
    plt.show()
