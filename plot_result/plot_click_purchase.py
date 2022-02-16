import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import config

# 20000, 40000, 80000, 250000, 400000, 700000
# CVR
cvr = []
cvr.append([0.036167146974063404, 0.022731439046746106, 0.017520805957074025, 0.015885837372105548, 0.014080459770114942, 0.0073898906296186815])
cvr.append([0.03767569917403275, 0.02214324767632586, 0.017593397046046916, 0.015055787068154322, 0.014533176746917205, 0.008632040965618142])
cvr.append([0.04024478041756659, 0.023206751054852322, 0.019052271617000488, 0.016240191061071305, 0.015020680647267977, 0.010690850106908501])
cvr.append([0.03911179560265266, 0.023564910214895495, 0.018513071612713462, 0.01590332168970081, 0.015197285866988892, 0.011240741178976265])
cvr.append([0.03904775624864943, 0.023965001151277917, 0.018588729770132373, 0.016024469798476052, 0.014708865895719212, 0.011270400758271378])
cvr.append([0.039049828566735785, 0.02445562021116933, 0.01848087458786782, 0.016447647963164686, 0.01473371069702103, 0.011263719989506376])

# click
click = []
click.append([6940, 5455, 4566, 3714, 3480, 3383])
click.append([13802, 10974, 9208, 7439, 6812, 6835])
click.append([27780, 21804, 18423, 14655, 13781, 13563])
click.append([87007, 67940, 57797, 46091, 43034, 42257])
click.append([138830, 108575, 92314, 73887, 69006, 67522])
click.append([243535, 189895, 161356, 129441, 121015, 118167])


# purchase
purchase = []
purchase.append([251, 124, 80, 59, 49, 25])
purchase.append([520, 243, 162, 112, 99, 59])
purchase.append([1118, 506, 351, 238, 207, 145])
purchase.append([3403, 1601, 1070, 733, 654, 475])
purchase.append([5421, 2602, 1716, 1184, 1015, 761])
purchase.append([9510, 4644, 2982, 2129, 1783, 1331])


for i in range(6):
    if i == 2:
        plt.title('Purchase')
    index = i
    plot_data = np.reshape(np.array(purchase[index]),(2,3))
    plt.subplot(2, 3, i+1)
    sns.heatmap(plot_data, cmap='Reds')

plt.show()

for i in range(5):
    if i == 2:
        plt.title('Purchase_diff')
    index = i
    diff = np.array(purchase[index+1]) - np.array(purchase[index])
    plot_data = np.reshape(diff,(2,3))
    plt.subplot(2, 3, i+1)
    sns.heatmap(plot_data, cmap='Reds')

plt.show()