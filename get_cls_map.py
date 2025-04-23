import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y, support_label, indices):
    x_test, y_test, x_train, y_train = indices
    height = y.shape[0]
    width = y.shape[1]
    cls_labels = np.zeros((height, width))
    for i in range(len(x_test)):
        cls_labels[x_test[i], y_test[i]]=y_pred[i]
    for i in range(len(x_train)):
        cls_labels[x_train[i], y_train[i]]=y_pred[i]
    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([60, 91, 112]) / 255.
        # if item == 7:
        #    y[index] = np.array([255, 255, 255]) / 255.
        if item == 7:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 8:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 255]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def get_cls_map(y_pred, support_label, y, indices):
    cls_labels = get_classification_map(y_pred+1, y, support_label, indices)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    classification_map(y_re, y, 300,
                       '/mnt/HDD/data/zwj/model_2/my_model/paviac_pre.png')
    classification_map(gt_re, y, 300,
                       '/mnt/HDD/data/zwj/model_2/my_model/paviac_gt.png')
    print('------Get classification maps successful-------')