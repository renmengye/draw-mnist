from __future__ import division

from utils import logger
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

log = logger.get()

im_width = 28
im_height = 28
filter_size = 48


def sel_attn(img, g_x, g_y, delta, lg_var, lg_gamma=0.0):
    # [W, 1]
    span_x = (np.arange(im_width) + 1).reshape([-1, 1])
    # [H, 1]
    span_y = (np.arange(im_height) + 1).reshape([-1, 1])
    # [1, F]
    span_f = (np.arange(filter_size) + 1).reshape([1, -1])

    # [1, F]
    mu_x = g_x + delta * (span_f - filter_size / 2.0 - 0.5)
    log.info('Mu x: {}'.format(mu_x))
    # [1, F]
    mu_y = g_y + delta * (span_f - filter_size / 2.0 - 0.5)
    log.info('Mu y: {}'.format(mu_y))

    # [W, F]
    filter_x = (
        1 / np.sqrt(np.exp(lg_var)) / np.sqrt(2 * np.pi) *
        np.exp(-0.5 * (span_x - mu_x) * (span_x - mu_x) /
               np.exp(lg_var)))
    # [H, F]
    filter_y = (
        1 / np.sqrt(np.exp(lg_var)) / np.sqrt(2 * np.pi) *
        np.exp(-0.5 * (span_y - mu_y) * (span_y - mu_y) /
               np.exp(lg_var)))

    print filter_x.sum(0), filter_x.sum(1)
    print filter_y.sum(0), filter_y.sum(1)

    read = np.exp(lg_gamma) * filter_y.transpose().dot(img).dot(filter_x)
    write = np.exp(-lg_gamma) * filter_y.dot(read).dot(filter_x.transpose())

    # filter_y_inv = filter_y.transpose()
    # filter_y_inv = filter_y_inv / filter_y_inv.sum(0)
    # filter_x_inv = filter_x.transpose()
    # filter_x_inv = filter_x_inv / filter_x_inv.sum(0)
    # write = filter_y_inv.transpose().dot(read).dot(filter_x_inv)
    # print filter_x_inv.sum(0)
    # print filter_y_inv.sum(0)

    return read, write


if not os.path.exists('img.npy'):
    from data_api import mnist
    dataset = mnist.read_data_sets("../MNIST_data/", one_hot=True)
    img = dataset.train.images[0].reshape([im_height, im_width])
    np.save('img.npy', img)
else:
    img = np.load('img.npy')
    # Controller variables.
    num_img = 8
    f, axarr = plt.subplots(3, num_img)

    for ii in xrange(num_img):
        gg_x = 0.0
        gg_y = 0.0
        ddelta = 0.1 * (ii + 1)
        log.info('ddelta: {}'.format(ddelta))
        # lg_var = 0.01
        # lg_var = -3
        lg_var = 0.5 * (np.log(ddelta * im_width) - np.log(filter_size))
        log.error(lg_var)
        log.error(np.exp(lg_var))
        filter_area = filter_size * filter_size
        area = ddelta * im_height * ddelta * im_width
        lg_gamma = np.log(filter_area) - np.log(area)
        log.error(lg_gamma)
        # lg_gamma = 0.0

        g_x = (gg_x + 1) * (im_width + 1) / 2.0
        g_y = (gg_y + 1) * (im_height + 1) / 2.0
        delta = (max(im_width, im_height) - 1) / \
            float(filter_size - 1) * ddelta
        log.info('delta: {}'.format(delta))

        read, write = sel_attn(img, g_x, g_y, delta, lg_var, lg_gamma)

        axarr[0, ii].imshow(img, cmap=cm.Greys_r)
        axarr[0, ii].add_patch(
            patches.Rectangle(
                (g_x - delta * (filter_size - 1) / 2.0,
                 g_y - delta * (filter_size - 1) / 2.0),
                delta * (filter_size - 1),
                delta * (filter_size - 1),
                fill=False,
                color='r')
        )
        axarr[0, ii].text(0, -0.5, '[{:.2g}, {:.2g}]'.format(
            img.min(), img.max()),
            color=(0, 0, 0), size=8)
        axarr[0, ii].set_axis_off()
        axarr[1, ii].imshow(read, cmap=cm.Greys_r)
        axarr[1, ii].text(0, -0.5, '[{:.2g}, {:.2g}]'.format(
            read.min(), read.max()),
            color=(0, 0, 0), size=8)
        axarr[1, ii].set_axis_off()

        axarr[2, ii].imshow(write, cmap=cm.Greys_r)
        axarr[2, ii].text(0, -0.5, '[{:.2g}, {:.2g}]'.format(
            write.min(), write.max()),
            color=(0, 0, 0), size=8)
        axarr[2, ii].set_axis_off()
    plt.show()
