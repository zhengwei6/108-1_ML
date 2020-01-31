import numpy as np
from matplotlib import pyplot as plt

from univariate_rng import UnivariateRNG
from datasets import parse_mnist_dataset

def binarize_imgs(imgs): 
    imgs_bin = np.asarray(imgs > 0.5, dtype=np.float) 
    return imgs_bin

class EMAlgorithm(object): 
    def __init__(self, imgs, lbls, lbda_0=None, proba_0=None): 
        self.imgs = binarize_imgs(imgs)
        self.lbls = lbls

        self.n_classes = 10
        self.n_imgs = self.lbls.shape[0]
        self.img_x = 28
        self.img_y = 28

        self.rng = UnivariateRNG()

        if proba_0 is not None: 
            assert(proba_0.shape == (self.img_x, self.img_y, self.n_classes))
        else: 
            proba_0 = np.array(
                [self.rng.rand_uniform(0.25, 0.75) for i in range(self.img_x * self.img_y * self.n_classes)]
                ).reshape((self.img_x, self.img_y, self.n_classes))

            proba_0[:,:] =  proba_0[:,:] / np.sum(proba_0, axis=(0,1))
            # print(np.sum(proba_0, axis=(0,1)))

            # proba_0 = np.ones((self.img_x, self.img_y, self.n_classes))
            # proba_0 *= 0.4
        self.proba = proba_0

        if lbda_0 is not None:
            assert(lbda_0.shape[0] == self.n_classes)
        else: 
            lbda_0 = [1. / self.n_classes for i in range(self.n_classes)]
            # lbda_0 = np.random.random(self.n_classes)
        self.lbda = lbda_0

    def class_proba(self, imgs): 
        n_imgs = imgs.shape[2]
        p_c = np.ones((n_imgs, self.n_classes))
        for n in range(n_imgs): 
            for c in range(self.n_classes): 
                p_c[n, c] = np.nanprod(np.multiply(np.power(self.proba[:,:, c], imgs[:,:, n]), 
                    np.power(1 - self.proba[:, :, c], 1 - imgs[:,:,n]))) * self.lbda[c]
                # for i in range(self.img_x):
                #     for j in range(self.img_y): 
                #         p_c[n, c] *= self.proba[i, j, c]**self.imgs[i, j, n] *\
                #             (1 - self.proba[i, j, c])**(1 - self.imgs[i, j, n]) 
                # p_c[n, c] *= self.lbda[c] 
        return p_c

    def responsability(self, p_c): 
        w = p_c
        marginal = np.sum(w, axis=1)
        for c in range(self.n_classes): 
            w[:, c] /= marginal
        return np.nan_to_num(w)

    def update_params(self, w): 
        self.lbda = np.sum(w, axis=0) / self.n_imgs

        for c in range(self.n_classes): 
            self.proba[:, :, c] = w[0, c] * self.imgs[:, :, 0]
            for n in range(1, self.n_imgs): 
                self.proba[:, :,c] += w[n, c] * self.imgs[:, :, n]

        self.proba /= self.lbda * self.n_imgs

    def run_once(self): 
        p_c = self.class_proba(self.imgs)
        w = self.responsability(p_c)
        self.update_params(w)
        return np.copy(self.lbda), np.copy(self.proba)

    def __call__(self, imgs): 
        p_c = self.class_proba(imgs)
        s = np.sum(p_c, axis=1)
        for c in range(self.n_classes): 
            p_c[:,c] /= s
        return np.nan_to_num(p_c)


def confusion_matrix(predicted, expected, n_classes=None): 
    if n_classes is None: 
        n_classes = int(np.amax([np.amax(expected), np.amax(predicted)]) - np.amin([np.amin(expected), np.amin(predicted)])) + 1
    m = np.zeros((n_classes, n_classes))
    for i in range(predicted.shape[0]):
        m[int(predicted[i])][int(expected[i])] += 1.
    return m

def sort_cm(m): 
    new_m = np.zeros(m.shape)
    old_m = np.copy(m)
    association = -np.ones((m.shape[0], 2), dtype=int) 
    unas = np.ones((m.shape[0],2))
    for k in range(m.shape[0]):
        i, j = np.unravel_index(np.argmax(old_m), old_m.shape)
        if (association[:, 1] != j).all():
            old_m[i, :] = np.zeros((m.shape[0],))
            old_m[:, j] = np.zeros((1, m.shape[0]))
            association[i][0] = i
            association[i][1] = j
            unas[i, 0] = 0
            unas[j, 1] = 0
    for k in range(m.shape[0]): 
        if unas[k, 0] == 1: 
            association[k][0] = k
            j = int(np.argmax(unas[:, 1]))
            association[k][1] = j
            unas[k, 0] = 0
            unas[j, 1] = 0
        j = association[k, 1]
        new_m[j, :] = m[k, :]

    return new_m, association

def random_batch(imgs, lbls, size):
    index = np.random.randint(0, lbls.shape[0], size=size)
    return imgs[:,:,index], lbls[index]


if __name__ == "__main__": 
    PREFIX = ""
    TRAIN_IMGS = PREFIX + "train-images.idx3-ubyte"
    TRAIN_LBLS = PREFIX + "train-labels.idx1-ubyte"
    imgs, lbls = parse_mnist_dataset(TRAIN_IMGS, TRAIN_LBLS)
    TEST_IMGS = PREFIX + "t10k-images.idx3-ubyte"
    TEST_LBLS = PREFIX + "t10k-labels.idx1-ubyte"
    t_imgs, t_lbls = parse_mnist_dataset(TEST_IMGS, TEST_LBLS)

    imgs = binarize_imgs(imgs)
    t_imgs = binarize_imgs(t_imgs)

    em = EMAlgorithm(imgs[:,:,:20000], lbls[:20000]) 

    n_rows = 2
    n_cols = 5

    K = 1000
    lb_old = 0
    pr_old = 0
    for k in range(50): 
        lb, pr = em.run_once() 

        delta_lb = np.linalg.norm(lb - lb_old)
        delta_pr = np.linalg.norm(pr - pr_old)

        if delta_lb < 0.000001 and delta_pr < 0.0001: 
            break

        print("Update pi : %f"%(delta_lb))
        print("Update mu : %f"%(delta_pr))

        lb_old = lb
        pr_old = pr

        if (k+1) % 49 == 0: 
            plt.figure(1)   
            for i in range(0, 10): 
                plt.imshow(em.proba[:,:,i])



ti, tl = random_batch(imgs, lbls, 10000)

p = np.argmax(em(ti), axis=1) 
e = tl

m,a = sort_cm(confusion_matrix(p,e))

acc = 0.
for i in range(m.shape[0]): 
    acc += m[i, i]
acc /= 10000

print("Accuracy : %f"%(acc))
plt.close(1)
plt.close(2)
plt.figure(1)

m,a = sort_cm(confusion_matrix(p,e))
plt.imshow(m)
plt.colorbar()    

plt.figure(2)
for i in range(0, 10): 
    print(n_rows,n_cols,a[i][1] + 1)
    plt.subplot(n_rows,n_cols,a[i][1] + 1)
    plt.imshow(em.proba[:,:,i])

plt.show(True)