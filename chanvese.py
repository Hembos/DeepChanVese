import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from PIL import Image
from tqdm import tqdm
import torch

def mask_to_phi(mask):
    phi = np.zeros(mask.shape)

    for row, elems in enumerate(mask):
        for col, elem in enumerate(elems):
            phi[row][col] = -1 if elem == 0 else 1

    return phi

def phi_to_mask(phi):
    return phi >= 0.0
    
def sussman_sign(D):
    return D / np.sqrt(D**2 + 1)

class ChanVese():
    def __init__(self, num_iter=20) -> None:
        self.v = 0
        self.eta = 1e-8

        self.num_iter = num_iter

    def run(self, input, initial_phi, dt = 0.5, eps = 1, lam1 = 1, lam2 = 1, mu = 0.2):
        phi = initial_phi
        for i in tqdm(range(self.num_iter), total=self.num_iter):
            print(f"iter number: {i + 1}")

            #c1 - average intensity inside object, c2 - average intensity outside object
            c1, c2 = self.__calc_aver_intensity__(input, phi, eps[i])
            # print(f"average intensities: {c1}, {c2}")

            phi = self.__update_phi__(input, phi, c1, c2, dt[i], eps[i], lam1, lam2, mu)

        return phi

    def __heaviside__(self, x, eps):
        return 0.5 * (1 + 2 / np.pi * np.arctan2(x / eps))

    def __dirac__(self, x, eps):
        return 1 / np.pi * eps / (np.power(eps, 2) + np.power(x, 2))
    
    def __calc_aver_intensity__(self, input, phi, eps):
        c1_ind = np.flatnonzero(phi >= 0)
        c2_ind = np.flatnonzero(phi < 0)

        c1 = np.sum(input.flat[c1_ind]) / (len(c1_ind) + eps)
        c2 = np.sum(input.flat[c2_ind]) / (len(c2_ind) + eps)

        return (c1, c2)
    
    def __update_phi__(self, input, phi, c1, c2, dt, eps, lam1, lam2, mu):
        idx = np.nonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) < 0:
            return

        size = len(phi)
        calc_a = lambda i, j: mu / np.sqrt(np.power(self.eta, 2) + 
                                                np.power(phi[i + 1 if i<size-1 else i][j] - phi[i][j], 2) + 
                                                np.power((phi[i][j + 1 if j<size-1 else j] - phi[i][j - 1 if j else 0]) / 2, 2))
        
        calc_b = lambda i, j: mu / np.sqrt(np.power(self.eta, 2) + 
                                                np.power((phi[i + 1 if i<size-1 else i][j] - phi[i - 1 if i else 0][j]) / 2, 2) +
                                                np.power(phi[i][j] - phi[i + 1 if i<size-1 else i][j], 2))
        
        for i, elems in enumerate(phi):
            for j, elem in enumerate(elems):
                a = calc_a(i, j)
                b = calc_b(i, j)
                up_a = calc_a(i - 1, j) if i else 0
                left_b = calc_b(i, j - 1) if j else 0

                dirac_val = self.__dirac__(elem, eps)

                intensity_vec = a[:,i:i+1,j:j+1].flatten()

                c1_norm = torch.norm(intensity_vec - c1)
                c2_norm = torch.norm(intensity_vec - c2)
                
                tmp = phi[i][j] + dt * dirac_val * (a * phi[i + 1 if i<size-1 else i][j] +
                                up_a * phi[i - 1 if i else 0][j] + 
                                b * phi[i][j + 1 if j<size-1 else j] + 
                                left_b * phi[i][j - 1 if j else 0] - self.v - 
                                lam1 * np.power(c1_norm, 2) + lam2 * np.power(c2_norm, 2))

                phi[i][j] =  tmp / (1 + dt * dirac_val * (a + b + up_a + left_b))

        return phi

if __name__=="__main__":
    with Image.open("brain.png") as image:
        img = np.array(image)

    plt.imshow(img, cmap='gray')
    # img = plt.imread("brain.png")
    img_with_mask = np.zeros(img.shape)
    img = img[:,:,2]
    # print(img.shape)
    mask = np.zeros(img.shape)
    mask[20:100, 20:100] = 1

    cv = ChanVese(img, mask)
    final_mask = cv.run()

    for i, elems in enumerate(img):
        for j, elem in enumerate(elems):
            img_with_mask[i][j] = (elem, elem, elem) if not final_mask[i][j] else (255, 0, 0)

    # imgplot = plt.imshow(img_with_mask)
    plt.imshow(img_with_mask / 255)
    plt.show()