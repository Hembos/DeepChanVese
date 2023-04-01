from numba.pycc import CC

import numpy as np

cc = CC('mask_labeling_module')

@cc.export('run_labeling', 'i8[:](b1[:], i8, i8, i8)')
def run_labeling(mask, width, height, offset):
    equivalent = dict()
    labeledMask = np.zeros(width * height, np.int64)
    labelNum = offset

    for i in range(height):
        for j in range(width):
            if mask[i * width + j]:
                neighbors = [
                    labeledMask[(i - 1) * width + (j - 1)] if i - 1 > 0 and j - 1 > 0 else 0, 
                    labeledMask[(i - 1) * width + (j)] if i - 1 > 0 else 0, 
                    labeledMask[(i - 1) * width + (j + 1)] if i - 1 > 0 and j + 1 < width else 0, 
                    labeledMask[(i) * width + (j - 1)] if j - 1 > 0 else 0
                ]

                minlabel = 0
                for k in range(4):
                    if minlabel == 0 or neighbors[k] != 0 and neighbors[k] < minlabel:
                        minlabel = neighbors[k]
                        continue

                for k in range(4):
                    if neighbors[k] != 0:
                        equivalent[neighbors[k]] = minlabel

                if minlabel != 0:
                    labeledMask[i * width + j] = minlabel
                else:
                    labelNum += 1
                    equivalent[labelNum] = labelNum
                    labeledMask[i * width + j] = labelNum

    stop = False
    while not stop:
        stop = True
        for i in range(height):
            for j in range(width):
                label = labeledMask[i * width + j]
                if label != 0 and label != equivalent[label]:
                    labeledMask[i * width + j] = equivalent[label]
                    stop = False

    return labeledMask


if __name__=="__main__":
    cc.compile()