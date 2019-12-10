import numpy as np
import cv2
import matplotlib.pyplot as plt

import BRIEF as br

if __name__ == '__main__':
    test_pattern_file = '../results/testPattern.npy'
    compareX, compareY = np.load(test_pattern_file)
    compareX, compareY = br.makeTestPattern()
    match_count = []
    for i in range(0,360,10):
        im1 = cv2.imread('../data/model_chickenbroth.jpg')
        im2 = cv2.imread('../data/model_chickenbroth.jpg')
        rows, cols, _ = im2.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-i,1)
        im2 = cv2.warpAffine(im2, M, (cols,rows))
        locs1, desc1 = br.briefLite(im1)
        locs2, desc2 = br.briefLite(im2)
        matches = br.briefMatch(desc1, desc2)
        match_count.append(matches.shape[0])
    match_count = np.asarray(match_count)
    plt.bar(np.arange(0,360,10), match_count)
    plt.xlabel('Degree of rotation', fontsize=15)
    plt.ylabel('No of matches', fontsize=15)
    plt.show()
    plt.close()
