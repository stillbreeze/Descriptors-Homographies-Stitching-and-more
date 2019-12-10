import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    N = p1.shape[1]
    A = np.zeros((2*N,9))

    x, y = p1[0], p1[1]
    u, v = p2[0], p2[1]

    A[::2,0] = 0
    A[::2,1] = 0
    A[::2,2] = 0

    A[1::2,0] = u
    A[1::2,1] = v
    A[1::2,2] = 1

    A[::2,3] = -u
    A[::2,4] = -v
    A[::2,5] = -1

    A[1::2,3] = 0
    A[1::2,4] = 0
    A[1::2,5] = 0

    A[::2,6] = u * y
    A[::2,7] = v * y
    A[::2,8] = y

    A[1::2,6] = -(x * u)
    A[1::2,7] = -(x * v)
    A[1::2,8] = -x

    U, S, V = np.linalg.svd(A)
    H2to1 = V[-1].reshape(3,3)
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    point_count = matches.shape[0]
    ones = np.ones((point_count)).reshape(1,point_count)
    p1_homo = np.concatenate((np.transpose(locs1[matches[:,0]][:,:2]), ones), axis=0)
    ones = np.ones((point_count)).reshape(1,point_count)
    p2_homo = np.concatenate((np.transpose(locs2[matches[:,1]][:,:2]), ones), axis=0)

    max_inliers = 0
    for i in range(num_iter):
        rand_matches = np.random.permutation(matches)[:4]
        p1 = locs1[rand_matches[:,0]][:,:2].T
        p2 = locs2[rand_matches[:,1]][:,:2].T
        H = computeH(p1, p2)
        p1_transformed = np.matmul(H, p2_homo)
        p1_transformed = p1_transformed / p1_transformed[-1,:]
        error = (np.square((p1_transformed - p1_homo)).sum(axis=0) < tol).astype('int')
        inliers = error.sum()
        if inliers > max_inliers:
            max_inliers = inliers
            best_error = error

    best_inliers = matches[np.where(best_error==True)] 
    p1 = locs1[best_inliers[:,0]][:,:2].T
    p2 = locs2[best_inliers[:,1]][:,:2].T
    bestH = computeH(p1, p2)
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

