import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    a,b,f = im1.shape
    z = np.zeros((a,b,f))
    im1 = np.concatenate((im1,z),axis=1)

    r,c,k = im1.shape
    warped_img2 = cv2.warpPerspective(im2, H2to1, (2*b,a))
    cv2.imwrite('../results/6_1.jpg', warped_img2)
    warped_img2 = cv2.warpPerspective(im2, H2to1, (c,r))
    pano_im = np.maximum(im1, warped_img2)
    # cv2.imwrite('../results/stitch.jpg', pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    full_cols = 1800
    r, c, _ = im2.shape
    boundary = np.array([[1,c,1,c],[1,1,r,r],[1,1,1,1]])
    warped_boundary = np.matmul(H2to1, boundary)
    warped_boundary = warped_boundary / warped_boundary[-1,:]
    warped_boundary = np.rint(warped_boundary).astype('int')
    min_c = min(1, warped_boundary[0].min())
    max_c = max(im1.shape[1], warped_boundary[0].max())
    min_r = min(1, warped_boundary[1].min())
    max_r = max(im1.shape[0], warped_boundary[1].max())
    full_rows = (full_cols * (max_r - min_r)) / (max_c - min_c)
    full_rows = np.rint(full_rows).astype('int')
    scaling_factor = full_cols / (max_c - min_c)
    M_translation = np.array([[1,0,0],[0,1,-min_r],[0,0,1]])
    M_scaling = np.array([[scaling_factor, 0, 0],[0, scaling_factor, 0],[0, 0, 1]])
    M = np.matmul(M_scaling, M_translation)
    warped_img1 = cv2.warpPerspective(im1, M, (full_cols,full_rows))
    warped_img2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (full_cols,full_rows))
    pano_im = np.maximum(warped_img1, warped_img2)
    # cv2.imwrite('stitch_no_clip.jpg', pano_im)
    cv2.imwrite('../results/6_2_pan.jpg', pano_im)
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/6_3.jpg', pano_im)
    pn = imageStitching(im1, im2, H2to1)
    return pn


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    pn = generatePanorama(im1, im2)
