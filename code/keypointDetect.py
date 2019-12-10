import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    DoG_pyramid = np.diff(gaussian_pyramid)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    # r = 10
    # threshold_r = (r + 1)**2 * 1.0 / r
    principal_curvature = None
    D_x = cv2.Sobel(DoG_pyramid, ddepth=cv2.CV_64F, dx=1, dy=0, borderType=2)
    D_y = cv2.Sobel(DoG_pyramid, ddepth=cv2.CV_64F, dx=0, dy=1, borderType=2)
    D_xx = cv2.Sobel(D_x, ddepth=cv2.CV_64F, dx=1, dy=0, borderType=2)
    D_xy = cv2.Sobel(D_x, ddepth=cv2.CV_64F, dx=0, dy=1, borderType=2)
    D_yy = cv2.Sobel(D_y, ddepth=cv2.CV_64F, dx=0, dy=1, borderType=2)
    D_yx = cv2.Sobel(D_y, ddepth=cv2.CV_64F, dx=1, dy=0, borderType=2)
    trace_H = D_xx + D_yy
    det_H = (D_xx * D_yy) - (D_xy * D_yx)
    # print (det_H)
    principal_curvature = (trace_H * trace_H * 1.0) / det_H
    principal_curvature[principal_curvature == -np.inf] = 0
    principal_curvature[principal_curvature == np.inf] = 0
    principal_curvature[principal_curvature == np.nan] = 0
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    threshold_mask = np.logical_and(DoG_pyramid > th_contrast, np.abs(principal_curvature) < th_r)

    neighbors_spatial = []
    neighbors_spatial.append(np.roll(DoG_pyramid, 1, axis=0))
    neighbors_spatial.append(np.roll(DoG_pyramid, -1, axis=0))
    neighbors_spatial.append(np.roll(DoG_pyramid, 1, axis=1))
    neighbors_spatial.append(np.roll(DoG_pyramid, -1, axis=1))
    neighbors_spatial.append(np.roll(np.roll(DoG_pyramid, 1, axis=0), 1, axis=1))
    neighbors_spatial.append(np.roll(np.roll(DoG_pyramid, 1, axis=0), -1, axis=1))
    neighbors_spatial.append(np.roll(np.roll(DoG_pyramid, -1, axis=0), 1, axis=1))
    neighbors_spatial.append(np.roll(np.roll(DoG_pyramid, -1, axis=0), -1, axis=1))

    neighbors_scale = []
    neighbors_scale.append(np.roll(DoG_pyramid, 1, axis=2))
    neighbors_scale.append(np.roll(DoG_pyramid, -1, axis=2))

    neighbors_spatial = np.asarray(neighbors_spatial)
    neighbors_scale = np.asarray(neighbors_scale)

    max_neighbor = np.max(neighbors_spatial, axis=0)
    min_neighbor = np.min(neighbors_spatial, axis=0)
    spatial_mask = np.logical_or((DoG_pyramid >= max_neighbor), (DoG_pyramid <= min_neighbor))

    max_neighbor = np.max(neighbors_scale, axis=0)
    min_neighbor = np.min(neighbors_scale, axis=0)
    scale_mask = np.logical_or((DoG_pyramid >= max_neighbor), (DoG_pyramid <= min_neighbor))

    final_mask = np.logical_and(np.logical_and(spatial_mask, scale_mask), threshold_mask)
    locsDoG = np.where(final_mask==True)
    locsDoG = np.asarray(locsDoG)
    locsDoG = locsDoG.T
    locsDoG[:,[0, 1]] = locsDoG[:,[1, 0]]
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # visualize_edge_suppression(im, locsDoG)
    return locsDoG, gauss_pyramid


def visualize_edge_suppression(im, locsDoG):
    implot = plt.imshow(im)
    plt.plot(locsDoG[:,0], locsDoG[:,1],'rx', markersize=4)
    plt.savefig('suppressed_edges.png')
    plt.close()


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)


