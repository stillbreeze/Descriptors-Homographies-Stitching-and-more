import numpy as np
import matplotlib.pyplot as plt
from planarH import computeH

def compute_extrinsics(K, H):
	H_prime = np.linalg.inv(K).dot(H)
	U,S,V = np.linalg.svd(H_prime[:,:2])
	S_new = np.array([[1,0],[0,1],[0,0]])
	R = U.dot(S_new).dot(V)
	r3 = np.cross(R[:,0],R[:,1])
	R = np.concatenate((R, np.expand_dims(r3,-1)), axis=-1)
	if np.rint(np.linalg.det(R)) == -1:
		R[:,-1] = -1 * R[:,-1]
	Lambda = np.divide(H_prime[:,:2], R[:, :2]).mean()
	t = H_prime[:,-1] / Lambda
	return R, t


def project_extrinsics(K, W, R, t):

	image_name = '../data/prince_book.jpeg'
	R_t = np.concatenate((R, np.expand_dims(t,-1)), axis=-1)
	transformed = K.dot(R_t).dot(W)
	transformed = transformed / transformed[2]
	
	im = plt.imread(image_name)
	implot = plt.imshow(im)
	plt.scatter(x=transformed[0,:], y=transformed[1,:], c='y', s=1)
	plt.show()
	plt.close()


if __name__ == '__main__':
	K = np.array([[3043.72,0.0,1196.00],[0.0,3043.72,1604.00],[0.0,0.0,1]])
	W = np.array([[0.0, 18.2, 18.2, 0.0],[0.0, 0.0, 26.0, 26.0]])
	X = np.array([[483, 1704, 2175, 67],[810, 781, 2217, 2286]])
	H = computeH(X, W)
	R, t = compute_extrinsics(K, H)
	H_prime = np.linalg.inv(K).dot(H)

	with open('../data/sphere.txt', 'r') as f:
		pts = f.readlines()
	W_new = []
	for i in pts:
		i = i.strip()
		results = list(map(float, i.split('  ')))
		results = np.asarray(results)
		W_new.append(results)
	W_new.append(np.ones(results.shape[0]))
	W_new = np.asarray(W_new)
	center = np.array([810,1430,1])
	shift = np.linalg.inv(H).dot(center)
	shift = shift / shift[-1]
	W_new[:3,:] = (W_new[:3,:].T + shift).T
	X = project_extrinsics(K, W_new, R, t)
