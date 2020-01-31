import numpy as np 
from matplotlib import pyplot as plt


def to_int(b32): 
	return b32[3] + b32[2] * 2**8 + b32[1] * 2**16 + b32[0]  * 2**24

def parse_ubyte_file(file): 
	with open(file, 'rb') as f: 
		magic_number = to_int(f.read(4))
		number_elts = to_int(f.read(4))
		if magic_number == 2051: 
			elts = f.read(number_elts * to_int(f.read(4)) * to_int(f.read(4)))
		elif magic_number == 2049: 
			elts = f.read(number_elts)
		f.close() 
		return magic_number, number_elts, elts

def parse_mnist_dataset(images_path, labels_path):
	im_magic, im_nbelts, imgs_elts = parse_ubyte_file(images_path) 
	lb_magic, lb_nbelts, lbls_elts = parse_ubyte_file(labels_path)
	
	if im_magic != 2051: 
		raise Exception("Invalid magic number for images file [" + images_path + "]") 
	if lb_magic != 2049: 
		raise Exception("Invalid magic number for labels file [" + labels_path + "]") 
	if im_nbelts != lb_nbelts: 
		raise Exception("Number of elements mismatch between [" + images_path + "] and [" + labels_path + "]")

	imgs = np.frombuffer(imgs_elts, dtype=np.uint8) 
	imgs = np.reshape(imgs, (28, 28, im_nbelts), order='F') 
	imgs = np.transpose(imgs, (1,0,2))
	imgs = np.asarray(imgs / 255., dtype=np.float) 

	lbls = np.frombuffer(lbls_elts, dtype=np.uint8)
	lbls = np.asarray(lbls, dtype=np.float)

	return imgs, lbls 

if __name__=="__main__": 
	TRAIN_IMGS =  "train-images.idx3-ubyte"
	TRAIN_LBLS =  "train-labels.idx1-ubyte"
	imgs, lbls = parse_mnist_dataset(TRAIN_IMGS, TRAIN_LBLS)

	plt.imshow(imgs[:, :,2000], interpolation='none')	
	plt.show() 



