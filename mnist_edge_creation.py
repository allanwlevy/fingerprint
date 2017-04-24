'''
General idea:
1. Create a grow_img that "grows" the black area of the mnist digit.
2. For each pixel in grow_img, look at how many surrounding pixels are black. If 
	enough black pixels are reached, make that one black too.
3. Use Canny edge detection from skimage to make the next ring. 
4. Paste all rings together.
There will be a final_img that gets rings pasted on for each iteration, and a 
grow_img that has the growing mnist digit. 
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
import random
from keras.datasets import mnist
import scipy
import math


(X_train, y_train), (X_test, y_test) = mnist.load_data()


#mnist image size
base_size = (28, 28)

#what factor we want the final image to bigger by
enlarge_factor = 5

#distance between stripes
jump_factor = 3

#size of final image we want
final_size = tuple([enlarge_factor * x for x in base_size])

#look at each pixel of img. Returns new image that has "grown"
def grow(img):
	new_img = np.copy(img)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			neighbors = img[row-1:row+2, col-1:col+2]
			if neighbors.shape == (3, 3):
				total = np.sum(neighbors)
				if total <= 2200:
					#print((row, col))
					new_img[row][col] = 0
	return new_img

#look at 1000 mnist images so far
for image_index in range(1000):
	img = X_train[image_index]

	#original mnist digit is black background with white writing. This flips it - 255 is white, 0 is black.
	img = 255 - img

	img_pillow = Image.fromarray(img)

	#save base image to compare later
	img_pillow.save('mnist_stripe_data_smaller/' + str(image_index) + ".png")


	#create final, blank white image which will have rings pasted on
	old_size = img_pillow.size
	final_image = Image.new("L", final_size, color=255) 
	final_image.paste(img_pillow, (int((final_size[0]-old_size[0])/2),
	                      int((final_size[1]-old_size[1])/2)))
	final_image = np.array(final_image)


	#create the grow_img
	grow_img = Image.new("L", final_size, color=255) 
	grow_img.paste(img_pillow, (int((final_size[0]-old_size[0])/2),
	                      int((final_size[1]-old_size[1])/2)))
	grow_img = np.array(grow_img)




	for enlarge_num in range(-1, -1, 1):

		#tried using scipy's convolution function to do the same thing. It was off by some small but really annoying factor so I just did it manually
		'''
		convolved = scipy.signal.convolve2d(grow_img, np.ones((3,3)), mode='full', fillvalue=255)
		#convolved = np.roll(convolved, 1, axis=1)
		#convolved = np.roll(convolved, 1, axis=0)
		print(convolved.shape)
		grow_img[convolved <= 2200] = 0
		'''


		
		grow_img = grow(grow_img)
		img_pillow = Image.fromarray(grow_img)
		img_pillow.save('test' + str(enlarge_num) + 'enlarge.png')

		#don't generate ring for every iteration or else rings would be too close together
		if enlarge_num % 3 == 0:
			newEdges = 255 - feature.canny(grow_img).astype(int) * 255
			#print((newEdges != 255).sum())

			final_image = np.where(newEdges != 255, newEdges, final_image)

			#overlayPillow = Image.fromarray(np.uint8(final_image))
			#overlayPillow.save('mnist_stripe_data_smaller/' + str(image_index) + str(enlarge_num) + "overlayed.png")





		'''
		#resizing base edges image by resize_factor
		new_shape = tuple([resize_factor * x for x in base_size])
		edgesToAdd = edgesPillow.resize(new_shape)
		#resize edges to final size and surround with 0's/black until final_size is reached 
		old_size = edgesToAdd.size
		new_im = Image.new("L", final_size) 
		new_im.paste(edgesToAdd, (int((final_size[0]-old_size[0])/2),
		                      int((final_size[1]-old_size[1])/2)))
		#need numpy version to overlay images together
		npEdgesToAdd = np.array(new_im)
		#the larger the enlarge_factor, the more noise there is. Use Gaussian filter to smooth it out
		if (enlarge_factor > 6):
			npEdgesToAdd = ndi.filters.gaussian_filter(npEdgesToAdd, int(0.5*enlarge_factor), mode='nearest')
		#one more edge-detection layer on the now-enlarged edge
		newEdges = feature.canny(npEdgesToAdd).astype(int) * 255
		
		#overlaying images together
		final_image = np.where(newEdges != 0, newEdges, final_image)
		'''

	
	
	#save final image
	overlayPillow = Image.fromarray(np.uint8(final_image))
	overlayPillow.save('mnist_stripe_data_smaller/' + str(image_index) + "overlayed.png")
	
	if image_index % 50 == 0:
		print(str(image_index) + ' images done')

