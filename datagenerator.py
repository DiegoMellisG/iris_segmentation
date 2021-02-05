import os
import cv2
import json
import glob
import skimage.draw
import tensorflow
import numpy as np
from PIL import Image
from utils import load_image

class Dataset(tensorflow.keras.utils.Sequence):
	"""The base class for dataset classes.
	To use it, create a new class that adds functions specific to the dataset
	you want to use. For example:

	class EyesDataset(Dataset):
		def load_eyes(self):
			...
		def load_mask(self, image_id):
			...
		def image_reference(self, image_id):
			...
	"""

	def __init__(
		self, shuffle=True, dim=(480, 640), augmentation=None,
		channels=3, batch_size=1, class_map=None, preprocess=[]
	):
		self.dim = dim
		self._image_ids = []
		self.image_info = []
		self.shuffle = shuffle
		self.channels = channels
		self.batch_size = batch_size
		self.augmentation = augmentation
		self.input_shape = (*dim, channels)
		self.class_info = []
		self.preprocess = preprocess
		self.source_class_ids = {}

	def add_class(self, source, class_id, class_name):
		assert "." not in source, "Source name cannot contain a dot"
		# Does the class exist already?
		for info in self.class_info:
			if info['source'] == source and info["id"] == class_id:
				# source.class_id combination already available, skip
				return
		# Add the class
		self.class_info.append({
			"source": source,
			"id": class_id,
			"name": class_name,
		})

	def add_image(self, source, image_id, path, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def prepare(self, class_map=None):
		"""Prepares the Dataset class for use.

		TODO: class map is not supported yet. When done, it should handle mapping
			  classes from different datasets to the same class ID.
		"""

		def clean_name(name):
			"""Returns a shorter version of object names for cleaner display."""
			return ",".join(name.split(",")[:1])

		# Build (or rebuild) everything else from the info dicts.
		self.num_classes = len(self.class_info)
		self.class_ids = np.arange(self.num_classes)
		self.class_names = [clean_name(c["name"]) for c in self.class_info]
		self.num_images = len(self.image_info)

        # image_ids = [1, 2, 3, 4, ...]
		self._image_ids = np.arange(self.num_images)

	def get_source_class_id(self, class_id, source):
		"""Map an internal class ID to the corresponding class ID in the source dataset."""
		info = self.class_info[class_id]
		assert info['source'] == source
		return info['id']

	@property
	def image_ids(self):
		return self._image_ids

	def source_image_link(self, image_id):
		"""Returns the path or URL to the image.
		Override this to return a URL to the image if it's available online for easy
		debugging.
		"""
		return self.image_info[image_id]["path"]

	def load_image(self, image_ids):
		"""Load the specified image and return a [BS,H,W,3] Numpy array.
		"""
		bs = np.zeros(
			[self.batch_size, *self.dim, self.channels], np.uint8
		)

		for i, idx in enumerate(image_ids):

			# Load image
			# image = skimage.io.imread(self.image_info[idx]['path'])
			image = load_image(self.image_info[idx]['path'])
			image = cv2.resize(image, self.dim[::-1])
			# If grayscale. Convert to RGB for consistency.
			if image.ndim != 3:
				image = skimage.color.gray2rgb(image)
			# If has an alpha channel, remove it for consistency
			if image.shape[-1] == 4:
				image = image[..., :3]
			
			# asign image to batch
			bs[i, ] = image
		
		return bs

	def load_mask(self, image_id):
		"""Load instance masks for the given image.

		Different datasets use different ways to store masks. Override this
		method to load instance masks and return them in the form of am
		array of binary masks of shape [height, width, instances].

		Returns:
			masks: A bool array of shape [height, width, instance count] with
				a binary mask per instance.
			class_ids: a 1D array of class IDs of the instance masks.
		"""
		# Override this function to load a mask from your dataset.
		# Otherwise, it returns an empty mask.
		logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
		mask = np.empty([0, 0, 0])
		class_ids = np.empty([0], np.int32)
		return mask, class_ids
	
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			np.random.shuffle(self._image_ids)
	
	def __len__(self):
		raise NotImplementedError('abstract method \'__len__\' not implemented')

	def __getitem__(self, index):
		raise NotImplementedError('abstract method \'__getitem__\' not implemented')

class EyeDataset(Dataset):

	def load_eyes(self, dataset_dir, subset):
		"""Load a subset of the Eye dataset.
		dataset_dir: Root directory of the dataset.
		subset: Subset to load: train or val
		"""
		# Add classes
		self.add_class("eye", 0, "eye")
		self.add_class("eye", 1, "iris")
		self.add_class("eye", 2, "pupil")
		self.add_class("eye", 3, "sclera")
		# self.add_class("eye", 4, "eye")
		
		# Train, test or validation dataset?
		assert subset in ["train", "test", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)

		"""
		# Load annotations
		# regions = JSON file
		# regions:
		# {
		# 	image_id: 
		# 		{ 'filename': filename,
        #         'size': size,
        #         'regions':
        #           [
		# 			    {'shape_attributes': {
		# 				    'name': 'polygon', 
		# 				    'all_points_x': [...], 
		# 				    'all_points_y': [...]
		# 			        },
        #                 'region_attributes': 
        #                   {'Eye': 'iris'}
        #               }
        #           ]
		# 			
		# 		 }
		# 		...
		# 	]
		# }
		"""
		
		annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))

		# Add images
		for key in annotations:
			# key = image_id from VIA
			# Get the x, y coordinaets of points of the polygons that make up
			# the outline of each object instance. These are stores in the
			# shape_attributes (see json format above)
			polygons = [ r['shape_attributes'] for r in annotations[key]['regions']]
			objects = [ s['region_attributes'] for s in annotations[key]['regions']]
			
			# num_ids = [1, 2, 3] => ['iris', 'pupil', 'sclera', ]
			num_ids = []
			
			for obj in objects:
				for cl_info in self.class_info:
					if cl_info['name'] == obj['Eye']:
						num_ids.append(cl_info['id'])
			#print(num_ids)
			# load_mask() needs the image size to convert polygons to masks.
			# Unfortunately, VIA doesn't include it in JSON, so we must read
			# the image. This is only managable since the dataset is tiny.
			image_path = os.path.join(dataset_dir, annotations[key]['filename'])
			# height, width = skimage.io.imread(image_path).shape[:2]
			# Pillow use less memory
			width, height = Image.open(image_path).size

			self.add_image(
				"eye",
				image_id=annotations[key]['filename'],  # use file name as a unique image id
				path=image_path,
				width=width, height=height,
				polygons=polygons,
				num_ids=num_ids
			)

	def load_mask(self, image_ids):
		"""Generate instance masks for an image.
		Returns:
		masks: A bool array of shape [bs, height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		
		bs_mask = np.zeros(
			[self.batch_size, *self.dim, self.num_classes], 
			dtype=np.float32
		)
        
		for idx, imid in enumerate(image_ids):
			# If not an eye dataset image, delegate to parent class.
			image_info = self.image_info[imid]
			if image_info["source"] != "eye":
				"""  """
				return super(self.__class__, self).load_mask(imid)
			num_ids = image_info['num_ids']

			# Convert polygons to a bitmap mask of shape
			# [height, width, instance_count]
			info = self.image_info[imid]
			mask = np.zeros(
				[image_info["height"], image_info["width"], self.num_classes],
				dtype=np.bool
			)

			for i, p in zip(num_ids, image_info["polygons"]):
				#print("hola ",i)
				# Get indexes of pixels inside the polygon and set them to 1
				try:
					if p['name'] in ['polygon', 'polyline', ]:
						rr, cc = skimage.draw.polygon(
							p['all_points_y'], p['all_points_x']
						)
					elif p['name'] in ['ellipse', ]:
						rr, cc = skimage.draw.ellipse(
							p['cy'], p['cx'],
							p['ry'], p['rx']
						)
					elif p['name'] in ['circle', ]:
						rr, cc = skimage.draw.circle(
							p['cy'], p['cx'], p['r']
						)
					#print(len(rr))
					mask[rr, cc, i] = True
						
				except (KeyError, IndexError) as ex:
					print(image_info, i, p, ex)
			
			#fix to sclera with iris
			id_sclera = next(d['id'] for d in self.class_info if 'sclera' in d['name'])
			# fix to iris with pupil
			id_pupil = next(d['id'] for d in self.class_info if 'pupil' in d['name'])
			#print("id_pupil",id_pupil)
			id_iris = next(d['id'] for d in self.class_info if 'iris' in d['name'])
			#print("id_iris",id_iris)
			id_eye = next(d['id'] for d in self.class_info if 'eye' in d['name'])
			#print("id_eye",id_eye)
			pupil = mask[..., id_pupil].copy().astype(np.bool)
			iris = mask[..., id_iris].copy().astype(np.bool)
			sclera = mask[..., id_sclera].copy().astype(np.bool)
			sclera_xor = np.logical_xor(sclera,iris)
			sclera_xor = np.where(sclera_xor == True, 1., 0.)
			iris_xor = np.logical_xor(iris, pupil)
			iris_xor = np.where(iris_xor == True, 1., 0.)
			mask[..., id_iris] = iris_xor.copy()
			mask[..., id_sclera] = sclera_xor.copy()

			one_mask = np.argmax(mask, axis=-1)
			#print(one_mask.shape)
			mask[..., id_eye] = np.where(one_mask == id_eye, 1., 0.)
			mask = cv2.resize(mask.astype(np.uint8), self.dim[::-1])
			mask = mask.astype(np.bool)
			#print(mask.shape)

			bs_mask[idx, ] = mask.astype(np.float32)

		return bs_mask

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "eye":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)

	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.num_images // self.batch_size

	def dataAugmentation(self, image, masks):
		import imgaug
		# This requires the imgaug lib (https://github.com/aleju/imgaug)
		# Augmenters that are safe to apply to masks
		# Some, such as Affine, have settings that make them unsafe, so always
		# test your augmentation on masks
		MASK_AUGMENTERS = [
			"Sequential", "SomeOf", "OneOf", "Sometimes",
			"Fliplr", "Flipud", "CropAndPad",
			"Affine", "PiecewiseAffine", "CoarseDropout",
			"KeepSizeByResize", "CropToFixedSize",
			"TranslateX", "TranslateY", "Pad", "Lambda"
		]

		def hook(images, augmenter, parents, default):
			"""Determines which augmenters to apply to masks."""
			return augmenter.__class__.__name__ in MASK_AUGMENTERS

		for bs in range(self.batch_size):
			# Store shapes before augmentation to compare
			image_shape = image[bs, ].shape
			mask_shape = masks[bs, ].shape
			# Make augmenters deterministic to apply similarly to images and masks
			det = self.augmentation.to_deterministic()
			image[bs, ] = det.augment_image(image[bs, ])
		
			# in one shot
			masks[bs, ...] = det.augment_image(
				masks[bs, ...].astype(np.uint8),
				hooks=imgaug.HooksImages(activator=hook)
			).astype(np.float32)
			
			# Verify that shapes didn't change
			assert image[bs, ].shape == image_shape, "Augmentation shouldn't change image size"
			assert masks[bs, ].shape == mask_shape, "Augmentation shouldn't change mask size"

		return image, masks

	def __getitem__(self, index):
		if index > self._image_ids.max():
			raise IndexError(
				f'List index out of range. Size of generator: {self.__len__()}'
			)
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self._image_ids[index*self.batch_size:(index+1)*self.batch_size]
		images = self.load_image(indexes)
		masks = self.load_mask(indexes)
		
		# data augmentation
		if self.augmentation:
			images, masks = self.dataAugmentation(images, masks)
		
		# image normalization
		images = images / 255.0
		return images, masks
	
	def get_image_by_name(self, imname):
		assert self.batch_size == 1, "Batch size must be 1."
		for i in range(len(self._image_ids)):
			if imname in self.image_reference(i):
				return self.__getitem__(i)
