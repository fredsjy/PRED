import tensorflow as tf
import numpy as np
from skimage import io

# def get_image(image_path):
#     """Reads the jpg image from image_path.
#     Returns the image as a tf.float32 tensor
#     Args:
#         image_path: tf.string tensor
#     Reuturn:
#         the decoded jpeg image casted to float32
#     """
#     return tf.image.convert_image_dtype(
#         tf.image.decode_jpeg(
#             tf.read_file(image_path), channels=3), dtype=tf.uint8)
# img = get_image("ps_images/1.jpg")
# img1 = tf.image.decode_jpeg(tf.read_file("ps_images/2.jpg"), channels = 3)

img = io.imread("ps_images/1.jpg")
print(img)
print("--------")
print(img[0].T.ravel())
# print(np.reshape(img, (1, img.shape[0]*img.shape[1]*img.shape[2])))


