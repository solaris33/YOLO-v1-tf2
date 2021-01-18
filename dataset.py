import tensorflow as tf
import numpy as np

# Reference : https://stackoverflow.com/questions/54567986/python-numpy-remove-empty-zeroes-border-of-3d-array
def bounds_per_dimension(ndarray):
  return map(
    lambda e: range(e.min(), e.max() + 1),
    np.where(ndarray != 0)
  )


def zero_trim_ndarray(ndarray):
  return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


# process ground-truth data for YOLO format
def process_each_ground_truth(original_image,
                              bbox,
                              class_labels,
                              input_width,
                              input_height
                              ):
  """
  Reference:
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/object_detection/voc.py#L115
    bbox return : (ymin / height, xmin / width, ymax / height, xmax / width)
  Args:
    original_image : (original_height, orignal_width, channel) image tensor
    bbox : (max_object_num_in_batch, 4) = (ymin / height, xmin / width, ymax / height, xmax / width)
    class_labels : (max_object_num_in_batch) = class labels without one-hot-encoding
    input_width : yolo input width
    input_height : yolo input height
  Returns:
    image: (resized_height, resized_width, channel) image ndarray
    labels: 2-D list [object_num, 5] (xcenter (Absolute Coordinate), ycenter (Absolute Coordinate), w (Absolute Coordinate), h (Absolute Coordinate), class_num)
    object_num: total object number in image
  """
  image = original_image.numpy()
  image = zero_trim_ndarray(image)

  # set original width height
  original_h = image.shape[0]
  original_w = image.shape[1]

  width_rate = input_width * 1.0 / original_w
  height_rate = input_height * 1.0 / original_h

  image = tf.image.resize(image, [input_height, input_width])

  object_num = np.count_nonzero(bbox, axis=0)[0]
  labels = [[0, 0, 0, 0, 0]] * object_num
  for i in range(object_num):
    xmin = bbox[i][1] * original_w
    ymin = bbox[i][0] * original_h
    xmax = bbox[i][3] * original_w
    ymax = bbox[i][2] * original_h

    class_num = class_labels[i]

    xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
    ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

    box_w = (xmax - xmin) * width_rate
    box_h = (ymax - ymin) * height_rate

    labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

  return [image.numpy(), labels, object_num]