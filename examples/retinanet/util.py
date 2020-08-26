from jax import numpy as jnp

import jax
from multiprocessing import Pool
import numpy as np
import tensorflow as tf

CATEGORY_MAP = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush"
}


def pi_init(pi):
  """Wrapper to log-based weight initializer function.

  This initializer is used for the bias term in the classification subnet, as
  described in https://arxiv.org/abs/1708.02002

  Args:
    pi: the prior probability of detecting an object

  Returns:
    An array for initializing a module's weights / biases
  """

  def _inner(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * (-jnp.log((1 - pi) / pi))

  return _inner


@jax.vmap
def clip_anchors(anchors, height, width):
  """Clips anchors to height and width of image.

  More specifically, the x coordinates of the base anchors are clipped,
  such that they are always found in the `[0, width]` interval, and
  the `y` coordinates are always found in the `[0, height]` interval.

  Args:
    anchors: a tensor of the shape (|A|, 4) where each row contains 
      the `[x1, y1, x2, y1]` of that anchor
    height: the height of the image
    width: the width of the image

  Returns:
    A matrix of the form (|A|, 4), which contains the clipped anchors, as well
    as an extra column which can be used to store the status of the anchor.
  """
  x1 = jnp.clip(anchors[:, 0], 0.0, width)
  y1 = jnp.clip(anchors[:, 1], 0.0, height)
  x2 = jnp.clip(anchors[:, 2], 0.0, width)
  y2 = jnp.clip(anchors[:, 3], 0.0, height)
  return jnp.stack([x1, y1, x2, y2], axis=1)


def non_max_suppression(bboxes, scores, t):
  """Implements the Non-Maximum Suppression algorithm.

  More specifically, this algorithm retains the bboxes based on their scores 
  (those that have a higher score are favored), and IoU's with the other bboxes
  (bboxes that have a high overlap with bboxes with higher scores are removed).

  Args:
    bboxes: a matrix of the form (|B|, 4), where |B| is the number of bboxes,
      and the columns represent the coordinates of each bbox: [x1, y1, x2, y2]
    scores: a vector of the form (|B|,) storing the confidence in each bbox
    t: the IoU threshold; overlap above this threshold with higher scoring 
      bboxes will imply the lower scoring bbox should be discarded

  Returns:
    The indexes of the bboxes which are retained after NMS is applied, as well
    as their indexes in the original matrix. Both are ordered in decreasing 
    order.
  """
  selected_idx = []

  # Split the bboxes so they're easier to manipulate throughout
  x1 = bboxes[:, 0]
  y1 = bboxes[:, 1]
  x2 = bboxes[:, 2]
  y2 = bboxes[:, 3]

  sorted_idx = np.argsort(scores)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)

  while sorted_idx.shape[0] > 0:
    # Select the index of the bbox with the highest score
    current = sorted_idx[-1]
    selected_idx.append(current)

    # Determine the height and width of the intersections with the current bbox
    xx1 = np.maximum(x1[current], x1[sorted_idx[:-1]])
    yy1 = np.maximum(y1[current], y1[sorted_idx[:-1]])
    xx2 = np.minimum(x2[current], x2[sorted_idx[:-1]])
    yy2 = np.minimum(y2[current], y2[sorted_idx[:-1]])

    width = np.maximum(0.0, xx2 - xx1 + 1)
    height = np.maximum(0.0, yy2 - yy1 + 1)

    # Compute the IoU between the current bbox and all the other bboxes
    intersection = width * height
    ious = intersection / (
        areas[current] + areas[sorted_idx[:-1]] - intersection)

    # Keep only the bboxes with the lower threshold
    sorted_idx = sorted_idx[np.where(ious < t)[0]]

  # Return the indexes of the non-suppressed bboxes
  selected_idx = np.array(selected_idx, dtype=np.int32)
  return bboxes[selected_idx], selected_idx


def vertical_pad(data, pad_count, dtype=jnp.float32):
  """Applies vertical padding to the data by adding extra rows with 0.
  
  Args:
    data: the data to be padded 
    pad_count: the number of extra rows of padding to be added to the data

  Returns:
    `data` with extra padding
  """
  pad_shape = (pad_count,) + data.shape[1:]
  pad_structure = jnp.zeros(pad_shape, dtype=dtype)
  return jnp.append(data, pad_structure, axis=0)


def vertical_pad_np(data, pad_count, dtype=float):
  """Applies vertical padding to the data by adding extra rows with 0.
  
  Args:
    data: the data to be padded 
    pad_count: the number of extra rows of padding to be added to the data

  Returns:
    `data` with extra padding
  """
  pad_shape = (pad_count,) + data.shape[1:]
  pad_structure = np.zeros(pad_shape, dtype=dtype)
  return np.append(data, pad_structure, axis=0)


def top_k(scores, k, t=0.0):
  """Applies top k selection on the `scores` parameter.

  Args:
    scores: a vector of arbitrary length, containing non-negative scores, from 
      which only the at most top `k` highest scoring entries are selected.
    k: the maximal number of elements to be selected from `scores`
    t: a thresholding parameter (inclusive) which is applied on `scores`; 
      elements failing to meet the threshold are removed 

  Returns:
    Top top k entries from `scores` after thresholding with `t` is applied,
    as well as their indexes in the original vector. The values are ordered in
    ascending order.
  """
  idx = np.argsort(scores)[-k:]
  top_k_scores = scores[idx]
  idx = idx[np.where(top_k_scores >= t)[0]]
  return scores[idx], idx


def filter_image_annotations(args):
  """This method applies top-k selection + filtering on image data.

  The filtering can be applied either at a per class granularity or across
  all the classes at once.

  This method is generally meant to be used in a `multiprocessing.Pool`, however
  it can be used independently as well.

  Args:
    args: a tuple containing the following:
      * bboxes: an array of the form `(A, 4)` where `A` is the number of anchors
      * scores: an array of the form `(A, K)` where `K` is the number of classes
      * k: the number of maximum elements to be selected 
      * per_class: if True does the filtering on a per class level
      * t: the threshold value

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    count (since all the other output structures may be padded). If 
    `per_class=True` the first dimension will be equal to `k * K` and `j` 
    otherwise. 
  """
  # Unpack the arguments
  bboxes, scores, k, per_class, t = args

  # The function below is responsible with filtering the image data
  def _inner(i_scores, i_labels):
    i_scores, idx = top_k(i_scores, k, t)
    i_labels = i_labels[idx]
    i_bboxes = bboxes[idx]
    return i_bboxes, i_scores, i_labels, i_bboxes.shape[0]

  # Create the dimension variables
  class_count = scores.shape[-1]
  row_count = k * class_count if per_class else k

  # Create the accumulators
  usable_rows = 0
  bbox_acc = np.zeros((row_count, 4))
  scores_acc = np.zeros(row_count)
  labels_acc = np.zeros(row_count, dtype=int)

  if per_class:
    # We'll need to keep track of where to add the intermediate results
    start_idx = 0

    # Start processing each class
    for i in range(class_count):
      class_scores = scores[:, i]
      class_labels = np.ones(class_scores.shape[0], dtype=int) * i
      res = _inner(class_scores, class_labels)

      # Add results to accumulators
      end_idx = start_idx + res[3]
      bbox_acc[start_idx:end_idx, :] = res[0]
      scores_acc[start_idx:end_idx] = res[1]
      labels_acc[start_idx:end_idx] = res[2]

      # Update the start_idx to point to the next location in the acc
      start_idx = end_idx

    # Save the number of usable rows
    usable_rows = start_idx
  else:
    # Get the label for each anchor, based on confidence
    class_labels = np.argmax(scores, axis=1)
    class_scores = scores[np.arange(scores.shape[0]), class_labels]
    res = _inner(class_scores, class_labels)

    # Assign the computed values to the accumulators
    usable_rows = res[3]
    bbox_acc[:usable_rows, :] = res[0]
    scores_acc[:usable_rows] = res[1]
    labels_acc[:usable_rows] = res[2]

  # Prepare the data structures for return
  usable_rows = [usable_rows]
  bbox_acc = np.expand_dims(bbox_acc, axis=0)
  scores_acc = np.expand_dims(scores_acc, axis=0)
  labels_acc = np.expand_dims(labels_acc, axis=0)

  return bbox_acc, scores_acc, labels_acc, usable_rows


def filter_layer_detections(bboxes, scores, k=1000, per_class=False, t=0.05):
  """Filter the detections obtained at a layer in the FPN.

  Filtering can either be done at a per class level or on all the classes. 
  Filtering itself implies, thresholding out detections with confidence lower 
  than `t`, and selecting the top `k` entries after.

  Args:
    bboxes: an array of the form `(N, A, 4)` where `N` is the batch size 
      and `A` is the number of anchors
    scores: an array of the form `(N, A, K)` where `K` is the number of 
      classes
    k: the number of maximum elements to be selected 
    per_class: if True does the filtering on a per class level
    t: the threshold value

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    counts. Note that if `per_class=True`, then the output shape across 
    the 2nd dimension is `k * K`. Otherwise, it is just `k`.
  """
  batch_size = scores.shape[0]

  # Leverage multiple cores to do the filtering
  pool = Pool(processes=batch_size)
  args = [(bboxes[i], scores[i], k, per_class, t) for i in range(batch_size)]
  res = pool.map(filter_image_annotations, args)

  # Construct the batch results
  return_tuple = ()
  for i in range(4):
    # Special case when we're collecting the usable_rows
    if i == 3:
      acc = []
      for j in range(batch_size):
        acc += res[j][i]
      acc = np.array(acc)
    else:
      acc = res[0][i]
      for j in range(1, batch_size):
        acc = np.append(acc, res[j][i], axis=0)

    # Concatenate to return tuple
    return_tuple = return_tuple + (acc,)

  # Return the batch filtered layer
  return return_tuple


def nms_image(args):
  """This method applies Non-Maximum Suppresion on bboxes.

  The filtering can be applied either at a per class granularity or across
  all the classes at once.

  This method is generally meant to be used in a `multiprocessing.Pool`, however
  it can be used independently as well.

  Args:
    args: a tuple containing the following:
      * bboxes: an array of the form `(A, 4)` where `A` is the number of anchors
      * scores: a vector of length `A` storing the confidence of the bbox
      * labels: a vector of length `A` storing the label of the bbox
      * usable_rows: number of rows in the data structures which are not padding
      * t: the NMS IoU threshold value
      * max_detections: the number of bboxes to be selected 
      * per_class: if True applies NMS on a per class level
      * classes: the number of classes object detection task

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    count (since all the other output structures may be padded). The output
    will always have `max_detections` rows.
  """
  # Unpack the arguments
  bboxes, scores, labels, usable_rows, t, max_detections, per_class, classes = args

  # Remove the padding
  bboxes = bboxes[:usable_rows]
  scores = scores[:usable_rows]
  labels = labels[:usable_rows]

  if per_class:
    # Create the accumulators
    bbox_acc = np.zeros((0, 4))
    scores_acc = np.zeros(0)
    labels_acc = np.zeros(0)

    # Apply NMS per class
    for i in range(classes):
      idx = np.where(labels == i)[0]

      # Check that there are entries for this class
      if idx.shape[0] != 0:
        bboxes_t = bboxes[idx]
        scores_t = scores[idx]
        labels_t = labels[idx]

        bboxes_t, idx = non_max_suppression(bboxes_t, scores_t, t)

        bbox_acc = np.append(bbox_acc, bboxes_t, axis=0)
        scores_acc = np.append(scores_acc, scores_t[idx], axis=0)
        labels_acc = np.append(labels_acc, labels_t[idx], axis=0)

    # Top k needs to be applied to the bboxes, in order to choose the best
    _, idx = top_k(scores_acc, max_detections)
    idx = np.flip(idx)  # Sort in descending order for the later selection
    scores = scores_acc[idx]
    bboxes = bbox_acc[idx]
    labels = labels_acc[idx]
  else:
    # Apply NMS across all the classes
    bboxes, idx = non_max_suppression(bboxes, scores, t)
    scores = scores[idx]
    labels = labels[idx]

  # Either pad or remove excess entries
  detection_count = bboxes.shape[0]
  if detection_count >= max_detections:
    usable_rows = max_detections
    bboxes = bboxes[:max_detections]
    scores = scores[:max_detections]
    labels = labels[:max_detections]
  elif detection_count < max_detections:
    usable_rows = detection_count
    pad_count = max_detections - usable_rows
    bboxes = vertical_pad_np(bboxes, pad_count)
    scores = vertical_pad_np(scores, pad_count)
    labels = vertical_pad_np(labels, pad_count, dtype=int)

  # Prepare the data structures for return
  usable_rows = [usable_rows]
  bboxes = np.expand_dims(bboxes, axis=0)
  scores = np.expand_dims(scores, axis=0)
  labels = np.expand_dims(labels, axis=0)

  return bboxes, scores, labels, usable_rows


def nms_batch(bboxes,
              scores,
              labels,
              usable_rows,
              classes,
              max_detections=100,
              per_class=False,
              t=0.5):
  """Filter the detections obtained at a layer in the FPN.

  Filtering can either be done at a per class level or on all the classes. 
  Filtering itself implies, thresholding out detections with confidence lower 
  than `t`, and selecting the top `k` entries after.

  Args:
    bboxes: an array of the form `(N, A, 4)` where `N` is the batch size 
      and `A` is the number of anchors
    scores: an array of the form `(N, A)` storing the confidence in each anchor
    labels: an array of the form `(N, A)` storing the label of each anchor
    usable_rows: a vector of length `N` storing the number of non-padded rows
      for each image
    classes: the number of classes in the object detection task
    max_detections: the number of maximum detections to be extracted
    per_class: if True applies NMS on a per class level
    t: the NMS IoU threshold value

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    counts. The output shape across the 2nd dimension is `max_detections`.
  """
  batch_size = scores.shape[0]

  # Leverage multiple cores to do the filtering
  pool = Pool(processes=batch_size)
  args = [(bboxes[i], scores[i], labels[i], usable_rows[i], t, max_detections,
           per_class, classes) for i in range(batch_size)]
  res = pool.map(nms_image, args)

  # Construct the batch results
  return_tuple = ()
  for i in range(4):
    # Special case when we're collecting the usable_rows
    if i == 3:
      acc = []
      for j in range(batch_size):
        acc += res[j][i]
      acc = np.array(acc)
    else:
      acc = res[0][i]
      for j in range(1, batch_size):
        acc = np.append(acc, res[j][i], axis=0)

    # Concatenate to return tuple
    return_tuple = return_tuple + (acc,)

  # Return the batch filtered layer
  return return_tuple


def process_inferences(inference_dict):
  """Processes the inferences produced by a model.

  The processing refers to top-k selection, and NMS.

  """
