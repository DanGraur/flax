from typing import Dict, Iterable, Mapping, Tuple

from jax import numpy as jnp
import jax
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import tensorflow as tf


def pi_init(pi: float):
  """Wrapper to log-based weight initializer function.

  This initializer is used for the bias term in the classification subnet, as
  described in https://arxiv.org/abs/1708.02002

  Args:
    pi: the prior probability of detecting an object

  Returns:
    A function used for initializing a module's weights / biases
  """

  def _inner(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * (-jnp.log((1 - pi) / pi))

  return _inner


def non_max_suppression(bboxes: jnp.ndarray, scores: jnp.ndarray, t: float):
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


def vertical_pad(data: jnp.ndarray,
                 pad_count: int,
                 dtype=jnp.float32) -> jnp.ndarray:
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


def vertical_pad_np(data: np.ndarray, pad_count: int, dtype=float):
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


def top_k(scores: jnp.ndarray, k: int, t: float = 0.0):
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


def filter_image_annotations(bboxes: np.ndarray, scores: np.ndarray, k: int,
                             per_class: bool, t: float):
  """This method applies top-k selection + filtering on image data.

  The filtering can be applied either at a per class granularity or across
  all the classes at once.

  Args:
    bboxes: an array of the form `(A, 4)` where `A` is the number of anchors
    scores: an array of the form `(A, K)` where `K` is the number of classes
    k: the number of maximum elements to be selected 
    per_class: if True does the filtering on a per class level
    t: the threshold value

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    count (since all the other output structures may be padded). If 
    `per_class=True` the first dimension will be equal to `k * K` and `j` 
    otherwise. 
  """

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


def filter_layer_detections(bboxes: np.ndarray,
                            scores: np.ndarray,
                            k: int = 1000,
                            per_class: bool = False,
                            t: float = 0.05):
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
  args = [(bboxes[i], scores[i], k, per_class, t) for i in range(batch_size)]
  with Pool(processes=batch_size) as pool:
    res = pool.starmap(filter_image_annotations, args)

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


def nms_image(bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
              usable_rows: int, t: float, max_detections: int, per_class: bool,
              classes: int):
  """This method applies Non-Maximum Suppresion on bboxes.

  The filtering can be applied either at a per class granularity or across
  all the classes at once.

  Args:
    bboxes: an array of the form `(A, 4)` where `A` is the number of anchors
    scores: a vector of length `A` storing the confidence of the bbox
    labels: a vector of length `A` storing the label of the bbox
    usable_rows: number of rows in the data structures which are not padding
    t: the NMS IoU threshold value
    max_detections: the number of bboxes to be selected 
    per_class: if True applies NMS on a per class level
    classes: the number of classes object detection task

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    count (since all the other output structures may be padded). The output
    will always have `max_detections` rows.
  """
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


def nms_batch(bboxes: np.ndarray,
              scores: np.ndarray,
              labels: np.ndarray,
              usable_rows: np.ndarray,
              classes: int,
              max_detections: int = 100,
              per_class: bool = False,
              t: float = 0.5):
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
  args = [(bboxes[i], scores[i], labels[i], usable_rows[i], t, max_detections,
           per_class, classes) for i in range(batch_size)]
  with Pool(processes=batch_size) as pool:
    res = pool.starmap(nms_image, args)

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


def concat_list_element(lst: Iterable, idx: int, usable_rows_idx: int,
                      img_idx: int) -> np.ndarray:
  """Given a list of tuples, concatenates the elements at `idx` together.

  More specifically, this method will iterate through `lst`, and concatenate
  all the elements in each tuple at `idx` together. The elements are expected
  to be batches, so `img_idx` will indicate which image to access in the batch
  for concatenation.

  Args:
    lst: a list which holds tuples of batched arrays, the tuples must also hold
      a structure which indicates how many rows are usable (non-padded)
    idx: the index in the tuples at which the elements to be concatenated
      can be found
    usable_rows_idx: the index of the structure in the tuple which indicates 
      how many rows are usable
    img_idx: the index in the batch, of the image whose data is to be 
      concatenated together
    
  Returns:
    An array, which stores the concatenated data, with an expanded first
    dimension for later stacking with the other images.
  """
  # Create an accumulator with the same shape, but no batch size
  target_size = 0
  acc = np.zeros(lst[0][idx].shape[1:])

  # Concatenate the individual levels together
  for entry in lst:
    usable_rows = entry[usable_rows_idx][img_idx]
    data = entry[idx][img_idx, :usable_rows, ...]
    acc = np.append(acc, data, axis=0)

    # Add to the target_size, so as to know the padding
    target_size += entry[idx].shape[1]

  # Pad the data, if need be
  usable_rows = acc.shape[0]
  if usable_rows < target_size:
    pad_count = target_size - usable_rows
    acc = vertical_pad_np(acc, pad_count)

  # Expand the first dimension to allow for stacking later on
  return np.expand_dims(acc, axis=0), usable_rows


def concat_list_batch(lst: Iterable, idx: int, usable_rows_idx: int,
                      batch_size: int) -> np.ndarray:
  """Given a list of tuples, concatenates the elements at `idx` together.

  More specifically, this method will iterate through `lst`, and concatenate
  all the elements in each tuple at `idx` together. The elements are expected
  to be batches, so the output will also maintain the batch dimension

  Args:
    lst: a list which holds tuples of batched arrays, the tuples must also hold
      a structure which indicates how many rows are usable (non-padded)
    idx: the index in the tuples at which the elements to be concatenated
      can be found
    usable_rows_idx: the index of the structure in the tuple which indicates 
      how many rows are usable
    batch_size: the number of elements in the batch
    
  Returns:
    An array, which stores the concatenated data, where the first dimension is 
    equal to the `batch_size`.
  """
  args = [(lst, idx, usable_rows_idx, i) for i in range(batch_size)]
  with Pool(processes=batch_size) as pool:
    res = pool.starmap(concat_list_element, args)

  return np.concatenate([x[0] for x in res], axis=0), [x[1] for x in res]


def process_inferences(
    inference_dict: Mapping[str, np.ndarray],
    per_level: bool = True,
    per_class: bool = False,
    level_detections: int = 1000,
    max_detections: int = 100,
    confidence_threshold: float = 0.05,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Processes the inferences produced by a model.

  The processing refers to top-k selection, and NMS. It can be done on a per
  FPN level granularity (i.e. top-k and confidence thresholding is done for
  each feature map produced by a classification subnet) if desired. Moreover,
  top-k and NMS can also be applied at a per class granularity. 

  Args:
    inference_dict: a dictionary containing the output of the model in inference
      mode; the dict should have the following structure:

      ```
      {
        <layer_name>: (regressed_anchors, anchor_scores),
        ...
      }
      ```
    per_level: if True, does top-k and thresholding at a per level granularity;
      otherwise it applies the aforementioned operations on all `inference_dict`
      entries at once
    per_class: if True, applies top-k, thresholding and NMS on a per class 
      level; otherwise, these operations are applied for all classes colectively 
    level_detections: the number of detections to be selected during top-k and
      thresholding
    max_detections: the number of detections to be generated by NMS; note that
      unlike other options, `max_detections` does not depend on `per_class`,
      and will produce the same amount of outputs regardless of flag 
      configurations  
    confidence_threshold: the threshold used in confidence thresholding; anchors
      with a confidence score lower than this value are discarded
    iou_threshold: the threshold used in NMS; IoUs above this value are 
      discarded

  Returns:
    A tuple consisting of the filtered bboxes, scores, labels, and usable_row 
    counts. The output shape across the 2nd dimension is `max_detections`.
  """
  # Initialize the variables relevant for this process
  shape = inference_dict[list(inference_dict.keys())[0]][1].shape
  batch_size = shape[0]
  class_count = shape[-1]

  # Do the top-k and thresholding
  if per_level:
    # ThreadPools are used, since nested daemonic processes are disallowed
    # Apply for each level in the FPN
    args = [(item[0], item[1], level_detections, per_class,
             confidence_threshold) for item in inference_dict.values()]
    with ThreadPool(processes=len(inference_dict)) as pool:
      res = pool.starmap(filter_layer_detections, args)

    # res is a list of (bboxes, scores, labels, usable_rows); concat first 3
    args = [(res, i, 3, batch_size) for i in range(3)]
    with ThreadPool(processes=3) as pool:
      res = pool.starmap(concat_list_batch, args)

    # Unpack the elements of res
    bboxes = res[0][0]
    scores = res[1][0]
    labels = res[2][0]
    usable_rows = res[2][1]

  else:
    # Re-structure the level results, such that they can be concatenated
    bboxes = np.concatenate([x[0] for x in inference_dict.values()], axis=1)
    scores = np.concatenate([x[1] for x in inference_dict.values()], axis=1)

    # Apply top-k and thresholding once
    bboxes, scores, labels, usable_rows = filter_layer_detections(
        bboxes, scores, level_detections, per_class, confidence_threshold)

  # Apply NMS
  bboxes, scores, labels, usable_rows = nms_batch(bboxes, scores, labels,
                                                  usable_rows, class_count,
                                                  max_detections, per_class,
                                                  iou_threshold)

  # Return the results, and ensure that labels is of int type
  return bboxes, scores, labels.astype(int), usable_rows
