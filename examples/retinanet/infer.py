from absl import app
from absl import flags
import flax
from flax.training import checkpoints
from functools import partial
import jax
from ml_collections import config_flags
import tensorflow as tf

import input_pipeline
from configs.default import get_config
from train import (State, create_model, create_optimizer, infer,
  post_process_inferences, restore_checkpoint)

# Config TPU.
try:
  import tpu_magic
except ImportError:
  print("Did not configure TPU.")

# Flag section
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Location of model checkpoint.")
flags.DEFINE_integer("seed", 42, "PRNG Seed")
flags.DEFINE_boolean("postprocessing", True, "If postprocessing should be used")
flags.DEFINE_boolean("per_level", False,
                     "If postprocessing should be done per level")
flags.DEFINE_boolean("per_class", True,
                     "If postprocessing should be done per class")
flags.DEFINE_integer(
    "level_detections", 1000,
    "The number of detections to be made per level in Top-K and Thresholding")
flags.DEFINE_integer("max_detections", 100,
                 "The maximal number of detections after NMS")
flags.DEFINE_float("confidence_threshold", 0.05,
                   "The threshold used for thresholding")
flags.DEFINE_float("iou_threshold", 0.5, "The IoU threshold used during NMS")
flags.DEFINE_integer("depth", 50, "The depth of the RetinaNet")
flags.DEFINE_string(
    "jax_backend_target", None,
    "JAX backend target to use. Set this to grpc://<TPU_IP_ADDRESS>:8740 to " \
    "use your TPU (2VM mode)."
)
flags.mark_flag_as_required('checkpoint_dir')


def produce_inferences(checkpoint_dir, seed, postprocessing, per_level,
                       per_class, level_detections, max_detections,
                       confidence_threshold, iou_threshold):
  # Read the default config file, it will be useful later
  config = get_config()

  # Use the COCO 2014 dataset
  rng = jax.random.PRNGKey(seed)
  rng, data_rng = jax.random.split(rng)
  ds_info, train_data, val_data = input_pipeline.create_datasets(
      config, data_rng)
  num_classes = ds_info.features["objects"]["label"].num_classes
  input_shape = list(train_data.element_spec["image"].shape)[1:]

  # Create a dummy state
  rng, model_rng = jax.random.split(rng)
  model, model_state = create_model(
      model_rng, shape=input_shape, classes=num_classes, depth=depth)
  optimizer = create_optimizer(model, beta=0.9, weight_decay=0.0001)
  state = State(optimizer=optimizer, model_state=model_state)
  del model, model_state, optimizer

  # Read the checkpoint
  state = restore_checkpoint(state, checkpoint_dir)
  print(state)
  if state.step == 0:
    raise ValueError(f"The {checkpoint_dir} folder does not contain models.")
  state = flax.jax_utils.replicate(state)
  p_infer_fn = jax.pmap(infer, axis_name="batch")

  # Create a partial function for processing the inferences
  process_inferences_partial_fn = partial(
      post_process_inferences,
      apply_filtering=postprocessing,
      per_level=per_level,
      per_class=per_class,
      level_detections=level_detections,
      max_detections=cmax_detections,
      confidence_threshold=cconfidence_threshold,
      iou_threshold=iou_threshold)

  # Yield inferences one by one
  for step, batch in enumerate(train_data):
    batch = jax.tree_map(lambda x: x._numpy(), batch)
    scores, regressions, inference_dict = p_infer_fn(batch, state)
    yield process_inferences_partial_fn(inference_dict)


def main(argv):
  del argv
  # Turn on omnistaging since it fixes some bugs but is not yet the default.
  jax.config.enable_omnistaging()

  if FLAGS.jax_backend_target:
    # Configure JAX to run in 2VM mode with a remote TPU node.
    jax.config.update("jax_xla_backend", "tpu_driver")
    jax.config.update("jax_backend_target", FLAGS.jax_backend_target)

  # Make sure TF does not allocate memory on the GPU.
  tf.config.experimental.set_visible_devices([], "GPU")

  itr = iter(
      produce_inferences(FLAGS.checkpoint_dir, FLAGS.seed, FLAGS.postprocessing,
                         FLAGS.per_level, FLAGS.per_class,
                         FLAGS.level_detections, FLAGS.max_detections,
                         FLAGS.confidence_threshold, FLAGS.iou_threshold))


if __name__ == "__main__":
  app.run(main)
