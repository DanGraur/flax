from absl import app
from absl import flags
import flax
from flax.training import checkpoints
from functools import partial
import jax
from ml_collections import config_flags
import tensorflow as tf

import input_pipeline
from train import State, create_model, create_optimizer, infer, post_process_inferences, restore_checkpoint

# Config TPU.
try:
  import tpu_magic
except ImportError:
  print("Did not configure TPU.")

# Flag section
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Location of model checkpoint.")
config_flags.DEFINE_config_file(
    "config", "configs/default.py", "Training configuration.", lock_config=True)
flags.DEFINE_string(
    "jax_backend_target", None,
    "JAX backend target to use. Set this to grpc://<TPU_IP_ADDRESS>:8740 to " \
    "use your TPU (2VM mode)."
)
flags.mark_flag_as_required('checkpoint_dir')


def produce_inferences(checkpoint_dir, config):
  # Use the COCO 2014 dataset
  rng = jax.random.PRNGKey(config.seed)
  rng, data_rng = jax.random.split(rng)
  ds_info, train_data, val_data = input_pipeline.create_datasets(
      config, data_rng)
  num_classes = ds_info.features["objects"]["label"].num_classes
  input_shape = list(train_data.element_spec["image"].shape)[1:]

  # Create a dummy state
  rng, model_rng = jax.random.split(rng)
  model, model_state = create_model(
      model_rng, shape=input_shape, classes=num_classes, depth=config.depth)
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
      apply_filtering=config.apply_filtering,
      per_level=config.per_level,
      per_class=config.per_class,
      level_detections=config.level_detections,
      max_detections=config.max_detections,
      confidence_threshold=config.confidence_threshold,
      iou_threshold=config.iou_threshold)

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

  # produce_inferences(FLAGS.checkpoint_dir, FLAGS.config)
  itr = iter(produce_inferences(FLAGS.checkpoint_dir, FLAGS.config))
  print(next(itr))


if __name__ == "__main__":
  app.run(main)
