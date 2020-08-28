import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.01
  config.per_device_batch_size = 2
  config.num_train_steps = 90_000
  config.warmup_steps = 30_000
  config.half_precision = False
  config.try_restore = False
  config.distributed_training = True

  # The number of layers in the RetinaNet backbone.
  config.depth = 50

  config.sync_steps = 100
  config.checkpoint_period = 1

  # Inference post processing options
  config.apply_filtering = False
  config.per_level = False
  config.per_class = False
  config.level_detections = 1000
  config.max_detections = 100
  config.confidence_threshold = 0.05
  config.iou_threshold = 0.5

  # Evaluation parameters
  config.eval_annotations_path = "/home/dgraur/data/files/coco_annotations/instances_val2014.json"
  config.eval_remove_background = True
  config.eval_threshold = 0.05

  config.seed = 42

  config.trial = 0  # Dummy for repeated runs.
  return config


def get_hyper(h):
  return h.product([
      h.sweep("trial", range(1)),
  ], name="config")
