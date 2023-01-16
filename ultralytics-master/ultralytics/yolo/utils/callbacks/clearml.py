# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import clearml
    from clearml import Task

    assert hasattr(clearml, '__version__')
except (ImportError, AssertionError):
    clearml = None


def _log_images(imgs_dict, group="", step=0):
    task = Task.current_task()
    if task:
        for k, v in imgs_dict.items():
            task.get_logger().report_image(group, k, step, v)


def on_pretrain_routine_start(trainer):
    # TODO: reuse existing task
    task = Task.init(project_name=trainer.args.project or "YOLOv8",
                     task_name=trainer.args.name,
                     tags=['YOLOv8'],
                     output_uri=True,
                     reuse_last_task_id=False,
                     auto_connect_frameworks={'pytorch': False})
    task.connect(dict(trainer.args), name='General')


def on_train_epoch_end(trainer):
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob('train_batch*.jpg')}, "Mosaic", trainer.epoch)


def on_fit_epoch_end(trainer):
    if trainer.epoch == 0:
        model_info = {
            "Parameters": get_num_params(trainer.model),
            "GFLOPs": round(get_flops(trainer.model), 3),
            "Inference speed (ms/img)": round(trainer.validator.speed[1], 3)}
        Task.current_task().connect(model_info, name='Model')


def on_train_end(trainer):
    Task.current_task().update_output_model(model_path=str(trainer.best),
                                            model_name=trainer.args.name,
                                            auto_delete_file=False)


callbacks = {
    "on_pretrain_routine_start": on_pretrain_routine_start,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end} if clearml else {}
