"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
"""

import os
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf  # noqa
from omegaconf import open_dict
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm

import ultralytics.yolo.utils as utils
from ultralytics import __version__
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.data.utils import check_dataset, check_dataset_yaml
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, RANK, TQDM_BAR_FORMAT, callbacks, colorstr, yaml_save
from ultralytics.yolo.utils.checks import check_file, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import ModelEMA, de_parallel, init_seeds, one_cycle, strip_optimizer


class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (OmegaConf): Configuration for the trainer.
        check_resume (method): Method to check if training should be resumed from a saved checkpoint.
        console (logging.Logger): Logger instance.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to last checkpoint.
        best (Path): Path to best checkpoint.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        """
        Initializes the BaseTrainer class.

        Args:
            config (str, optional): Path to a configuration file. Defaults to DEFAULT_CONFIG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        if overrides is None:
            overrides = {}
        self.args = get_config(config, overrides)
        self.check_resume()
        self.console = LOGGER
        self.validator = None
        self.model = None
        self.callbacks = defaultdict(list)
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = self.args.project or f"runs/{self.args.task}"
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = Path(
            self.args.get(
                "save_dir",
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in {-1, 0} else True)))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            with open_dict(self.args):
                self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', OmegaConf.to_container(self.args, resolve=True))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(dict(self.args))

        # Device
        self.device = utils.torch_utils.select_device(self.args.device, self.batch_size)
        self.amp = self.device.type != 'cpu'
        self.scaler = amp.GradScaler(enabled=self.amp)
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataloaders.
        self.model = self.args.model
        self.data = self.args.data
        if self.data.endswith(".yaml"):
            self.data = check_dataset_yaml(self.data)
        else:
            self.data = check_dataset(self.data)
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = None
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = defaultdict(list, {k: [v] for k, v in callbacks.default_callbacks.items()})  # add callbacks
        if RANK in {0, -1}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback. TODO: unused, consider removing
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.  TODO: unused, consider removing
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        world_size = torch.cuda.device_count()
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            command = generate_ddp_command(world_size, self)
            try:
                subprocess.run(command)
            except Exception as e:
                self.console(e)
            finally:
                ddp_cleanup(command, self)
        else:
            self._do_train(int(os.getenv("RANK", -1)), world_size)

    def _setup_ddp(self, rank, world_size):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '9020'
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', rank)
        self.console.info(f"DDP settings: RANK {rank}, WORLD_SIZE {world_size}, DEVICE {self.device}")
        dist.init_process_group("nccl" if dist.is_nccl_available() else "gloo", rank=rank, world_size=world_size)

    def _setup_train(self, rank, world_size):
        """
        Builds dataloaders and optimizer on correct rank process
        """
        # model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.args.weight_decay *= self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=self.args.weight_decay)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        # dataloaders
        batch_size = self.batch_size // world_size if world_size > 1 else self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=rank, mode="train")
        if rank in {0, -1}:
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode="val")
            self.validator = self.get_validator()
            metric_keys = self.validator.metric_keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
        self.resume_training(ckpt)
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        self._setup_train(rank, world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks("on_train_start")
        self.log(f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
                 f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                 f"Logging results to {colorstr('bold', self.save_dir)}\n"
                 f"Starting training for {self.epochs} epochs...")
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            self.model.train()
            if rank != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self.console.info("Closing dataloader mosaic")
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)

            if rank in {-1, 0}:
                self.console.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch["img"])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if rank != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if rank in {-1, 0}:
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            lr = {f"lr{ir}": x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.scheduler.step()
            self.run_callbacks("on_train_epoch_end")

            if rank in {-1, 0}:

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs)
                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **lr})

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks("on_fit_epoch_end")
            # TODO: termination condition

        if rank in {-1, 0}:
            # Do final val with best.pt
            self.log(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                     f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.log(f"Results saved to {colorstr('bold', self.save_dir)}")
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': self.args,
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        del ckpt

    def get_dataset(self, data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized
        """
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """
        load/create/download model for any task
        """
        if isinstance(self.model, torch.nn.Module):  # if loaded model is passed
            return
            # We should improve the code flow here. This function looks hacky
        model = self.model
        pretrained = not str(model).endswith(".yaml")
        # config
        if not pretrained:
            model = check_file(model)
        ckpt = self.load_ckpt(model) if pretrained else None
        weights = ckpt["model"] if isinstance(ckpt, dict) else ckpt  # torchvision weights are not dicts
        self.model = self.load_model(model_cfg=None if pretrained else model, weights=weights)
        return ckpt

    def load_ckpt(self, ckpt):
        return torch.load(ckpt, map_location='cpu')

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.
        # TODO: discuss validator class. Enforce that a validator metrics dict should contain
        "fitness" metric.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def log(self, text, rank=-1):
        """
        Logs the given text to given ranks process if provided, otherwise logs to all ranks
        :param text: text to log
        :param rank: List[Int]

        """
        if rank in {-1, 0}:
            self.console.info(text)

    def load_model(self, model_cfg=None, weights=None, verbose=True):
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0):
        """
        Returns dataloader derived from torch.data.Dataloader
        """
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor
        """
        raise NotImplementedError("criterion function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        pass

    def progress_string(self):
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        pass

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        pass

    def final_eval(self):
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    self.console.info(f'\nValidating {f}...')
                    self.validator.args.save_json = True
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        resume = self.args.resume
        if resume:
            last = Path(check_file(resume) if isinstance(resume, str) else get_latest_run())
            args_yaml = last.parent.parent / 'args.yaml'  # train options yaml
            if args_yaml.is_file():
                args = get_config(args_yaml)  # replace
            args.model, resume = str(last), True  # reinstate
            self.args = args
        self.resume = resume

    def resume_training(self, ckpt):
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without --resume, i.e. 'yolo task=... mode=train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch

    @staticmethod
    def build_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        """
        Builds an optimizer with the specified parameters and parameter groups.

        Args:
            model (nn.Module): model to optimize
            name (str): name of the optimizer to use
            lr (float): learning rate
            momentum (float): momentum
            decay (float): weight decay

        Returns:
            optimizer (torch.optim.Optimizer): the built optimizer
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                    f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
        return optimizer
