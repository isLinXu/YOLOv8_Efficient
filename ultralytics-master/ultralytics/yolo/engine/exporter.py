# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Export a YOLOv8 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlmodel
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Python:
    from ultralytics import YOLO
    model = YOLO('yolov8n.yaml')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pt format=onnx

Inference:
    $ python detect.py --weights yolov8n.pt                 # PyTorch
                                 yolov8n.torchscript        # TorchScript
                                 yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov8n_openvino_model     # OpenVINO
                                 yolov8n.engine             # TensorRT
                                 yolov8n.mlmodel            # CoreML (macOS-only)
                                 yolov8n_saved_model        # TensorFlow SavedModel
                                 yolov8n.pb                 # TensorFlow GraphDef
                                 yolov8n.tflite             # TensorFlow Lite
                                 yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov8n_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov8n_web_model public/yolov8n_web_model
    $ npm start
"""
import contextlib
import json
import os
import platform
import re
import subprocess
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch

import ultralytics
from ultralytics.nn.modules import Detect, Segment
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages
from ultralytics.yolo.data.utils import check_dataset
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, callbacks, colorstr, get_default_args, yaml_save
from ultralytics.yolo.utils.checks import check_imgsz, check_requirements, check_version, check_yaml
from ultralytics.yolo.utils.files import file_size
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import guess_task_from_head, select_device, smart_inference_mode

MACOS = platform.system() == 'Darwin'  # macOS environment


def export_formats():
    # YOLOv8 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv8 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


class Exporter:
    """
    Exporter

    A class for exporting a model.

    Attributes:
        args (OmegaConf): Configuration for the exporter.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        """
        Initializes the Exporter class.

        Args:
            config (str, optional): Path to a configuration file. Defaults to DEFAULT_CONFIG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        if overrides is None:
            overrides = {}
        self.args = get_config(config, overrides)
        self.callbacks = defaultdict(list, {k: [v] for k, v in callbacks.default_callbacks.items()})  # add callbacks
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def __call__(self, model=None):
        self.run_callbacks("on_export_start")
        t = time.time()
        format = self.args.format.lower()  # to lowercase
        fmts = tuple(export_formats()['Argument'][1:])  # available export formats
        flags = [x == format for x in fmts]
        assert sum(flags), f'ERROR: Invalid format={format}, valid formats are {fmts}'
        jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans

        # Load PyTorch model
        self.device = select_device('cpu' if self.args.device is None else self.args.device)
        if self.args.half:
            if self.device.type == 'cpu' and not coreml:
                LOGGER.info('half=True only compatible with GPU or CoreML export, i.e. use device=0 or format=coreml')
                self.args.half = False
            assert not self.args.dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic'

        # Checks
        # if self.args.batch == model.args['batch_size']:  # user has not modified training batch_size
        self.args.batch = 1
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert self.device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
        file = Path(getattr(model, 'pt_path', None) or getattr(model, 'yaml_file', None) or model.yaml['yaml_file'])
        if file.suffix == '.yaml':
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for k, m in model.named_modules():
            if isinstance(m, (Detect, Segment)):
                m.dynamic = self.args.dynamic
                m.export = True

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if self.args.half and not coreml:
            im, model = im.half(), model.half()  # to FP16
        shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

        # Warnings
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else (x.shape for x in y)
        self.metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
        self.pretty_name = self.file.stem.replace('yolo', 'YOLO')

        # Exports
        f = [''] * len(fmts)  # exported filenames
        if jit:  # TorchScript
            f[0], _ = self._export_torchscript()
        if engine:  # TensorRT required before ONNX
            f[1], _ = self._export_engine()
        if onnx or xml:  # OpenVINO requires ONNX
            f[2], _ = self._export_onnx()
        if xml:  # OpenVINO
            f[3], _ = self._export_openvino()
        if coreml:  # CoreML
            f[4], _ = self._export_coreml()
        if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
            raise NotImplementedError('YOLOv8 TensorFlow export support is still under development. '
                                      'Please consider contributing to the effort if you have TF expertise. Thank you!')
            assert not isinstance(model, ClassificationModel), 'ClassificationModel TF exports not yet supported.'
            nms = False
            f[5], s_model = self._export_saved_model(nms=nms or self.args.agnostic_nms or tfjs,
                                                     agnostic_nms=self.args.agnostic_nms or tfjs)
            if pb or tfjs:  # pb prerequisite to tfjs
                f[6], _ = self._export_pb(s_model)
            if tflite or edgetpu:
                f[7], _ = self._export_tflite(s_model,
                                              int8=self.args.int8 or edgetpu,
                                              data=self.args.data,
                                              nms=nms,
                                              agnostic_nms=self.args.agnostic_nms)
                if edgetpu:
                    f[8], _ = self._export_edgetpu()
                self._add_tflite_metadata(f[8] or f[7], num_outputs=len(s_model.outputs))
            if tfjs:
                f[9], _ = self._export_tfjs()
        if paddle:  # PaddlePaddle
            f[10], _ = self._export_paddle()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            task = guess_task_from_head(model.yaml["head"][-1][-2])
            s = "-WARNING ⚠️ not yet supported for YOLOv8 exported models"
            LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                        f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                        f"\nPredict:         yolo task={task} mode=predict model={f[-1]} {s}"
                        f"\nValidate:        yolo task={task} mode=val model={f[-1]} {s}"
                        f"\nVisualize:       https://netron.app")

        self.run_callbacks("on_export_end")
        return f  # return list of exported files/dirs

    @try_export
    def _export_torchscript(self, prefix=colorstr('TorchScript:')):
        # YOLOv8 TorchScript model export
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = self.file.with_suffix('.torchscript')

        ts = torch.jit.trace(self.model, self.im, strict=False)
        d = {"shape": self.im.shape, "stride": int(max(self.model.stride)), "names": self.model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f'{prefix} optimizing for mobile...')
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def _export_onnx(self, prefix=colorstr('ONNX:')):
        # YOLOv8 ONNX export
        check_requirements('onnx>=1.12.0')
        import onnx  # noqa

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = str(self.file.with_suffix('.onnx'))

        output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # --dynamic only compatible with cpu
            self.im.cpu() if dynamic else self.im,
            f,
            verbose=False,
            opset_version=self.args.opset,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(self.model.stride)), 'names': self.model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if self.args.simplify:
            try:
                check_requirements('onnxsim')
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                subprocess.run(f'onnxsim {f} {f}', shell=True)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        return f, model_onnx

    @try_export
    def _export_openvino(self, prefix=colorstr('OpenVINO:')):
        # YOLOv8 OpenVINO export
        check_requirements('openvino-dev')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        import openvino.inference_engine as ie  # noqa

        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(self.file).replace(self.file.suffix, f'_openvino_model{os.sep}')
        f_onnx = self.file.with_suffix('.onnx')

        cmd = f"mo --input_model {f_onnx} --output_dir {f} --data_type {'FP16' if self.args.half else 'FP32'}"
        subprocess.run(cmd.split(), check=True, env=os.environ)  # export
        yaml_save(Path(f) / self.file.with_suffix('.yaml').name, self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def _export_paddle(self, prefix=colorstr('PaddlePaddle:')):
        # YOLOv8 Paddle export
        check_requirements(('paddlepaddle', 'x2paddle'))
        import x2paddle  # noqa
        from x2paddle.convert import pytorch2paddle  # noqa

        LOGGER.info(f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
        f = str(self.file).replace(self.file.suffix, f'_paddle_model{os.sep}')

        pytorch2paddle(module=self.model, save_dir=f, jit_type='trace', input_examples=[self.im])  # export
        yaml_save(Path(f) / self.file.with_suffix('.yaml').name, self.metadata)  # add metadata.yaml
        return f, None

    @try_export
    def _export_coreml(self, prefix=colorstr('CoreML:')):
        # YOLOv8 CoreML export
        check_requirements('coremltools>=6.0')
        import coremltools as ct  # noqa

        class iOSModel(torch.nn.Module):
            # Wrap an Ultralytics YOLO model for iOS export
            def __init__(self, model, im):
                super().__init__()
                b, c, h, w = im.shape  # batch, channel, height, width
                self.model = model
                self.nc = len(model.names)  # number of classes
                if w == h:
                    self.normalize = 1.0 / w  # scalar
                else:
                    self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)

            def forward(self, x):
                xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
                return cls, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)

        LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
        f = self.file.with_suffix('.mlmodel')

        model = iOSModel(self.model, self.im).eval() if self.args.nms else self.model
        ts = torch.jit.trace(model, self.im, strict=False)  # TorchScript model
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=self.im.shape, scale=1 / 255, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if self.args.int8 else (16, 'linear') if self.args.half else (32, None)
        if bits < 32:
            if MACOS:  # quantization only supported on macOS
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                LOGGER.info(f'{prefix} quantization only supported on macOS, skipping...')
        if self.args.nms:
            ct_model = self._pipeline_coreml(ct_model)

        ct_model.save(str(f))
        return f, ct_model

    @try_export
    def _export_engine(self, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
        # YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt
        assert self.im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `device==0`'
        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if platform.system() == 'Linux':
                check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
            import tensorrt as trt  # noqa

        check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=8.0.0
        self._export_onnx()
        onnx = self.file.with_suffix('.onnx')

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = self.file.with_suffix('.engine')  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if self.args.dynamic:
            shape = self.im.shape
            if shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
            config.add_optimization_profile(profile)

        LOGGER.info(
            f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and self.args.half else 32} engine as {f}')
        if builder.platform_has_fast_fp16 and self.args.half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        return f, None

    @try_export
    def _export_saved_model(self,
                            nms=False,
                            agnostic_nms=False,
                            topk_per_class=100,
                            topk_all=100,
                            iou_thres=0.45,
                            conf_thres=0.25,
                            prefix=colorstr('TensorFlow SavedModel:')):

        # YOLOv8 TensorFlow SavedModel export
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
            import tensorflow as tf  # noqa
        check_requirements(("onnx", "onnx2tf", "sng4onnx", "onnxsim", "onnx_graphsurgeon"),
                           cmds="--extra-index-url https://pypi.ngc.nvidia.com ")

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(self.file).replace(self.file.suffix, '_saved_model')

        # Export to ONNX
        self._export_onnx()
        onnx = self.file.with_suffix('.onnx')

        # Export to TF SavedModel
        subprocess.run(f'onnx2tf -i {onnx} --output_signaturedefs -o {f}', shell=True)

        # Load saved_model
        keras_model = tf.saved_model.load(f, tags=None, options=None)

        return f, keras_model

    @try_export
    def _export_saved_model_OLD(self,
                                nms=False,
                                agnostic_nms=False,
                                topk_per_class=100,
                                topk_all=100,
                                iou_thres=0.45,
                                conf_thres=0.25,
                                prefix=colorstr('TensorFlow SavedModel:')):
        # YOLOv8 TensorFlow SavedModel export
        try:
            import tensorflow as tf  # noqa
        except ImportError:
            check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
            import tensorflow as tf  # noqa
        # from models.tf import TFModel
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(self.file).replace(self.file.suffix, '_saved_model')
        batch_size, ch, *imgsz = list(self.im.shape)  # BCHW

        tf_models = None  # TODO: no TF modules available
        tf_model = tf_models.TFModel(cfg=self.model.yaml, model=self.model.cpu(), nc=self.model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if self.args.dynamic else batch_size)
        outputs = tf_model.predict(inputs, nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if self.args.keras:
            keras_model.save(f, save_format='tf')
        else:
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = tf.function(lambda x: keras_model(x))  # full model
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if nms else frozen_func(x), [spec])
            tfm.__call__(im)
            tf.saved_model.save(tfm,
                                f,
                                options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
                                if check_version(tf.__version__, '2.6') else tf.saved_model.SaveOptions())
        return f, keras_model

    @try_export
    def _export_pb(self, keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
        # YOLOv8 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
        import tensorflow as tf  # noqa
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        return f, None

    @try_export
    def _export_tflite(self, keras_model, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
        # YOLOv8 TensorFlow Lite export
        import tensorflow as tf  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(self.im.shape)  # BCHW
        f = str(self.file).replace(self.file.suffix, '-fp16.tflite')

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:

            def representative_dataset_gen(dataset, n_images=100):
                # Dataset generator for use with converter.representative_dataset, returns a generator of np arrays
                for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
                    im = np.transpose(img, [1, 2, 0])
                    im = np.expand_dims(im, axis=0).astype(np.float32)
                    im /= 255
                    yield [im]
                    if n >= n_images:
                        break

            dataset = LoadImages(check_dataset(check_yaml(data))['train'], imgsz=imgsz, auto=False)
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, n_images=100)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(self.file).replace(self.file.suffix, '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        tflite_model = converter.convert()
        open(f, "wb").write(tflite_model)
        return f, None

    @try_export
    def _export_edgetpu(self, prefix=colorstr('Edge TPU:')):
        # YOLOv8 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
            LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in (
                    'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                    'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '  # no comma
                    'sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                    'sudo apt-get update',
                    'sudo apt-get install edgetpu-compiler'):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
        f = str(self.file).replace(self.file.suffix, '-int8_edgetpu.tflite')  # Edge TPU model
        f_tfl = str(self.file).replace(self.file.suffix, '-int8.tflite')  # TFLite model

        cmd = f"edgetpu_compiler -s -d -k 10 --out_dir {self.file.parent} {f_tfl}"
        subprocess.run(cmd.split(), check=True)
        return f, None

    @try_export
    def _export_tfjs(self, prefix=colorstr('TensorFlow.js:')):
        # YOLOv8 TensorFlow.js export
        check_requirements('tensorflowjs')
        import tensorflowjs as tfjs  # noqa

        LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
        f = str(self.file).replace(self.file.suffix, '_web_model')  # js dir
        f_pb = self.file.with_suffix('.pb')  # *.pb path
        f_json = Path(f) / 'model.json'  # *.json path

        cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
              f'--output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f}'
        subprocess.run(cmd.split())

        with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
            subst = re.sub(
                r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}, '
                r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
                r'"Identity_1": {"name": "Identity_1"}, '
                r'"Identity_2": {"name": "Identity_2"}, '
                r'"Identity_3": {"name": "Identity_3"}}}', f_json.read_text())
            j.write(subst)
        return f, None

    def _add_tflite_metadata(self, file, num_outputs):
        # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
        with contextlib.suppress(ImportError):
            # check_requirements('tflite_support')
            from tflite_support import flatbuffers  # noqa
            from tflite_support import metadata as _metadata  # noqa
            from tflite_support import metadata_schema_py_generated as _metadata_fb  # noqa

            tmp_file = Path('/tmp/meta.txt')
            with open(tmp_file, 'w') as meta_f:
                meta_f.write(str(self.metadata))

            model_meta = _metadata_fb.ModelMetadataT()
            label_file = _metadata_fb.AssociatedFileT()
            label_file.name = tmp_file.name
            model_meta.associatedFiles = [label_file]

            subgraph = _metadata_fb.SubGraphMetadataT()
            subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
            subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
            model_meta.subgraphMetadata = [subgraph]

            b = flatbuffers.Builder(0)
            b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
            metadata_buf = b.Output()

            populator = _metadata.MetadataPopulator.with_model_file(file)
            populator.load_metadata_buffer(metadata_buf)
            populator.load_associated_files([str(tmp_file)])
            populator.populate()
            tmp_file.unlink()

    def _pipeline_coreml(self, model, prefix=colorstr('CoreML Pipeline:')):
        # YOLOv8 CoreML pipeline
        import coremltools as ct  # noqa

        LOGGER.info(f'{prefix} starting pipeline with coremltools {ct.__version__}...')
        batch_size, ch, h, w = list(self.im.shape)  # BCHW

        # Output shapes
        spec = model.get_spec()
        out0, out1 = iter(spec.description.output)
        if MACOS:
            from PIL import Image
            img = Image.new('RGB', (w, h))  # img(192 width, 320 height)
            # img = torch.zeros((*opt.img_size, 3)).numpy()  # img size(320,192,3) iDetection
            out = model.predict({'image': img})
            out0_shape = out[out0.name].shape
            out1_shape = out[out1.name].shape
        else:  # linux and windows can not run model.predict(), get sizes from pytorch output y
            out0_shape = self.output_shape[1], self.output_shape[2] - 5  # (3780, 80)
            out1_shape = self.output_shape[1], 4  # (3780, 4)

        # Checks
        names = self.metadata['names']
        nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
        na, nc = out0_shape
        # na, nc = out0.type.multiArrayType.shape  # number anchors, classes
        assert len(names) == nc, f'{len(names)} names found for nc={nc}'  # check

        # Define output shapes (missing)
        out0.type.multiArrayType.shape[:] = out0_shape  # (3780, 80)
        out1.type.multiArrayType.shape[:] = out1_shape  # (3780, 4)
        # spec.neuralNetwork.preprocessing[0].featureName = '0'

        # Flexible input shapes
        # from coremltools.models.neural_network import flexible_shape_utils
        # s = [] # shapes
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(320, 192))
        # s.append(flexible_shape_utils.NeuralNetworkImageSize(640, 384))  # (height, width)
        # flexible_shape_utils.add_enumerated_image_sizes(spec, feature_name='image', sizes=s)
        # r = flexible_shape_utils.NeuralNetworkImageSizeRange()  # shape ranges
        # r.add_height_range((192, 640))
        # r.add_width_range((192, 640))
        # flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=r)

        # Print
        print(spec.description)

        # Model from spec
        model = ct.models.MLModel(spec)

        # 3. Create NMS protobuf
        nms_spec = ct.proto.Model_pb2.Model()
        nms_spec.specificationVersion = 5
        for i in range(2):
            decoder_output = model._spec.description.output[i].SerializeToString()
            nms_spec.description.input.add()
            nms_spec.description.input[i].ParseFromString(decoder_output)
            nms_spec.description.output.add()
            nms_spec.description.output[i].ParseFromString(decoder_output)

        nms_spec.description.output[0].name = 'confidence'
        nms_spec.description.output[1].name = 'coordinates'

        output_sizes = [nc, 4]
        for i in range(2):
            ma_type = nms_spec.description.output[i].type.multiArrayType
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[0].lowerBound = 0
            ma_type.shapeRange.sizeRanges[0].upperBound = -1
            ma_type.shapeRange.sizeRanges.add()
            ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
            ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
            del ma_type.shape[:]

        nms = nms_spec.nonMaximumSuppression
        nms.confidenceInputFeatureName = out0.name  # 1x507x80
        nms.coordinatesInputFeatureName = out1.name  # 1x507x4
        nms.confidenceOutputFeatureName = 'confidence'
        nms.coordinatesOutputFeatureName = 'coordinates'
        nms.iouThresholdInputFeatureName = 'iouThreshold'
        nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
        nms.iouThreshold = 0.45
        nms.confidenceThreshold = 0.25
        nms.pickTop.perClass = True
        nms.stringClassLabels.vector.extend(names.values())
        nms_model = ct.models.MLModel(nms_spec)

        # 4. Pipeline models together
        pipeline = ct.models.pipeline.Pipeline(input_features=[('image', ct.models.datatypes.Array(3, ny, nx)),
                                                               ('iouThreshold', ct.models.datatypes.Double()),
                                                               ('confidenceThreshold', ct.models.datatypes.Double())],
                                               output_features=['confidence', 'coordinates'])
        pipeline.add_model(model)
        pipeline.add_model(nms_model)

        # Correct datatypes
        pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

        # Update metadata
        pipeline.spec.specificationVersion = 5
        pipeline.spec.description.metadata.versionString = f'Ultralytics YOLOv{ultralytics.__version__}'
        pipeline.spec.description.metadata.shortDescription = f'Ultralytics {self.pretty_name} CoreML model'
        pipeline.spec.description.metadata.author = 'Ultralytics (https://ultralytics.com)'
        pipeline.spec.description.metadata.license = 'GPL-3.0 license (https://ultralytics.com/license)'
        pipeline.spec.description.metadata.userDefined.update({
            'IoU threshold': str(nms.iouThreshold),
            'Confidence threshold': str(nms.confidenceThreshold)})

        # Save the model
        model = ct.models.MLModel(pipeline.spec)
        model.input_description['image'] = 'Input image'
        model.input_description['iouThreshold'] = f'(optional) IOU threshold override (default: {nms.iouThreshold})'
        model.input_description['confidenceThreshold'] = \
            f'(optional) Confidence threshold override (default: {nms.confidenceThreshold})'
        model.output_description['confidence'] = 'Boxes × Class confidence (see user-defined metadata "classes")'
        model.output_description['coordinates'] = 'Boxes × [x, y, width, height] (relative to image size)'
        LOGGER.info(f'{prefix} pipeline success')
        return model

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def export(cfg):
    cfg.model = cfg.model or "yolov8n.yaml"
    cfg.format = cfg.format or "torchscript"

    # exporter = Exporter(cfg)
    #
    # model = None
    # if isinstance(cfg.model, (str, Path)):
    #     if Path(cfg.model).suffix == '.yaml':
    #         model = DetectionModel(cfg.model)
    #     elif Path(cfg.model).suffix == '.pt':
    #         model = attempt_load_weights(cfg.model, fuse=True)
    #     else:
    #         TypeError(f'Unsupported model type {cfg.model}')
    # exporter(model=model)

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.export(**cfg)


if __name__ == "__main__":
    """
    CLI:
    yolo mode=export model=yolov8n.yaml format=onnx
    """
    export()
