# Ultralytics YOLO 🚀, GPL-3.0 license

import glob
import inspect
import math
import platform
import urllib
from pathlib import Path
from subprocess import check_output
from typing import Optional

import cv2
import numpy as np
import pkg_resources as pkg
import torch

from ultralytics.yolo.utils import (AUTOINSTALL, FONT, LOGGER, ROOT, USER_CONFIG_DIR, TryExcept, colorstr, emojis,
                                    is_docker, is_jupyter_notebook)


def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)


def check_imgsz(imgsz, stride=32, min_dim=1, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int or List[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        List[int]: Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]

    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {stride}, updating to {sz}')

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False,
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        bool: True if minimum version is met, False otherwise.
    """
    from pkg_resources import parse_version
    current, minimum = (parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    warning_message = f"WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed"
    if hard:
        assert result, emojis(warning_message)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result


def check_font(font: str = FONT, progress: bool = False) -> None:
    """
    Download font file to the user's configuration directory if it does not already exist.

    Args:
        font (str): Path to font file.
        progress (bool): If True, display a progress bar during the download.

    Returns:
        None
    """
    font = Path(font)

    # Destination path for the font file
    file = USER_CONFIG_DIR / font.name

    # Check if font file exists at the source or destination path
    if not font.exists() and not file.exists():
        # Download font file
        url = f'https://ultralytics.com/assets/{font.name}'
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def check_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    import socket
    try:
        # Check host accessibility by attempting to establish a connection
        socket.create_connection(("1.1.1.1", 443), timeout=5)
        return True
    except OSError:
        return False


def check_python(minimum: str = '3.7.0') -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.

    Returns:
        None
    """
    check_version(platform.python_version(), minimum, name='Python ', hard=True)


@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt', exclude=(), install=True, cmds=''):
    # Check installed dependencies meet YOLOv5 requirements (pass *.txt file or list of packages or single package str)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            s += f'"{r}" '
            n += 1

    if s and install and AUTOINSTALL:  # check environment variable
        LOGGER.info(f"{prefix} YOLOv8 requirement{'s' * (n > 1)} {s}not found, attempting AutoUpdate...")
        try:
            assert check_online(), "AutoUpdate skipped (offline)"
            LOGGER.info(check_output(f'pip install {s} {cmds}', shell=True).decode())
            source = file if 'file' in locals() else requirements
            s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
                f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
            LOGGER.info(s)
        except Exception as e:
            LOGGER.warning(f'{prefix} ❌ {e}')


def check_suffix(file='yolov8n.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            LOGGER.info(f'Found {url} locally at {file}')  # file already exists
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        for d in 'models', 'yolo/data':  # search directories
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_imshow(warn=False):
    # Check if environment supports image displays
    try:
        assert not is_jupyter_notebook()
        assert not is_docker()
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


def git_describe(path=ROOT):  # path must be a directory
    # Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    try:
        assert (Path(path) / '.git').is_dir()
        return check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    except Exception:
        return ''


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
