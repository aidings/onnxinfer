import cv2
# cv2.setNumThreads(0)
import os
import requests
from io import BytesIO
import numpy as np
from PIL import Image

try:
    from turbojpeg import TurboJPEG
    TBJPEG_FLAG = True
except ImportError:
    TBJPEG_FLAG = False
    # print('if you use torbojpeg, please get some more information: https://github.com/lilohuang/PyTurboJPEG.git')

__all__ = ["ImageReader"]


class ImageReader(object):
    """视觉资源读取

        Args:
            read_type (str, optional): 读取类型，支持pillow|opencv. Defaults to 'pillow'.
        Raises:
            ValueError: 不支持上述两种类型
            ImportError: 缺少PyTurboJPEG库或以来库
    """
    IMAGE_PILLOW = 'pillow'
    IMAGE_OPENCV = 'opencv'
    IMAGE_TBJPEG = 'turbojpeg'

    def __init__(self, ftype='pillow', *args, **kwargs):
        self.ftype = ftype
        if ftype == 'turbojpeg':
            if not TBJPEG_FLAG:
                raise ImportError("please pip PyTurboJPEG library before use it, get some more information: https://github.com/lilohuang/PyTurboJPEG.git") 
            else:
                lib_path = kwargs.get('lib_path', None)
                self.jpeg = TurboJPEG(lib_path=lib_path)
                if 'lib_path' in kwargs:
                    kwargs.pop('lib_path')
                # pixel_format = kwargs.get('pixel_format', None)
                # if pixel_format is None:
                #     kwargs['pixel_format'] = TJPF_RGB
                self.kwargs = kwargs

        self.__decodes = {
            "pillow": self.__read_impil,
            "opencv": self.__read_imocv,
            'turbojpeg': self.__read_imjpg
        }

        try:
            self.__decode = self.__decodes[self.ftype]
        except KeyError:
            raise ValueError("invalid read type('pillow, opencv'): {}".format(self.ftype))

    def decode(self, inp_buf):
        return self.__decode(inp_buf)

    def image_decode(self, inp_buf):
        return self.__decode(inp_buf)

    @staticmethod
    def __read_impil(inp_buf):
        if isinstance(inp_buf, bytes):
            img = np.asarray(bytearray(inp_buf), dtype='uint8')
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(inp_buf, str) and os.path.exists(inp_buf):
            image = Image.open(inp_buf)
            image = image.convert('RGB')
        elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
            resp = requests.get(inp_buf)
            image = Image.open(BytesIO(resp.content))
            image = image.convert('RGB')
        elif isinstance(inp_buf, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(inp_buf, cv2.COLOR_BGR2RGB))
        elif isinstance(inp_buf, Image.Image):
            image = inp_buf
        else:
            raise ValueError("Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)")
        
        return image

    @staticmethod
    def __read_imocv(inp_buf):
        if isinstance(inp_buf, bytes):
            img = np.asarray(bytearray(inp_buf), dtype='uint8')
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and os.path.isfile(inp_buf):
            image = cv2.imread(inp_buf, cv2.IMREAD_COLOR)
        elif isinstance(inp_buf, str) and 'http' == inp_buf.strip()[:4]:
            resp = requests.get(inp_buf)
            image = cv2.imdecode(np.fromstring(resp.content, np.unit8), 1)
        elif isinstance(inp_buf, np.ndarray):
            image = inp_buf
        elif isinstance(inp_buf, Image.Image):
            image = cv2.cvtColor(np.asarray(inp_buf), cv2.COLOR_RGB2BGR)
        else:
            image = None
            raise ValueError('Error: Not support this type image buffer(byte, str, np.ndarry, PIL.Image)')

        return image
    
    def __read_imjpg(self, jpg_buf):
        jpg_ext = ('.jpg', '.jpeg', '.JPG', '.JPEG')
        if isinstance(jpg_buf, bytes):
            image = self.jpeg.decode(jpg_buf, **self.kwargs)
        elif isinstance(jpg_buf, str) and jpg_buf.endswith(jpg_ext) and os.path.isfile(jpg_buf):
            jfile = open(jpg_buf, 'rb')
            image = self.jpeg.decode(jfile.read(), **self.kwargs)
        elif isinstance(jpg_buf, np.ndarray):
            image = jpg_buf
        elif isinstance(jpg_buf, Image.Image):
            image = np.asarray(jpg_buf)
        else:
            image = None
            raise ValueError('Error: Not support this type image buffer(byte, str(.jpeg, .jpg), np.ndarry, PIL.Image')
        
        return image