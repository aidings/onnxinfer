import cv2
import numpy as np
import numbers
from PIL import Image

class FuncBase(object):
    OPENCV = 'opencv'
    PILOW = 'pillow'
    def __init__(self, ftype='opencv', *args, **kwargs) -> None:
        self.ftype = ftype
    

class Normalize(FuncBase):
    def __init__(self, mean, std, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean = np.asarray(mean).reshape(-1, 1, 1).astype(np.float32)
        self.std = np.asarray(std).reshape(-1, 1, 1).astype(np.float32)
    
    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std

        return image


class Resize(FuncBase):
    def __init__(self, size, interpolation=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        if self.ftype == FuncBase.PILOW:
            self.interpolation = interpolation or Image.BILINEAR
            self.__func = self.__pillow
        else:
            self.interpolation = interpolation or cv2.INTER_LINEAR
            self.__func = self.__opencv

    def __pillow(self, image):
        image = image.resize(self.size, resample=self.interpolation)
        return image
    
    def __opencv(self, image):
        return cv2.resize(image, self.size, interpolation=self.interpolation)
    
    def __call__(self, image):
        return self.__func(image)


class ToTensor(FuncBase):
    """ return (img - mean) / std and image buffer to RGB type
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__func = self.__pillow if self.ftype == 'pillow' else self.__opencv
    
    def __pillow(self, image):
        image = np.asarray(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0
        return image

    def __opencv(self, image):
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0
        return image[:, :, ::-1]
        
    def __call__(self, image):
        return self.__func(image)


class CenterCrop(FuncBase):
    """
    Extract center crop of thevideo.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
    """
    def __init__(self, size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

        self.__func = self.__opencv if self.ftype == 'opencv' else self.__pillow
    
    def __opencv(self, image):
        crop_h, crop_w = self.size
        im_h, im_w, _ = image.shape

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)
        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        return image[h1:h1 + crop_h, w1:w1 + crop_w, :]
    
    def __pillow(self, image):
        crop_h, crop_w = self.size
        im_w, im_h = image.size

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        return image.crop((w1, h1, w1 + crop_w, h1 + crop_h))

    def __call__(self, image):
        return self.__func(image) 