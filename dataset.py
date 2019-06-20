from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
import glob
import os
import h5py

import numpy as np

from PIL import Image
from torch import Tensor
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN

        # TODO: CODE BEGIN
        # raise NotImplementedError
        self._path_to_data = path_to_data_dir
        self._mode = mode
        self._length = len(glob.glob(os.path.join(self._path_to_data, self._mode.value, '*.png')))
        # if is_train:
        #     self._length += len(glob.glob(os.path.join(self._path_to_data, 'extra/*')))
        # TODO: CODE END

    def __len__(self) -> int:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        return self._length
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        path_to_mat = os.path.join(self._path_to_data, self._mode.value, 'digitStruct.mat')
        _h5py_file = h5py.File(path_to_mat)
        _name_ref = _h5py_file.get('digitStruct').get('name')[index][0]
        _obj_name = _h5py_file.get('digitStruct').get(_name_ref)
        _image_filename = ''.join(chr(i) for i in _obj_name[:])
        _path_to_image = os.path.join(self._path_to_data, self._mode.value, _image_filename)

        # 29930.png has some unknown problem.
        if _path_to_image == './data/train/29930.png':
            new_index = index + 1
            return self.__getitem__(new_index)

        map_of_bbox = {}
        item = _h5py_file['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = _h5py_file[item][key]
            values = [_h5py_file[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [
                attr.value[0][0]]
            map_of_bbox[key] = values

        _left, _top, _right, _bottom = self.get_bounding_box(map_of_bbox)
        image = Image.open(_path_to_image)
        image = image.crop((_left, _top, _right, _bottom))
        image = image.resize([64, 64])
        image = self.preprocess(image)
        image = image.view(3, 54, 54)

        length = len(map_of_bbox['label'])

        digits = [10, 10, 10, 10, 10, 10]
        for idx in range(length):
            digits[idx] = map_of_bbox['label'][idx]
            if digits[idx] == 10:
                digits[idx] = 0

        return image, length, digits
        # TODO: CODE END

    @staticmethod
    def get_bounding_box(information: dict) -> Tuple[int, int, int, int]:
        """
        :param information: annotations of each digit
        :return: cropped range for images
        """
        bbox_left = int(np.min(information['left']))
        bbox_top = int(np.min(information['top']))
        bbox_right = int(
            np.max(information['left']) + [information['width'][index] for index in range(len(information['left']))
                                           if information['left'][index] == np.max(information['left'])][0])
        bbox_bottom = int(np.max(
            [information['top'][index] + information['height'][index] for index in range(len(information['top']))]))
        bbox_width = bbox_right - bbox_left
        bbox_height = bbox_bottom - bbox_top

        cropped_left = int(round(bbox_left - bbox_width * 0.15))
        cropped_top = int(round(bbox_top - bbox_height * 0.15))
        cropped_width = int(round(bbox_width * 1.3))
        cropped_height = int(round(bbox_height * 1.3))
        cropped_right = cropped_left + cropped_width
        cropped_bottom = cropped_top + cropped_height

        return cropped_left, cropped_top, cropped_right, cropped_bottom

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        transform = transforms.Compose([
            transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image)
        # TODO: CODE END


if __name__ == '__main__':
    _dataset = Dataset(path_to_data_dir='./data', mode=Dataset.Mode.TEST)
    _image, _length, _digits = _dataset[1111]
    # print(f'dataset length: {len(_dataset)}')
    # print(f'image type: {type(_image)}')
    # print(f'length type: {type(_length)}')
    # print(f'digits type: {type(_digits)}')
    # print(f'image shape: {np.array(_image).shape}')
    # print(f'length shape: {np.array(_length).shape}')
    # print(f'digits shape: {np.array(_digits).shape}')

    print('dataset length: ',len(_dataset))
    print('image type: ',type(_image))
    print('length type: ',type(_length))
    print('digits type: ',type(_digits))
    print('image shape: ',np.array(_image).shape)
    print('length shape: ',np.array(_length).shape)
    print('digits shape: ',np.array(_digits).shape)

    print(_length)
    print(_digits)