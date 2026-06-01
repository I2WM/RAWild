# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset
from typing import List, Union
import os.path as osp
import xml.etree.ElementTree as ET
import mmcv
from mmengine.fileio import get, get_local_path, list_from_file


@DATASETS.register_module()
class LOD_Dataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('bicycle', 'car', 'motorbike', 'chair', 'diningtable', 'bottle', 'tvmonitor', 'bus'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None


@DATASETS.register_module()
class LOD_RAWDAtaset(XMLDataset):
    """Dark-ISP LOD RAW dataset using .npz RAW files."""

    METAINFO = LOD_Dataset.METAINFO

    def __init__(self, clean_subdir: str = None, **kwargs):
        super().__init__(**kwargs)
        self.clean_subdir = clean_subdir

    def load_data_list(self) -> List[dict]:
        assert self._metainfo.get('classes', None) is not None, \
            '`classes` in `XMLDataset` can not be None.'
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo['classes'])}
        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f'{img_id}.npz')
            xml_path = osp.join(self.sub_data_root, self.ann_subdir, f'{img_id}.xml')
            raw_img_info = dict(img_id=img_id, file_name=file_name, xml_path=xml_path)
            data_list.append(self.parse_data_info(raw_img_info))
        return data_list

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        data_info = {}
        img_path = osp.join(self.sub_data_root, img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']
        with get_local_path(img_info['xml_path'], backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            height, width = img.shape[:2]
            del img, img_bytes
        data_info['height'] = height
        data_info['width'] = width
        data_info['instances'] = self._parse_instance_info(raw_ann_info, minus_one=True)
        return data_info
