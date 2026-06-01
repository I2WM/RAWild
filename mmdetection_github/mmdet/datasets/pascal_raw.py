# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset
import os.path as osp
from typing import List
from mmengine.fileio import list_from_file


@DATASETS.register_module()
class PASCAL_RAW(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'classes':
        ('car', 'person', 'bicycle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42)]
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
class PASCAL_RAWDataset(XMLDataset):
    """Dark-ISP style PASCAL RAW dataset using .npz RAW files."""

    METAINFO = {
        'classes': ('car', 'person', 'bicycle'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2007'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['dataset_type'] = 'VOC2012'
        else:
            self._metainfo['dataset_type'] = None

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
