# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from io import StringIO
import io
# import PIL.Image as Image
class ZipReader(object):
    def __init__(self, flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION, oss_bucket=False):
        self.zip_readers = {}
        self.flags = flags
        self.oss_bucket = oss_bucket

    def get_reader(self, zip_root):
        if zip_root in self.zip_readers:
            return self.zip_readers[zip_root]
        else:
            if not os.path.isfile(zip_root):
                print("zip file '%s' is not found" % (zip_root))
                assert 0
            self.zip_readers[zip_root] = zipfile.ZipFile(zip_root, 'r')
            return self.zip_readers[zip_root]

    def read(self, path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" % (path))
            assert 0
        path_root = path[:pos_at]
        path_file = path[pos_at + 1:]
        zip_reader = self.get_reader(path_root)
        with zip_reader.open(path_file, 'r') as f:
            data = f.read()
        return data

    def imread(self, path):
        data = self.read(path)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), self.flags)
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # return Image.open(io.BytesIO(data)).convert('RGB')
        return img

    def xmlread(self, path):
        data = self.read(path)
        return ET.fromstring(data)

    def txtread(self, path):
        data = self.read(path)
        string_io = StringIO(str(data, encoding="utf-8"))
        lines = string_io.readlines()
        return lines

    def zip_lines_read(self, path):
        result_lines = []
        lines = self.txtread(path)
        for line in lines:
            line = line.rstrip('\r\n').rstrip('\n').strip(' ')
            result_lines.append(line)
        return result_lines

    def close_all(self):
        for zip_root in self.zip_readers:
            self.zip_readers[zip_root].close()

    def close_one(self, path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" % (path))
            assert 0
        zip_root = path[:pos_at]
        if zip_root in self.zip_readers:
            self.zip_readers[zip_root].close()

    def list_file_in_zip(self, path_root, prefix=None, suffix=None):
        zip_reader = self.get_reader(path_root)
        info_list = zip_reader.infolist()
        file_list = []
        for info in info_list:
            if (not prefix is None) and (not info.filename.startswith(prefix)):
                continue

            if (not suffix is None) and (not info.filename.endswith(suffix)):
                continue
            file_list.append(info.filename)
        return file_list




