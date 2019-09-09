# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import oss2
import sys
from utils.oss_tools import OSS_Bucket

def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if OSS_Bucket.oss_bucket:
            fh = OSSLoggingHandler(os.path.join(save_dir, "log.txt"))
            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s\n")
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class OSSLoggingHandler(logging.StreamHandler):
    def __init__(self, log_file):
        super(OSSLoggingHandler, self).__init__()
        self._bucket = OSS_Bucket.bucket
        self._log_file = log_file
        if self._bucket.object_exists(self._log_file):
            self._bucket.delete_object(self._log_file)
            # raise ValueError('log file {} exists, Please check!'.format(self._log_file))
        self._pos = self._bucket.append_object(self._log_file, 0, '')

    def emit(self, record):
        msg = self.format(record)
        try:
            self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)
        except Exception as e:
            print(e)
            if isinstance(e, oss2.exceptions.PositionNotEqualToLength):
                raise ValueError('log file [{}] has changed, Please check!'.format(self._log_file))


class RedirectStdout(object):
    def __init__(self, save_dir):  # 保存标准输出流
        self.fname = os.path.join(save_dir, 'stdout.txt')
        self.file = None
        self.out = sys.stdout
        self.err = sys.stderr

    def start(self):  # 标准输出重定向至文件
        self.file = open(self.fname, 'w', encoding='utf-8')
        sys.stderr = self.file
        sys.stdout = self.file

    def end(self):  # 恢复标准输出流
        if self.file:
            sys.stdout = self.out
            sys.stderr = self.err
            self.file.close()

if __name__ == '__main__':
    OSS_Bucket.set(True)
    logger = setup_logger('test_logger', 'niding-nd', 0)
    for i in range(100):
        logger.info(','.join(['CCC']*i))