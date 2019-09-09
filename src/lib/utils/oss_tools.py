import torch
import tempfile
import oss2
import pickle

class IterDataloader(object):
    dataloader = None
    dataloader_iterator = None
    @classmethod
    def set(cls, dataloader):
        cls.dataloader = dataloader
        cls.dataloader_iterator = iter(cls.dataloader)

    @classmethod
    def next(cls):
        try:
            img_hard, target_hard = next(cls.dataloader_iterator)
        except StopIteration:
            cls.dataloader_iterator = iter(cls.dataloader)
            img_hard, target_hard = next(cls.dataloader_iterator)
        return img_hard, target_hard


class OSS_Bucket(object):
    oss_bucket = False
    bucket = None

    @classmethod
    def set(cls, flag):
        cls.oss_bucket = flag

def torch_load(path):
    if OSS_Bucket.oss_bucket:
        tmp = tempfile.NamedTemporaryFile()
        try:
            OSS_Bucket.bucket.get_object_to_file(path, tmp.name)
        except Exception as e:
            print(e)
        object = torch.load(tmp, map_location='cpu' if not torch.cuda.is_available() else None)
        tmp.close()
    else:
        object = torch.load(path, map_location='cpu' if not torch.cuda.is_available() else None)
    return object

def torch_save(obj, path):
    if OSS_Bucket.oss_bucket:
        tmp = tempfile.NamedTemporaryFile()
        torch.save(obj, tmp)
        try:
            OSS_Bucket.bucket.put_object_from_file(path, tmp.name)
        except Exception as e:
            print(e)
        tmp.close()
    else:
        torch.save(obj, path)

def pickle_load(path):
    if OSS_Bucket.oss_bucket:
        tmp = tempfile.NamedTemporaryFile()
        try:
            OSS_Bucket.bucket.get_object_to_file(path, tmp.name)
        except Exception as e:
            print(e)
        object = pickle.load(tmp)
        tmp.close()
    else:
        with open(path, 'rb') as f:
            object = pickle.load(f)
    return object

def pickle_dump(obj, path):
    if OSS_Bucket.oss_bucket:
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'wb') as f:
            pickle.dump(obj, f)
        try:
            OSS_Bucket.bucket.put_object_from_file(path, tmp.name)
        except Exception as e:
            print(e)
        tmp.close()
    else:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

if __name__ == '__main__':
    print(OSS_Bucket.oss_bucket)
    OSS_Bucket.set(True)
    print(OSS_Bucket.oss_bucket)