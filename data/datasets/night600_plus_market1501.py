import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class Night600_Market1501(BaseImageDataset):
    dataset_dir = "night600_plus_market1501"
    def __init__(self, root="/mnt/disk1/data", verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        
        self._check_before_run()
        
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        if verbose:
            print("=> Night600 + Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)
            
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        img_paths += glob.glob(osp.join(dir_path, "*.png"))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_container = set()
        for img_path in img_paths:
            if img_path[-7]=="_": # image in the market1501 dataset, we add 2000 to pid so that id of the market1501 is different with the one in night600
                pid, _ = map(int, pattern.search(img_path).groups())
                pid+=2000
            else:    
                pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            if img_path[-7]=="_":
                pid, camid = map(int, pattern.search(img_path).groups())
                pid+=2000
                camid+=10
            else:
                pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            camid -= 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            
        return dataset