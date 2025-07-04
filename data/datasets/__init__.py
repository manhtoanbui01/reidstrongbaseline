# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .night600 import Night600
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset
from .nightreid import NightReID
from .nightreid_plus_market1501 import NightReID_Market1501
from .night600_plus_market1501 import Night600_Market1501
from .nightreid_plus_tiktok import NightReID_TikTok

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'night600': Night600,
    'nightreid': NightReID,
    'nightreid_market1501': NightReID_Market1501,
    'night600_market1501': Night600_Market1501,
    'nightreid_tiktok': NightReID_TikTok
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
