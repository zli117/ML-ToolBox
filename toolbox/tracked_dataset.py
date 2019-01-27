from abc import ABC

from torch.utils.data import Dataset

from toolbox.trackable import Trackable


class TrackedDataset(Dataset, Trackable, ABC):
    """
    Implement this class to have a trackable dataset
    """
