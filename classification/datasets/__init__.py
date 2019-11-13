from .loaders import DefaultCreditCardDataset
from .loaders import AdultDataset
from .loaders import YeastDataset
from .loaders import SeismicBumpsDataset
from .loaders import StatlogAustralianDataset
from .loaders import StatlogGermanDataset
from .loaders import SteelPlatesFaultsDataset
from .loaders import RetinopathyDataset
from .loaders import BreastCancerDataset
from .loaders import ThoraricSurgeryDataset

from .utils import get_min_k_fold_k_value

all_datasets = [StatlogAustralianDataset, StatlogGermanDataset, SteelPlatesFaultsDataset,
                ThoraricSurgeryDataset, YeastDataset, SeismicBumpsDataset,
                RetinopathyDataset, AdultDataset, DefaultCreditCardDataset,
                BreastCancerDataset]