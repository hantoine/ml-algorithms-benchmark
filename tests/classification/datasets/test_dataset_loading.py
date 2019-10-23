import classification.datasets as ds
from shutil import rmtree
from os.path import isdir

ds.workdir = 'test-workdir'
if isdir(ds.workdir):
    rmtree(ds.workdir)

def test_default_credit_card_dataset_loading():
    ds.DefaultCreditCardDataset.get()
