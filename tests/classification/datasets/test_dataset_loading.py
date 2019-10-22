import classification.datasets as ds

def test_default_credit_card_dataset_loading():
    ds.DefaultCreditCardDataset.get()
