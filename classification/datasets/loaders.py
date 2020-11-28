import os
from os.path import isfile
import urllib.parse
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.io import arff

from utils import Dataset
from config import TEST_SIZE, RANDOM_STATE
from config import DEFAULT_DATA_DIR


class RetinopathyDataset(Dataset):
    filename = "messidor_features.arff"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"

    feature_names = (
        ["quality", "pre-screening_label"]
        + list(range(2, 16))
        + ["dist_betw_centers", "od_diameter", "AM_FM_label", "class"]
    )
    categorical_features = []
    metric = "accuracy"
    is_metric_maximized = True

    @classmethod
    def preprocess(cls, df):
        df.columns = cls.feature_names
        df["class"] = pd.to_numeric(df["class"])
        df.drop(columns=["quality"])
        return df.drop(columns=["class"]), df.loc[:, "class"]

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        data, _ = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        X, y = cls.preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class DefaultCreditCardDataset(Dataset):
    filename = "default of credit card clients.xls"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls"
    categorical_features = []
    metric = "f1"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_excel(dataset_path, header=[0, 1])
        y = df["Y"][df["Y"].columns[0]]
        X = df[[f"X{i}" for i in range(1, 24)]]
        X.columns = X.columns.droplevel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class BreastCancerDataset(Dataset):
    filename = "breast-cancer-wisconsin.data"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    categorical_features = []
    metric = "f1"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, header=None)
        X = df[df.columns[1:-1]]
        X = X.apply(partial(pd.to_numeric, errors="coerce"))  # replace '?' by NaN
        y = (df[df.columns[-1]] == 4).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class StatlogAustralianDataset(Dataset):
    filename = "australian.dat"
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
    categorical_features = [0, 3, 4, 5, 7, 8, 10, 11]
    metric = "accuracy"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=" ", header=None)
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class StatlogGermanDataset(Dataset):
    filename = "german.data-numeric"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
    metric = "f1"
    is_metric_maximized = True
    categorical_features = []  # already encoded

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, delim_whitespace=True, header=None)
        y = (df[df.columns[-1]] == 2).astype(int)
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class SteelPlatesFaultsDataset(Dataset):
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults.NNA",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00198/Faults27x7_var",
    ]
    filenames = ["Faults.NNA", "Faults27x7_var"]
    categorical_features = []
    metric = "f1"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filenames[0])
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep="\t", header=None)

        dataset_path = os.path.join(workdir, cls.filenames[1])
        if not isfile(dataset_path):
            cls.download(workdir)
        with open(dataset_path, "r") as f:
            cls.feature_names = f.read().strip().split("\n")

        X = df[df.columns[:27]]
        y = df[df.columns[27:]]

        # Convert labels from onehot to ordinal
        y.columns = range(7)
        y = y.idxmax(axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class AdultDataset(Dataset):
    filenames = ["adult.data", "adult.test"]
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    ]
    feature_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    desc_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names"
    )
    metric = "f1"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        df_train, df_test = cls.get_raw(workdir)
        X_train, y_train = cls.parse_dataset(df_train)
        X_test, y_test = cls.parse_dataset(df_test)

        le = LabelEncoder().fit(y_train)
        y_train = le.transform(y_train)
        y_test = y_test.str[:-1]  # Additional . at the end of labels in test
        y_test = le.transform(y_test)

        return (X_train, y_train), (X_test, y_test)

    @classmethod
    def get_raw(cls, workdir):
        dataset_path = os.path.join(workdir, cls.filenames[0])
        if not isfile(dataset_path):
            cls.download(workdir)
        df_train = pd.read_csv(
            os.path.join(workdir, cls.filenames[0]),
            header=None,
            skiprows=1,
            sep=", ",
            engine="python",
        )
        df_test = pd.read_csv(
            os.path.join(workdir, cls.filenames[1]),
            header=None,
            skiprows=1,
            sep=", ",
            engine="python",
        )
        return df_train, df_test

    @classmethod
    def parse_dataset(cls, df):
        X, y = df[df.columns[:14]], df[df.columns[14]]
        X.columns = cls.feature_names
        return X, y


class YeastDataset(Dataset):
    filename = "yeast.data"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    metric = "f1"
    is_metric_maximized = True
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, delim_whitespace=True, header=None)
        df = df.drop_duplicates()

        X = df[df.columns[1:-1]]
        y = df[df.columns[-1]]
        y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class ThoraricSurgeryDataset(Dataset):
    filename = "ThoraricSurgery.arff"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff"
    metric = "f1"
    is_metric_maximized = True
    categorical_features = ["DGN", "PRE6"]

    @classmethod
    def preprocess(cls, df):
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        y = (y == b"T").astype(int)

        # Encode ordinal feature
        X["PRE14"] = X["PRE14"].str[-2:].astype(int)

        # Merge categories for which there is not enough examples
        # (1, 5, 6 and 8 with 1, 15, 4 and 2 examples respectively)
        X["DGN"].replace(b"DGN1", b"others", inplace=True)
        X["DGN"].replace(b"DGN5", b"others", inplace=True)
        X["DGN"].replace(b"DGN6", b"others", inplace=True)
        X["DGN"].replace(b"DGN8", b"others", inplace=True)

        # Encode binary categories as int
        X.replace(b"T", 1, inplace=True)
        X.replace(b"F", 0, inplace=True)

        return X, y

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        data, _ = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        X, y = cls.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)


class SeismicBumpsDataset(Dataset):
    filename = "seismic-bumps.arff"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
    categorical_features = ["seismic", "seismoacoustic", "shift", "ghazard"]
    metric = "f1"
    is_metric_maximized = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        data, _ = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        df["class"] = pd.to_numeric(df["class"])

        str_df = df.select_dtypes([np.object])
        str_df = str_df.stack().str.decode("utf-8").unstack()
        for col in str_df:
            df[col] = str_df[col]

        X = df.drop(columns=["class"])
        y = df.loc[:, "class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        return (X_train, y_train), (X_test, y_test)
