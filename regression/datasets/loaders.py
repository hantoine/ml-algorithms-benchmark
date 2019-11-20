import os
from os.path import isfile, isdir
import urllib.parse
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import patoolib  # for rar files
from zipfile import ZipFile
import time
from scipy.io import arff
import shutil
import kaggle
import zipfile

from utils import Dataset, encode_feature_as_cyclical
from config import TEST_SIZE, RANDOM_STATE
from config import DEFAULT_DATA_DIR


class MerckMolecularActivityDataset(Dataset):
    filename = 'TrainingSet.zip'
    metric = 'r2'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        if not isdir(os.path.join(workdir, 'TrainingSet')):
            cls.prepare(workdir)
        df_1 = pd.read_csv(os.path.join(workdir, 'TrainingSet/ACT2_competition_training.csv'))
        df_2 = pd.read_csv(os.path.join(workdir, 'TrainingSet/ACT4_competition_training.csv'))
        df = pd.concat((df_1, df_2), axis=0, sort=True)
        X = df[df.columns[1:-1]]
        y = df['Act']

        # Remove columns with more NaNs than values (still more than 5k columns left)
        X = X.loc[:, X.count() > 0.5 * len(X)]

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Replace NaNs by mean (Interactive Imputer is extremely slow with a datasets of this size)
        imp = SimpleImputer()
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        return (X_train, y_train), (X_test, y_test)

    @classmethod
    def prepare(cls, workdir):
        kaggle.api.competition_download_file(competition='MerckActivity',
                                             file_name=cls.filename,
                                             path=workdir,
                                             quiet=True)

        with zipfile.ZipFile(os.path.join(workdir, cls.filename), 'r') as zip_file:
            zip_file.extract('TrainingSet/ACT2_competition_training.csv', path=workdir)
            zip_file.extract('TrainingSet/ACT4_competition_training.csv', path=workdir)


class WhiteWineQualityDataset(Dataset):
    filename = 'winequality-red.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=';')
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class RedWineQualityDataset(Dataset):
    filename = 'winequality-white.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=';')
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class CommunitiesAndCrimeDataset(Dataset):
    filename = 'communities.data'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=',', header=None)
        X = df[df.columns[5:-1]]  # First 5 columns are not predictive according to the datasets description
        y = df[df.columns[-1]]

        X = X.apply(partial(pd.to_numeric, errors='coerce'))  # replace '?' by NaN

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class QsarAquaticToxicityDataset(Dataset):
    filename = 'qsar_aquatic_toxicity.csv'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00505//qsar_aquatic_toxicity.csv'
    features = ['TPSA', 'SAacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040', 'LC50']
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_csv(dataset_path, sep=';')
        df.columns = cls.features
        y = df[df.columns[-1]]
        X = df[df.columns[:-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        return (X_train, y_train), (X_test, y_test)


class ParkinsonMultipleSoundRecordingDataset(Dataset):
    filename = 'Parkinson_Multiple_Sound_Recording.rar'
    filenames = ['train_data.txt',
                 'test_data.txt']
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00301/Parkinson_Multiple_Sound_Recording.rar'

    column_names = ['subject_id',
                    'jitter_local', 'jitter_local_abs', 'jitter_rap',
                    'jitter_ppq5', 'jitter_ddp', 'shimmer_local',
                    'shimmer_local_db', 'shimmer_apq3', 'shimmer_apq5',
                    'shimmer_apq11', 'shimmer_dda', 'AC', 'NTH', 'HTN',
                    'median_pitch', 'mean_pitch', 'std', 'minimum_pitch',
                    'maximum_pitch', 'nb_pulses', 'nb_periods', 'mean_period',
                    'std', 'frac_local_unvoiced_frames', 'nb_voice_breaks',
                    'degree_voice_breaks', 'UPDRS', 'class']
    metric = '-rmse'
    categorical_features = []
    need_grouped_split = True

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)

        extracted_path = os.path.join(workdir, 'parkinson_dataset')
        if isdir(extracted_path):  # patoollib does not support overwriting
            shutil.rmtree(extracted_path)
        os.mkdir(extracted_path)
        patoolib.extract_archive(dataset_path, outdir=extracted_path, verbosity=-1)
        time.sleep(2)

        """
            This datasets is initially a classification datasets whose goal is to predict for 
            each example if the patient is healthy or sick.
            It can be used for regression. In that case the goal is to predict the UPDRS (Unified
            Parkinson's Disease Rating Scale) score. However, this score is only given for examples
            in the training set. We will therefore use the training set for bothe training and testing.
            To evaluate the performance of the model on new subject (patients), we will ensure the test
            set contains only new subjects.
        """
        df = pd.read_csv(os.path.join(extracted_path, cls.filenames[0]), sep=',', header=None)
        df.columns = cls.column_names
        groups = df['subject_id']
        X = df[df.columns[1:-2]]
        y = df['UPDRS']

        splitter = GroupShuffleSplit(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        return (X_train, y_train, groups_train), (X_test, y_test)


class FacebookMetricsDataset(Dataset):
    filename = 'Facebook_metrics.zip'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip'
    metric = '-rmse'
    categorical_features = ['Type', 'Category', 'Post Weekday']
    """
        We decide to encode Post Hour and Post Month as cyclical numeric.

        Would be possible to keep them as is or encode them as categories, it would be
        intersting to implement datasets specific hyperparameters for this kind of thing
    """

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with ZipFile(dataset_path, 'r') as zipfile:
            zipfile.extractall(workdir)
        df = pd.read_csv(os.path.join(workdir, 'dataset_Facebook.csv'), sep=';')
        X = df[df.columns[:7]]
        X = encode_feature_as_cyclical(X, 'Post Month', 12)
        X = encode_feature_as_cyclical(X, 'Post Hour', 24)

        y = df[df.columns[7:]]
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def select_label_col(data, name):
        X, y = data
        y = y[name]
        return X[~y.isna()], y[~y.isna()]


class FacebookLikesDataset(FacebookMetricsDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        train, test = super(FacebookLikesDataset, cls).get(workdir)
        select_likes = partial(FacebookMetricsDataset.select_label_col, name='like')
        return select_likes(train), select_likes(test)


class FacebookInteractionsDataset(FacebookMetricsDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        train, test = super(FacebookInteractionsDataset, cls).get(workdir)
        select_interactions = partial(FacebookMetricsDataset.select_label_col,
                                      name='Total Interactions')
        return select_interactions(train), select_interactions(test)


class FacebookShareDataset(FacebookMetricsDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        train, test = super(FacebookShareDataset, cls).get(workdir)
        select_shares = partial(FacebookMetricsDataset.select_label_col, name='share')
        return select_shares(train), select_shares(test)


class FacebookCommentDataset(FacebookMetricsDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        train, test = super(FacebookCommentDataset, cls).get(workdir)
        select_comments = partial(FacebookMetricsDataset.select_label_col, name='comment')
        return select_comments(train), select_comments(test)


class BikeSharingDataset(Dataset):
    filename = 'Bike-Sharing-Dataset.zip'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
    metric = '-rmse'
    categorical_features = ['season', 'weekday']

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with ZipFile(dataset_path, 'r') as zipfile:
            zipfile.extractall(workdir)
        df = pd.read_csv(os.path.join(workdir, 'hour.csv'), sep=',')
        # The first column is an id and the second the date
        # The 3 last columns are labels
        X = df[df.columns[2:-3]]
        y = df[df.columns[-1]]
        X = encode_feature_as_cyclical(X, 'hr', 24)
        X = encode_feature_as_cyclical(X, 'mnth', 12)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class StudentPerformanceBaseDataset(Dataset):
    filename = 'student.zip'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
    metric = '-rmse'
    categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

    @classmethod
    def get(cls, subject, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with ZipFile(dataset_path, 'r') as zipfile:
            zipfile.extractall(workdir)
        df = pd.read_csv(os.path.join(workdir, f'student-{subject}.csv'), sep=';')

        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        # Encode yes/no categories as int
        X = X.replace('yes', 1)
        X = X.replace('no', 0)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class StudentPerformanceNoPrevGradesBaseDataset(StudentPerformanceBaseDataset):
    @classmethod
    def get(cls, subject, workdir=DEFAULT_DATA_DIR):
        train, test = super(StudentPerformanceNoPrevGradesBaseDataset, cls).get(subject, workdir)
        train = cls.remove_previous_grades(train)
        test = cls.remove_previous_grades(test)
        return train, test

    @staticmethod
    def remove_previous_grades(data):
        X, y = data
        X = X.drop('G1', axis='columns')
        X = X.drop('G2', axis='columns')
        return X, y


class StudentMathPerformanceDataset(StudentPerformanceBaseDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        return super(StudentMathPerformanceDataset, cls).get('mat', workdir)


class StudentPortuguesePerformanceDataset(StudentPerformanceBaseDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        return super(StudentPortuguesePerformanceDataset, cls).get('por', workdir)


class StudentMathPerformanceNoPrevGradesDataset(StudentPerformanceNoPrevGradesBaseDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        return super(StudentMathPerformanceNoPrevGradesDataset, cls).get('mat', workdir)


class StudentPortuguesePerformanceNoPrevGradesDataset(StudentPerformanceNoPrevGradesBaseDataset):
    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        return super(StudentPortuguesePerformanceNoPrevGradesDataset, cls).get('por', workdir)


class ConcreteCompressiveStrengthDataset(Dataset):
    filename = 'Concrete_Data.xls'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        df = pd.read_excel(dataset_path)

        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)


class SGEMMGPUKernelPerformancesDataset(Dataset):
    filename = 'sgemm_product_dataset.zip'
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip'
    metric = '-rmse'
    categorical_features = []

    @classmethod
    def get(cls, workdir=DEFAULT_DATA_DIR):
        dataset_path = os.path.join(workdir, cls.filename)
        if not isfile(dataset_path):
            cls.download(workdir)
        with ZipFile(dataset_path, 'r') as zipfile:
            zipfile.extractall(workdir)
        df = pd.read_csv(os.path.join(workdir, 'sgemm_product.csv'), sep=',')
        X = df[df.columns[:14]]
        y = df[df.columns[14:]].mean(axis=1)

        # All columns but the last 4 only contain powers of 2
        X_log_2 = {col: np.log2(X[col]) for col in X.columns[:-4]}
        X = X.assign(**X_log_2)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return (X_train, y_train), (X_test, y_test)
