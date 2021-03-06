from sklearn.naive_bayes import GaussianNB
from utils import NonTreeBasedModel


class GaussianNBModel(NonTreeBasedModel):
    @staticmethod
    def build_estimator(args, train_data=None):
        return GaussianNB()
