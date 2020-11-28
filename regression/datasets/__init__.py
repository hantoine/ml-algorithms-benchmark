from .loaders import *

all_datasets = [
    WhiteWineQualityDataset,
    RedWineQualityDataset,
    CommunitiesAndCrimeDataset,
    QsarAquaticToxicityDataset,
    ParkinsonMultipleSoundRecordingDataset,
    FacebookLikesDataset,
    FacebookShareDataset,
    FacebookCommentDataset,
    ConcreteCompressiveStrengthDataset,
    StudentMathPerformanceDataset,
    StudentPortuguesePerformanceDataset,
    StudentMathPerformanceNoPrevGradesDataset,
    StudentPortuguesePerformanceNoPrevGradesDataset,
    # This ones were tuned for longer (3000s)
    MerckMolecularActivityDataset,
    BikeSharingDataset,
    SGEMMGPUKernelPerformancesDataset,
]
