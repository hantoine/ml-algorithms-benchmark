
from utils.tuning import save_tuning_results
from os.path import isdir, join as joinpath
from shutil import rmtree
import json

workdir = 'test-workdir'
if isdir(workdir):
    rmtree(workdir)

def check_written_tuning_results(hp, score, n_trials, tuning_time):
    with open(joinpath(workdir, 'tuning.json'), 'r') as file:
        written_data = json.load(file)
    assert written_data['hp'] == hp
    assert  written_data['score'] == score
    assert written_data['n_trials'] == n_trials
    assert written_data['tuning_time'] == tuning_time

def test_save_tuning_results():
    hp = {'a': 1, 'b': 2}
    score = 0.5
    n_trials = 15
    tuning_time = 1.5

    # Check saves results
    save_tuning_results(workdir, hp, score, n_trials, tuning_time)
    check_written_tuning_results(hp, score, n_trials, tuning_time)

    # Check does not save results if lower score
    lower_score = 0.3
    save_tuning_results(workdir, hp, lower_score, n_trials, tuning_time)
    check_written_tuning_results(hp, score, n_trials, tuning_time)

    # Check overwrite results if better score
    better_score = 0.7
    hp['a'] = 3
    save_tuning_results(workdir, hp, better_score, n_trials, tuning_time)
    check_written_tuning_results(hp, better_score, n_trials, tuning_time)