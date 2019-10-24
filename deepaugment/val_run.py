import numpy as np

import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, parent_dir_of_file)

from run_full_model import run_full_model
from build_features import DataOp


def main():

    X, y, input_shape = DataOp.load("cifar10")

    run_full_model(
        X, y, test_proportion=0.1,
        model="wrn_28_10", epochs=100, batch_size=32,
        policies_path="/home/acb11354uz/B4researchMain/B4ResearchDeepaugment/reports/best_policies/top20_policies_cifar10_exp_2019-02-08_03-54_3000_iters.csv"
    )


if __name__ == "__main__":
    main()
