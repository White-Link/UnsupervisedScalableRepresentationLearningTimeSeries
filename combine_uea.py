# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import os
import json
import numpy
import torch
import sklearn
import argparse

import uea
import scikit_wrappers


def load_classifier(save_path, dataset, cuda, gpu):
    """
    Loads and returns classifier from the given parameters.

    @param save_path Path where the model is located.
    @param dataset Name of the dataset.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()
    hf = open(
        os.path.join(
            save_path,
            dataset + '_hyperparameters.json'
        ), 'r'
    )
    hp_dict = json.load(hf)
    hf.close()
    hp_dict['cuda'] = cuda
    hp_dict['gpu'] = gpu
    classifier.set_params(**hp_dict)
    classifier.load(os.path.join(save_path, dataset))
    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets, ' +
                    'using the features of several precomputed encoders, ' +
                    'possibly with different hyperparameters, and combining ' +
                    'their computed representations to train an SVM on top ' +
                    'of them.'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--model_path', type=str, metavar='PATH',
                        required=True,
                        help='path where the folders containing models for ' +
                             'different hyperparameters are located')
    parser.add_argument('--folders', type=str, metavar='FOLDERS',
                        required=True, nargs='+',
                        help='list of folders, each one containing a model ' +
                             'for the chosen dataset')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the classifier is/should be saved')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the classifier instead of ' +
                             'training it')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    # Train, test datasets
    train, train_labels, test, test_labels = uea.load_UEA_dataset(
        args.path, args.dataset
    )

    # List of classifiers
    classifiers = [
        load_classifier(
            os.path.join(args.model_path, folder), args.dataset, args.cuda,
            args.gpu
        ) for folder in args.folders
    ]

    train_representations = numpy.concatenate([
        c.encode(train) for c in classifiers
    ], axis=1)
    test_representations = numpy.concatenate([
        c.encode(test) for c in classifiers
    ], axis=1)

    classifier = sklearn.svm.SVC(C=numpy.inf, gamma='scale')

    if not args.load:
        nb_classes = numpy.shape(
            numpy.unique(train_labels, return_counts=True)[1]
        )[0]
        train_size = numpy.shape(train_representations)[0]
        if train_size // nb_classes < 5 or train_size < 50:
            classifier.fit(train_representations, train_labels)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                classifier, {
                    'C': [
                        0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                        numpy.inf
                    ],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [None]
                },
                cv=5, iid=False, n_jobs=5
            )
            if train_size <= 10000:
                grid_search.fit(train_representations, train_labels)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = sklearn.model_selection.train_test_split(
                    train_representations, train_labels,
                    train_size=10000, random_state=0, stratify=train_labels
                )
                grid_search.fit(split[0], split[2])
            classifier = grid_search.best_estimator_
        sklearn.externals.joblib.dump(
            classifier, os.path.join(
                args.save_path, args.dataset + '_CausalCNN_classifier.pkl'
            )
        )
    else:
        classifier = sklearn.externals.joblib.load(os.path.join(
            args.save_path, args.dataset + '_CausalCNN_classifier.pkl'
        ))

    # Testing
    print("Test accuracy: " + str(
        classifier.score(test_representations, test_labels)
    ))
