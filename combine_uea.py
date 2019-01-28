import os
import json
import torch
import sklearn
import argparse
import sklearn.ensemble

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
                    'them using a voting classifier.'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the folders containing models for ' +
                             'different hyperparameters are located')
    parser.add_argument('--folders', type=str, metavar='FOLDERS',
                        required=True, nargs='+',
                        help='list of folders, each one containing a model ' +
                             'for the chosen dataset')
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
            os.path.join(args.save_path, folder), args.dataset, args.cuda,
            args.gpu
        ) for folder in args.folders
    ]

    # Voting classifier
    classifier = sklearn.ensemble.VotingClassifier(classifiers)
    # In ordeer to not refit all models, tweak internal parameters
    classifier.estimators_ = classifiers
    classifier.le_ = sklearn.preprocessing.LabelEncoder().fit(train_labels)

    # Testing
    print("Test accuracy: " + str(classifier.score(test, test_labels)))
