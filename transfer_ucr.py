import os
import json
import torch
import argparse

import ucr
import scikit_wrappers


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Uses the learned representations for a dataset to ' +
                    'learn classifiers for all other UCR datasets'
    )
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the UCR datasets are located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the encoder is saved')
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
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

    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    hf = open(
        os.path.join(args.save_path, args.dataset + '_hyperparameters.json'),
        'r'
    )
    hp_dict = json.load(hf)
    hf.close()
    hp_dict['cuda'] = args.cuda
    hp_dict['gpu'] = args.gpu
    classifier.set_params(**hp_dict)
    classifier.load(os.path.join(args.save_path, args.dataset))

    print("Classification tasks...")

    # List of folders / datasets in the given path
    datasets = [x[0][len(args.path) + 1:] for x in os.walk(args.path)][1:]
    for dataset in datasets:
        train, train_labels, test, test_labels = ucr.load_UCR_dataset(
            args.path, dataset
        )
        classifier.fit_classifier(classifier.encode(train), train_labels)
        print(
            dataset,
            "Test accuracy: " + str(classifier.score(test, test_labels))
        )
