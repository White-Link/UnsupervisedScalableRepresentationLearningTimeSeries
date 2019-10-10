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


import math
import numpy
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection

import utils
import losses
import networks


class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    """
    "Virtual" class to wrap an encoder of time series as a PyTorch module and
    a SVM classifier with RBF kernel on top of its computed representations in
    a scikit-learn class.

    All inheriting classes should implement the get_params and set_params
    methods, as in the recommendations of scikit-learn.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        sklearn.externals.joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = sklearn.externals.joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_classifier(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an SVM
        classifier with RBF kernel.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        train_size = numpy.shape(features)[0]
        # To use a 1-NN classifier, no need for model selection, simply
        # replace the code by the following:
        # import sklearn.neighbors
        # self.classifier = sklearn.neighbors.KNeighborsClassifier(
        #     n_neighbors=1
        # )
        # return self.classifier.fit(features, y)
        self.classifier = sklearn.svm.SVC(
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else numpy.inf,
            gamma='scale'
        )
        if train_size // nb_classes < 5 or train_size < 50:
            return self.classifier.fit(features, y)
        else:
            if self.penalty is None:
                grid_search = sklearn.model_selection.GridSearchCV(
                    self.classifier, {
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
                    grid_search.fit(features, y)
                else:
                    # If the training set is too large, subsample 10000 train
                    # examples
                    split = sklearn.model_selection.train_test_split(
                        features, y,
                        train_size=10000, random_state=0, stratify=y
                    )
                    grid_search.fit(split[0], split[2])
                self.classifier = grid_search.best_estimator_
                return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.encoder)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.encoder.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.encoder

    def fit(self, X, y, save_memory=False, verbose=False):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Fitting encoder
        self.encoder = self.fit_encoder(
            X, y=y, save_memory=save_memory, verbose=verbose
        )

        # SVM classifier training
        features = self.encode(X)
        self.classifier = self.fit_classifier(features, y)

        return self

    def encode(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        """
        Outputs the representations associated to the input by the encoder,
        for each subseries of the input of the given size (sliding window
        representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA.
        @param window_batch_size Size of batches of windows to compute in a
               run of encode, to save RAM.
        """
        features = numpy.empty((
                numpy.shape(X)[0], self.out_channels,
                numpy.shape(X)[2] - window + 1
        ))
        masking = numpy.empty((
            min(window_batch_size, numpy.shape(X)[2] - window + 1),
            numpy.shape(X)[1], window
        ))
        for b in range(numpy.shape(X)[0]):
            for i in range(math.ceil(
                (numpy.shape(X)[2] - window + 1) / window_batch_size)
            ):
                for j in range(
                    i * window_batch_size,
                    min(
                        (i + 1) * window_batch_size,
                        numpy.shape(X)[2] - window + 1
                    )
                ):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j: j + window]
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = numpy.swapaxes(
                    self.encode(masking[:j0 + 1], batch_size=batch_size), 0, 1
                )
        return features

    def predict(self, X, batch_size=50):
        """
        Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, batch_size=50):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.score(features, y)


class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


class LSTMEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps an LSTM encoder of time series as a PyTorch module and a SVM
    classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of
           the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param cuda Transfers, if True, all computations to the GPU.
    @param in_channels Number of input channels of the time series.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, in_channels=1, cuda=False,
                 gpu=0):
        super(LSTMEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(cuda, gpu), {}, in_channels, 160, cuda, gpu
        )
        assert in_channels == 1
        self.architecture = 'LSTM'

    def __create_encoder(self, cuda, gpu):
        encoder = networks.lstm.LSTMEncoder()
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'in_channels': self.in_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, in_channels, cuda, gpu
        )
        return self
