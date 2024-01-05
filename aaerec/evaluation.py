from timeit import default_timer as timer
from datetime import timedelta
import os
import random
import sys
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import minmax_scale
import numpy as np
import scipy.sparse as sp
from . import rank_metrics_with_std as rm
from .datasets import corrupt_lists
from .transforms import lists2sparse
from aaerec.rank_metrics_with_std import mean_average_f1


def argtopk(X, k):
    """
    Picks the top k elements of (sparse) matrix X
    modified to work with repeating elements
    >>> X = np.arange(10).reshape(1, -1)
    >>> i = argtopk(X, 3)
    >>> i
    (array([[0]]), array([[9, 8, 7]]))
    >>> X[argtopk(X, 3)]
    array([[9, 8, 7]])
    >>> X = np.arange(20).reshape(2,10)
    >>> ix, iy = argtopk(X, 3)
    >>> ix
    array([[0],
           [1]])
    >>> iy
    array([[9, 8, 7],
           [9, 8, 7]])
    >>> X[ix, iy]
    array([[ 9,  8,  7],
           [19, 18, 17]])
    >>> X = np.arange(6).reshape(2,3)
    >>> X[argtopk(X, 123123)]
    array([[2, 1, 0],
           [5, 4, 3]])
    """
    assert len(X.shape) == 2, "X should be two-dimensional array-like"
    assert k is None or k > 0, "k should be positive integer or None"
    rows = np.arange(X.shape[0])[:, np.newaxis]

    new_inds = None # handle repeating elements
    c_max = int(np.ceil(np.max(X)))
    for r_i in range(X.shape[0]):
        c_x = X[r_i]
        n_x = c_x.copy()
        ns_x = c_x.copy()
        for nr_i in range(c_max):
            n_x = n_x - 1
            n_x[n_x < 0] = 0
            ns_x = np.vstack((ns_x, n_x))
        ns_x_flat = ns_x.flatten()
        new_ind = np.argsort(-ns_x_flat, axis=0)
        new_ind = new_ind % X.shape[1]
        if new_inds is not None:
            new_inds = np.vstack((new_inds, new_ind))
        else:
            new_inds = new_ind

    if k is not None and k < X.size:
        new_inds = new_inds[:, :k]

    return rows, new_inds # new_inds = rank score



    # # todo: handle repeating items
    # ind = np.argpartition(X, -k, axis=1)[:, -k:]
    # # sort indices depending on their X values
    # cols = ind[rows, np.argsort(X[rows, ind], axis=1)][:, ::-1]
    # return rows, cols


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, y_true, y_pred, average=True):
        pass

class RankingMetric(Metric):
    """ Base class for all ranking metrics
    may also be used on its own to quickly get ranking scores from Y_true,
    Y_pred pair
    """

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', None)
        super().__init__()

    def __call__(self, y_true, y_pred, average=True):
        """ Gets relevance scores,
        Sort based on y_pred, then lookup in y_true
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> RankingMetric(k=2)(Y_true, Y_pred)
        array([[0, 1], # because the best predicted item is the 2nd (0.3), so at first returned relevance score  will be 0, next one will be 1 
               [1, 0]])  # first item returned will be the 3rd (0.7), so that is a rs of 1 already there
        """ # y_true[3,1] = 2; [3,0/2] = 1; y_pred[3,] = argsort = 10,  2,  5,  0,  9, 12,  1, 11,  6,  3,  4,  7,  8, 13, 14, 15; then we expect rs [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ind = argtopk(y_pred, self.k)

        rs = y_true[ind]
        # n_splits = int(rs.shape[1] / y_true.shape[1])
        # if n_splits != 0:
        #     portions = np.split(rs, n_splits, axis=1)
        #     new_rs = None
        #     for portion in portions:
        #         if new_rs is None:
        #             new_rs = portion
        #         else:
        #             new_rs = new_rs + portion
        #     return new_rs
        return rs


class MRR(RankingMetric):
    """ Mean reciprocal rank at k

    >>> mrr_at_5 = MRR(5)
    >>> callable(mrr_at_5)
    True
    >>> Y_true = np.array([[1,0,0],[0,0,1]])
    >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
    >>> MRR(2)(Y_true, Y_pred)
    (0.75, 0.25)
    >>> Y_true = np.array([[1,0,1],[1,0,1]])
    >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
    >>> MRR(3)(Y_true, Y_pred)
    (1.0, 0.0)
    """
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        # compute mrr wrt k
        rs = super().__call__(y_true, y_pred)
        return rm.mean_reciprocal_rank(rs, average=average)


class MAP(RankingMetric):
    """ Mean average precision at k """
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        """
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> MAP(2)(Y_true, Y_pred)
        (0.75, 0.25)
        >>> Y_true = np.array([[1,0,1],[1,0,1]])
        >>> Y_pred = np.array([[0.3,0.2,0.3],[0.6,0.5,0.7]])
        >>> MAP(3)(Y_true, Y_pred)
        (1.0, 0.0)
        >>> Y_true = np.array([[1,0,1],[1,1,1]])
        >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
        >>> MAP(3)(Y_true, Y_pred)
        (0.9166666666666666, 0.08333333333333337)
        """
        rs = super().__call__(y_true, y_pred)
        if average:
            return rm.mean_average_precision(rs)
        else:
            return np.array([rm.average_precision(r) for r in rs])

class MAF1(RankingMetric):
    """ Mean average F1 score at k """
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        """
        """
        rs = super().__call__(y_true, y_pred)
        k = self.k if self.k is not None else y_true.shape[1]
        # map, map_std = mean_average_precision(rs)
        # mar, mar_std = mean_average_recall(rs)
        # rec = [ recall_at_k(rs[i], k, sum(y_true[i])) for i in range(len(rs))]
        all_pos_nums = np.array([sum(y_true[i]) for i in range(y_true.shape[0])])
        maf1, maf1_std = mean_average_f1(rs, all_pos_nums)
        return maf1, maf1_std

class P(RankingMetric):
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        """
        >>> Y_true = np.array([[1,0,1,0],[1,0,1,0]])
        >>> Y_pred = np.array([[0.2,0.3,0.1,0.05],[0.2,0.5,0.7,0.05]])
        >>> P(2)(Y_true, Y_pred)
        (0.5, 0.0)
        >>> P(4)(Y_true, Y_pred)
        (0.5, 0.0)
        """
        # compute p wrt k
        rs = super().__call__(y_true, y_pred)
        ps = (rs > 0).mean(axis=1)
        if average:
            return ps.mean(), ps.std()
        else:
            return ps

BOUNDED_METRICS = {
    # (bounded) ranking metrics
    '{}@{}'.format(M.__name__.lower(), k): M(k)
    for M in [MRR, MAP, P, MAF1] for k in [5, 10, 20]
}
BOUNDED_METRICS['P@1'] = P(1)


UNBOUNDED_METRICS = {
    # unbounded metrics
    M.__name__.lower(): M()
    for M in [MRR, MAP, MAF1]
}

METRICS = { **BOUNDED_METRICS, **UNBOUNDED_METRICS }


def remove_non_missing(Y_pred, X_test, copy=True):
    """
    Scales the predicted values between 0 and 1 and  sets the known values to
    zero.
    >>> Y_pred = np.array([[0.6,0.5,-1], [40,-20,10]])
    >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
    >>> remove_non_missing(Y_pred, X_test)
    array([[0.    , 0.9375, 0.    ],
           [1.    , 0.    , 0.5   ]])
    """
    Y_pred_scaled = Y_pred.copy()
    # Y_pred_scaled = minmax_scale(Y_pred,
    #                              feature_range=(0, 1),
    #                              axis=1,  # Super important!
    #                              copy=copy)
    # we remove the ones that were already present in the orig set
    Y_pred_scaled[X_test.nonzero()] -= 1.
    Y_pred_scaled[Y_pred_scaled < 0] = 0
    return Y_pred_scaled


def evaluate(ground_truth, predictions, metrics, batch_size=None):
    """
    Main evaluation function, used by Evaluation class but can also be
    reused to recompute metrics
    """

    n_samples = ground_truth.shape[0]
    # x = pd.DataFrame(ground_truth.toarray())
    # # todo: hack - make more robust based on drop percentage
    # x = x[x.sum(axis=1) >= 1]

    assert predictions.shape[0] == n_samples

    metrics = [m if callable(m) else METRICS[m] for m in metrics]

    if batch_size is not None:
        batch_size = int(batch_size)

        # Important: Results consist of Mean + Std dev
        # Add all results per sample to array
        # Average later
        results_per_metric = [[] for _ in range(len(metrics))]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples) 
            pred_batch = predictions[start:end, :]
            gold_batch = ground_truth[start:end, :]
            if sp.issparse(pred_batch):
                pred_batch = pred_batch.toarray()
            if sp.issparse(gold_batch):
                gold_batch = gold_batch.toarray()

            for i, metric in enumerate(metrics):
                results_per_metric[i].extend(metric(gold_batch, pred_batch, average=False))

        results = [(x.mean(), x.std()) for x in map(np.array, results_per_metric)]
    else:
        if sp.issparse(ground_truth):
            ground_truth = ground_truth.toarray()
        if sp.issparse(predictions):
            predictions = predictions.toarray()
        results = [metric(ground_truth, predictions) for metric in metrics]

    return results


def reevaluate(gold_file, predictions_file, metrics):
    """ Recompute metrics from files """
    Y_test = sp.load_npz(gold_file)
    Y_pred = np.load(predictions_file)
    return evaluate(Y_test, Y_pred, metrics)


def maybe_open(logfile, mode='a'):
    """
    If logfile is something that can be opened, do so else return STDOUT
    """
    return open(logfile, mode) if logfile else sys.stdout


def maybe_close(log_fh):
    """ Close if log_fh is not STDOUT """
    if log_fh is not sys.stdout:
        log_fh.close()


class Evaluation(object):
    def __init__(self,
                 dataset,
                 year,
                 metrics=METRICS,
                 logfile=sys.stdout,
                 logdir=None):
        self.dataset = dataset
        self.year = year
        self.metrics = metrics
        self.logfile = logfile
        self.logdir = logdir

        self.train_set, self.test_set = None, None
        self.x_test, self.y_test = None, None

    def setup(self, seed=42, min_elements=1, max_features=None,
              min_count=None, drop=1):
        # we could specify split criterion and drop choice here
        """ Splits and corrupts the data accordion to criterion """
        log_fh = maybe_open(self.logfile)
        random.seed(seed)
        np.random.seed(seed)
        # train_set, test_set = self.dataset.split(self.split_test,
        #                                          self.split_train)
        train_set, test_set = self.dataset.train_test_split(on_year=self.year)
        print("=" * 80, file=log_fh)
        print("Train:", train_set, file=log_fh)
        print("Test:", test_set, file=log_fh)
        print("Next Pruning:\n\tmin_count: {}\n\tmax_features: {}\n\tmin_elements: {}"
              .format(min_count, max_features, min_elements), file=log_fh)
        train_set = train_set.build_vocab(min_count=min_count,
                                          max_features=max_features,
                                          apply=True)
        test_set = test_set.apply_vocab(train_set.vocab)
        # Train and test sets are now BagsWithVocab
        train_set.prune_(min_elements=min_elements)
        test_set.prune_(min_elements=min_elements)
        print("Train:", train_set, file=log_fh)
        print("Test:", test_set, file=log_fh)
        print("Drop parameter:", drop)

        noisy, missing = corrupt_lists(test_set.data, drop=drop)

        assert len(noisy) == len(missing) == len(test_set)

        test_set.data = noisy
        print("-" * 80, file=log_fh)
        maybe_close(log_fh)

        # THE GOLD
        self.y_test = lists2sparse(missing, test_set.size(1)).tocsr(copy=False)

        self.train_set = train_set
        self.test_set = test_set

        # just store for not recomputing the stuff
        self.x_test = lists2sparse(noisy, train_set.size(1)).tocsr(copy=False)
        return self

    def __call__(self, recommenders, batch_size=None):
        if None in (self.train_set, self.test_set, self.x_test, self.y_test):
            raise UserWarning("Call .setup() before running the experiment")

        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            vocab_path = os.path.join(self.logdir, "vocab.txt")
            with open(vocab_path, 'w') as vocab_fh:
                print(*self.train_set.index2token, sep='\n', file=vocab_fh)
            gold_path = os.path.join(self.logdir, "gold")
            sp.save_npz(gold_path, self.y_test)

        for recommender in recommenders:
            log_fh = maybe_open(self.logfile)
            print(recommender, file=log_fh)
            maybe_close(log_fh)
            train_set = self.train_set.clone()
            test_set = self.test_set.clone()
            t_0 = timer()
            # DONE FIXME copy.deepcopy is not enough!
            recommender.train(train_set)
            log_fh = maybe_open(self.logfile)
            print("Training took {} seconds."
                  .format(timedelta(seconds=timer()-t_0)), file=log_fh)

            t_1 = timer()
            y_pred = recommender.predict(test_set)
            if sp.issparse(y_pred):
                y_pred = y_pred.toarray()
            else:
                # dont hide that we are assuming an ndarray to be returned
                y_pred = np.asarray(y_pred)

            # set likelihood of documents that are already cited to zero, so
            # they don't influence evaluation
            y_pred = remove_non_missing(y_pred, self.x_test, copy=True)

            print("Prediction took {} seconds."
                  .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            if self.logdir:
                t_1 = timer()
                pred_file = os.path.join(self.logdir, repr(recommender))
                np.save(pred_file, y_pred)
                print("Storing predictions took {} seconds."
                      .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            t_1 = timer()
            results = evaluate(self.y_test, y_pred, metrics=self.metrics, batch_size=batch_size)
            print("Evaluation took {} seconds."
                  .format(timedelta(seconds=timer()-t_1)), file=log_fh)

            print("\nResults:\n", file=log_fh)
            for metric, (mean, std) in zip(self.metrics, results):
                print("- {}: {} ({})".format(metric, mean, std),
                      file=log_fh)
            print("\nOverall time: {} seconds."
                  .format(timedelta(seconds=timer()-t_0)), file=log_fh)
            print('-' * 79, file=log_fh)
            maybe_close(log_fh)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
