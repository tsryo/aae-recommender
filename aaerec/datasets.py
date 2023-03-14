# coding: utf-8
import math
import random
from builtins import filter # nanu?
from collections import Counter
import itertools as it

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import defaultdict
try:
    from .transforms import lists2sparse, sparse2lists
except SystemError:
    from transforms import lists2sparse, sparse2lists

def split_by_mask(data, condition):
    """ Splits data on index depending on condition """
    truthy = [d for i, d in enumerate(data) if condition[i]]
    falsy = [d for i, d in enumerate(data) if not condition[i]]
    return truthy, falsy



def magic(S, N, alpha=0.05):
    return S**2 * math.log(S * N / alpha)


def build_vocab(sets, min_count=None, max_features=None):
    """ Builds the vocabulary with respect to :code:`max_features` most common
    tokens and :code:`min_count` minimal set length """
    # sort descending by frequency and only keep max_features
    counts = Counter(it.chain.from_iterable(sets)).most_common(max_features)
    # [ ('token', 42) ]

    # can be optimized
    if min_count:
        counts = list(it.takewhile(lambda c: c[1] >= min_count, counts))
        # counts = list(filter(lambda c: c[1] >= min_count, counts))

    # incrementally assign indicies to tokens
    vocab = {}

    for token, __ in counts:
        vocab[token] = len(vocab)

    return vocab, counts


def filter_vocab(lists, vocab):
    """
    Filters out-of-vocabulary tokens from iterable of iterables.
    We could impose special UNK treatment here.
    """
    return [[t for t in tokens if t in vocab] for tokens in lists]


def apply_vocab(lists, vocab):
    """ Applys vocab to iterable of iterables. This function can also be used
    to invert the mapping. """
    return [[vocab[t] for t in l] for l in lists]

def filter_apply_vocab(lists, vocab):
    """ Filters for valid tokens and transforms them to index. Faster then
    first filter then apply. """
    return [[vocab[t] for t in tokens if t in vocab] for tokens in lists]

def filter_length(lists, min_length, *supplements):
    """
    Filters lists and supplements with respect to len(list) > min_length
    """
    enough = [len(bag) >= min_length for bag in lists]
    lists_reduced = [bag for i, bag in enumerate(lists) if enough[i]]
    if not supplements:
        return lists_reduced

    sup_reduced = []
    for supplement in supplements:
        sup_reduced.append([o for i, o in enumerate(supplement) if enough[i]])
    return (lists_reduced, *sup_reduced)



def split_set(s, criterion):
    """
    Splits a set according to criterion
    if criterion is float: toss a coin for each element
    if criterion is an int: drop as many random elements
    if criterion is callable: drop each element iff criterion(element) returns
    False

    In either case, the result is (remainder_set, dropped_elements)
    """
    s = set(s)

    if callable(criterion):
        todrop = {e for e in s if criterion(e)}
    elif type(criterion) == float:
        assert criterion > 0 and criterion < 1, "Float not bounded in (0,1)"
        n_elems_to_drop = len(s)*criterion
        remainder = n_elems_to_drop - np.floor(n_elems_to_drop)
        should_round_up = random.random() >= 1-remainder
        n_elems_to_drop = np.ceil(n_elems_to_drop) if should_round_up else np.floor(n_elems_to_drop)
        todrop = random.sample(s, int(n_elems_to_drop))
        # todrop = {e for e in s if random.random() < criterion}
    elif type(criterion) == int:
        try:
            todrop = random.sample(s, criterion)
        except ValueError:  # too few elements in s
            todrop = s
    else:
        raise ValueError('int, float, or callable expected')

    todrop = set(todrop)
    return s - todrop, todrop


def corrupt_sets(sets, drop=1):
    """

    Splits a list of sets into two sub-sets each,
    one containing corrupted sets and one retaining the removed elements
    """
    split = [split_set(s, drop) for s in sets]

    return tuple(zip(*split))


class Bags(object):
    """
    Loads the data set and creates the mapping from tokens to indices

    1. Split line by tab and extract the owner token from the first column and
       split the second column by comma to obtain sets
    2. Use remaining columns to extract attributes of the owner
    3. Count element occurrences of these sets and retain ``max_featues``
        most common.
    3. Assign indices to tokens (descending with respect to their count)
    4. Remove sets that then have less than ``min_elements`` elements.
    5. Store mappings and data
            * ``vocabulary`` : token -> index
            * ``index2token`` : index -> token
            * ``data`` : sets with tokens replaced by indices
            * ``owner_attributes`` : dictionary of dictionaries, where
              owner_attributes[<attribute>][<token>] returns the <attribute>
              value of owner <token>
            * (DEPRECATED) ``element_attributes`` : dictionary of dictionaries, where
              element_attributes[<attribute>][<token>] returns the <attribute>
              value of set element <token>

    There is no special treatment for ``'UNK'`` tokens.

    >>> bags = Bags.from_sets([[1,2,3] , [2,3,4,1] , [42, 1, 7]])
    >>> len(bags)
    3
    >>> bags.size()
    (3, 6)
    """
    def __init__(self,
                 data,
                 owners,
                 owner_attributes=None):
        """

        :param data: ???, ???
        :param owners: iterable (prob list), of ids
        :param owner_attributes: dict, of dicts in form of {attribute: {id: <unknown>}}
        """
        # TODO: think about: split between prediction relevant attributes and data splitting infos like year
        assert len(owners) == len(data)
        self.data = data
        self.bag_owners = owners
        # attributes are called by keys --> just adding new key will suffice
        self.owner_attributes = owner_attributes

        # consumes performance
        # self.index2token = invert_mapping(vocab)

    def clone(self, start_from=0, n_items=None):
        """ Creates a really deep copy.
            @param start_from - (Optional) specify from which item to start cloning (default is 0)
            @param n_items - (Optional) specify how many items to clone from original object (default is all)
        """
        n_items = len(self.data) if n_items is None else n_items + start_from
        data = [[t for t in self.data[b]] for b in range(start_from, n_items)]
        bag_owners = [self.bag_owners[o] for o in range(start_from, n_items)]
        if self.owner_attributes is not None:
            owner_attributes = {list(self.owner_attributes.keys())[i]: {list(list(self.owner_attributes.items())[i][1].keys())[j]: list(list(self.owner_attributes.items())[i][1].values())[j]
                                                                        for j in range(start_from, n_items)}
                                for i in range(len(self.owner_attributes.items()))}

        return Bags(data, bag_owners, owner_attributes=owner_attributes)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "{} records with {} ratings".format(len(self), self.numel())

    def __getitem__(self, idx):
        return self.data[idx]

    def maxlen(self):
        """ Returns the maximum bag length """
        return max(map(len, self.data))

    def numel(self):
        """ Computes the number of (non-zero) elements """
        return sum(map(len, self.data))

    def get_single_attribute(self, attribute):
        # TODO: rename to get_attribute() again? at least check for all other calls
        # TODO: find representation in owner_attributes for doctex
        """
        Retrieves the attribute 'attribute' of each bag owner and returns them as a list
        in the same order as self.bag_owners.
        :param attribute: hashable (str), key in owner_attributes (~ side_info)
        :return: vectorizable, ordered like Bag data containing respective attribute
        """
        if self.owner_attributes is None or self.bag_owners is None:
            raise ValueError("Owners not present")

        # find how attributes are used --> starting at top level to see what is needed
        # Answer: vectorizable
        # use it like before, it worked there



        attribute_l = []
        for owner in self.bag_owners:
            attribute_l.append(self.owner_attributes[attribute][owner])

        return attribute_l

    def get_attributes(self, attribute_list):
        return [self.get_single_attribute(a) for a in attribute_list]

    def to_dict(self):
        return dict(enumerate(self.data))

    @classmethod
    def load_tabcomma_format(self, path, meta_data_dic = False, unique=False, owner_str="owner", set_str="set"):
        """

        Returns ordered lists for Owner, Set of Owner and a
        Arguments
        =========

        """
        # loading

        df = pd.read_csv(path, sep=",", dtype=str, error_bad_lines=False)
        df = df.fillna("")

        set_owners = df[owner_str].values
        sets = df[set_str].values
        sets = list(map(lambda x: x.split(","), sets))
        if unique:
            print("Making items unique within user.")
            sets = [list(set(s)) for s in sets]


        owner_attributes = dict()
        print("Found", len(sets), 'rows')

        header = list(df.columns.values)
        fields_freqs = ["{}={}".format(h, df[h].count() / len(df)) for h in header]
        print("Metadata-fields' frequencies: " + ", ".join(fields_freqs))

        meta_vals = []
        for meta_header in header[2:]:
            meta_vals.append(df[meta_header].values)
        #print("with", len(header) - 2, "metadata columns.")

        for i in range(2, len(header)):
            owner_attributes[header[i]] = {}
            for j, owner in enumerate(set_owners):
                owner_attributes[header[i]][owner] = meta_vals[i - 2][j]


        if meta_data_dic:


            def iterate_metadata(meta_data, mtdt_transform_table):
                """
                    accumulates lists for specified attributes per "owner"
                    Warning: not working if dependent on other additional keys
                :param meta_data: pandas DataFrame with the columns for each attribute
                :param mtdt_transform_table: a dictionary with information which attributes are relevant
                    # key: name of a table
                    # owner_id: ID of citing paper
                    # fields: list of column names in table
                    # target names: key for these data in the owner_attributes dictionary
                    # path: absolute path to the csv file
                :return: dict owner_attributes[<attr_name>][<documentID>].append(<attribute(s)>)
                """
                print("create dict from df")
                owner_attributes = {}
                for target_attr_name in mtdt_transform_table["target_names"]:
                    owner_attributes[target_attr_name] = defaultdict(list)

                for index, row in meta_data.iterrows():
                    # owner_attributes[<attr_name>][<documentID>].append(<attribute(s)>)
                    for attr, target_name in zip(mtdt_transform_table["fields"],mtdt_transform_table["target_names"]):
                        owner_id = row[mtdt_transform_table["owner_id"]]
                        attr_value = row[attr]
                        owner_attributes[target_name][owner_id].append(attr_value)



                print("creating dict finished")
                return owner_attributes

            # loads each table from file, selects the relevant attributes and adds them to the owner_attributes
            for key in meta_data_dic.keys():
                mtdt_transform_table = meta_data_dic[key]
                # loading meta data and select the relevant
                auth_path = mtdt_transform_table["path"]
                meta_data = pd.read_csv(auth_path, error_bad_lines=False, dtype=str)
                print("Metadata-fields' frequencies: {}={}".format(key,
                    meta_data[mtdt_transform_table["owner_id"]].nunique() / len(df)))

                targets = [mtdt_transform_table["owner_id"]] + mtdt_transform_table["fields"]
                meta_data = meta_data[targets]
                # add the additional attributes

                owner_attributes.update(iterate_metadata(meta_data, mtdt_transform_table))

        bags = Bags(sets, set_owners, owner_attributes=owner_attributes)

        return bags

    def train_test_split(self, on_year=None, n_folds = 1, **split_params):
        """ Returns one training bag instance and one test bag instance.
        Builds the vocabulary from the training set.

        :param on_year: int, split on this year
        :param **split_params:
        :return: tuple, first training bag instance, second test bag instance
        """
        if on_year is not None:
            print("Splitting data on year:", on_year)
            assert self.owner_attributes['year'], "Cant split on non-existing 'year'"
            on_year = int(on_year)
            is_train = [int(y) < on_year for y in self.get_single_attribute('year')]
            train_data, test_data = split_by_mask(self.data, is_train)
            train_owners, test_owners = split_by_mask(self.bag_owners, is_train)
        else:
            print("Splitting on params:", split_params)
            split = train_test_split(self.data, self.bag_owners, **split_params)
            train_data, test_data, train_owners, test_owners = split
        print("{} train, {} test documents.".format(len(train_data), len(test_data)))
        if self.owner_attributes is not None:
            metadata_columns = list(self.owner_attributes.keys())
            train_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                                train_owners} for k in metadata_columns}
            test_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                               test_owners} for k in metadata_columns}
        else:
            train_attributes = test_attributes = None
        train_set = Bags(train_data, train_owners, owner_attributes=train_attributes)
        test_set = Bags(test_data, test_owners, owner_attributes=test_attributes)
        return train_set, test_set

    def create_kfolds_train_test(self, n_folds = 1):
        """
        """
        if n_folds == 1:
            train_set, test_set = self.train_test_split()
            return [train_set], [test_set]

        kf = KFold(n_folds)
        folds = kf.split(self.data)
        fold_idxs = [(train_idx, test_idx) for train_idx,test_idx in folds]
        train_sets,  test_sets = [], []
        for c_fold in range(len(fold_idxs)):
            c_train_idxs, c_test_idxs = fold_idxs[c_fold]
            train_data = [self.data[x] for x in c_train_idxs]
            test_data = [self.data[x] for x in c_test_idxs]
            train_owners = [self.bag_owners[x] for x in c_train_idxs]
            test_owners = [self.bag_owners[x] for x in c_test_idxs]
            print("fold {} ; {} train, {} test documents.".format(c_fold, len(train_data), len(test_data)))
            if self.owner_attributes is not None:
                metadata_columns = list(self.owner_attributes.keys())
                train_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                                    train_owners} for k in metadata_columns}
                test_attributes = {k: {owner: self.owner_attributes[k][owner] for owner in
                                   test_owners} for k in metadata_columns}
            else:
                train_attributes = test_attributes = None
            train_set = Bags(train_data, train_owners, owner_attributes=train_attributes)
            test_set = Bags(test_data, test_owners, owner_attributes=test_attributes)
            train_sets.append(train_set)
            test_sets.append(test_set)

        return train_sets, test_sets

    def create_kfold_train_validate_test(self, n_folds = 1):
        """invokes train_test_split but further splits test set into validate and test sets in a 1:1 ratio"""
        train_sets, test_sets = self.create_kfolds_train_test(n_folds = n_folds)
        # use first half of "test_set" as validate set, and second half as true test set
        val_sets, new_test_sets = [], []
        for i in range(n_folds):
            test_set = test_sets[i]
            idx_splt = int(np.floor(len(test_set)/2))
            idx_splt2 = int(np.ceil(len(test_set)/2))
            validate_set = test_set.clone(start_from=0, n_items=idx_splt)
            test_set = test_set.clone(start_from=idx_splt, n_items=idx_splt2)
            new_test_sets.append(test_set)
            val_sets.append(validate_set)

        return train_sets, val_sets, new_test_sets
    def train_validate_test_split(self, on_year=None, **split_params):
        """invokes train_test_split but further splits test set into validate and test sets in a 1:1 ratio"""
        train_set, test_set = self.train_test_split(on_year=on_year, **split_params)
        # use first half of "test_set" as validate set, and second half as true test set
        idx_splt = int(np.floor(len(test_set)/2))
        idx_splt2 = int(np.ceil(len(test_set)/2))
        validate_set = test_set.clone(start_from=0, n_items=idx_splt)
        test_set = test_set.clone(start_from=idx_splt, n_items=idx_splt2)
        return train_set, validate_set, test_set

    def build_vocab(self, min_count=None, max_features=None, apply=True):
        """
        Returns BagsWithVocab instance if apply is True, else vocabulary along with counts.
        """
        vocab, counts = build_vocab(self.data, min_count=min_count,
                                    max_features=max_features)
        if apply:
            return self.apply_vocab(vocab)

        return vocab, counts


    def apply_vocab(self, vocab):
        """
        Applies `vocab` and returns `BagsWithVocab` instance
        """
        data_ix = filter_apply_vocab(self.data, vocab)
        return BagsWithVocab(data_ix, vocab, owners=self.bag_owners,
                             attributes=self.owner_attributes)

    def prune_(self, min_elements=0):
        """ Prunes data and set_owners according such that only rows with
        min_elements in data are retained """
        data = self.data
        owners = self.bag_owners
        if min_elements:
            data, owners = filter_length(data, min_elements, owners)
            attributes = {k: {owner: self.owner_attributes[k][owner] for owner
                              in owners} for k in
                          list(self.owner_attributes.keys())}
        self.data = data
        self.bag_owners = owners
        self.owner_attributes = attributes
        return self

    def inflate(self, factor):
        """
        Inflates the bag by injecting 'factor' repetitions of the current data (and respective owner) into the bag.

        Parameters:
        =============
        factor : int
            Determines how often the data is repeated in the bag.


        Returns:
        =============
        inflated_bag : The same Bags instance on which the function is called.
        """

        # no
        # current_data = self.data.copy()
        # current_owners = self.bag_owners.copy()
        raise DeprecationWarning("This method should not be used")

        for __i in range(1, factor):
            self.data.extend([[t for t in b] for b in self.data])
            self.bag_owners.extend([o for o in self.bag_owners])

        return self


class BagsWithVocab(Bags):
    def __init__(self, data, vocab, owners=None, attributes=None):
        super(BagsWithVocab, self).__init__(data, owners,
                                            owner_attributes=attributes)
        self.vocab = vocab
        # array of tokens which acts as reverse vocab
        self.index2token = {v: k for k, v in vocab.items()}

    def clone(self, start_from=0, n_items=None):
        """ Creates a really deep copy """
        # safe cloning of an instance
        # deepcopy is NOT enough
        n_items = len(self.data) if n_items is None else n_items + start_from
        data = [[t for t in self.data[b]] for b in range(start_from, n_items)]
        vocab = {k: v for k, v in self.vocab.items()}
        bag_owners = [self.bag_owners[o] for o in range(start_from, n_items)]
        owner_attributes = None
        if self.owner_attributes is not None:
            owner_attributes = {list(self.owner_attributes.keys())[i]: {list(list(self.owner_attributes.items())[i][1].keys())[j]: list(list(self.owner_attributes.items())[i][1].values())[j]
                                                                        for j in range(start_from, n_items)}
                                for i in range(len(self.owner_attributes.items()))}

        return BagsWithVocab(data, vocab, owners=bag_owners,
                             attributes=owner_attributes)

    def build_vocab(self, min_count=None, max_features=None, apply=True):
        """ Override to prevent errors like building vocab of indices """
        raise ValueError("Instance already has vocabulary.")

    def apply_vocab(self, vocab):
        """ Override to prevent errors like building vocab of indices """
        raise ValueError("A vocabulary has already been applied.")

    def __str__(self):
        s = "{} elements in [{}, {}] [data_points,vocabulary_size] with density {}"
        return s.format(self.numel(), *self.size(), self.density())

    def size(self, dim=None):
        sizes = (len(self.data), len(self.vocab))

        if dim is not None:
            return sizes[dim]
        else:
            return sizes

    def tocsr(self, data=None):
        """
        Use ``size`` to transform into scipy.sparse format
        """

        if data is None:
            data = self.data
            size = self.size()
        else:
            size = len(data), self.size(1)

        return lists2sparse(data, size).tocsr()

    def train_test_split(self, **split_params):
        """ Returns one training bags instance and one test bag instance.
        Builds the vocabulary from the training set.
        """
        train_bags, test_bags = super().train_test_split(**split_params)
        train_set = BagsWithVocab(train_bags.data, self.vocab, owners=train_bags.bag_owners,
                                  attributes=train_bags.owner_attributes)
        test_set = BagsWithVocab(test_bags.data, self.vocab, owners=test_bags.bag_owners,
                                 attributes=test_bags.owner_attributes)
        return train_set, test_set

    def density(self):
        """ Computes the density: number of elements divided by dimensions """
        return self.numel() / np.product(self.size())

    def magic_number(self, std_factor=None, alpha=0.05):
        """ Computes the magic number for sparse retrieval with given error
        probability :code:`alpha`. Optionally adds fractions of the standard deviation of set length """
        lens = np.array(list(map(len, self.data)))
        S = lens.mean()

        if std_factor:
            S += std_factor * lens.std()

        return int(magic(S, self.size(1), alpha=alpha)) + 1

    def to_index(self, data):
        """
        Uses 'vocabulary' to transforms a list of
         sets of tokens ('data') to a list of sets of indices.
        """
        raise DeprecationWarning("Use apply_vocab(data, self.vocab) instead")
        return apply_vocab(data, self.vocab)

    def to_tokens(self, data):
        """
        Uses 'index2token' to transforms a list of
         sets of indices ('data') to a list of sets of tokens.
        """
        raise DeprecationWarning("Use apply_vocab(data, self.index2token) instead")
        return apply_vocab(data, self.index2token)

    def raw(self):
        """ Returns the data with original identifiers instead of indices """
        # important not to filter here, better raise if something wrong
        return apply_vocab(self.data, self.index2token)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
