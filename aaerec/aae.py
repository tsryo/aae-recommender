""" Adversarially Regualized Autoencoders """
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# sklearn
import sklearn
from .ub import AutoEncoderMixin

# numpy
import numpy as np
import scipy.sparse as sp

# own recommender stuff
from .base import Recommender
from .condition import _check_conditions
import copy


torch.manual_seed(42)
TINY = 1e-12

STATUS_FORMAT = "[ R: {:.4f} | D: {:.4f} | G: {:.4f} ]"

def assert_condition_callabilities(conditions):
    raise DeprecationWarning("Use _check_conditions(conditions, condition_data) instead")
    if type(conditions) == type(True):
        pass
    else:
        assert type(conditions) != type("") and hasattr(conditions,'__iter__'), "Conditions needs to be a list of different conditions. It is a {} now.".format(type(conditions))

def concat_side_info(vectorizer,training_set,side_info_subset):
    """
    Constructing an np.array with having the concatenated features in shape[1]
    :param training_set: Bag class dataset,
    :side_info_subset: list of str, the attribute keys in Bag class
    :return:
    """
    raise DeprecationWarning("Use ConditionList.encode_impose(...) instead")
    attr_vect = []
    # ugly substitute for do_until pattern
    for i, attribute in enumerate(side_info_subset):
        attr_data = training_set.get_single_attribute(attribute)
        if i < 1:
            attr_vect = vectorizer.fit_transform(attr_data)
        else:
            # rows are instances, cols are features --> adding cols makes up new features
            attr_vect = np.concatenate((attr_vect, vectorizer.fit_transform(attr_data)), axis=1)
    return attr_vect


def log_losses(*losses):
    print('\r'+STATUS_FORMAT.format(*losses), end='', flush=True)


def sample_categorical(size):
    batch_size, n_classes = size
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat


def sample_bernoulli(size):
    ber = np.random.randint(0, 1, size).astype('float32')
    return torch.from_numpy(ber)


PRIOR_SAMPLERS = {
    'categorical': sample_categorical,
    'bernoulli': sample_bernoulli,
    'gauss': torch.randn
}

PRIOR_ACTIVATIONS = {
    'categorical': 'softmax',
    'bernoulli': 'sigmoid',
    'gauss': 'linear'
}


class Encoder(nn.Module):
    """ Three-layer Encoder """
    def __init__(self, n_input, n_hidden, n_code, final_activation=None,
                 normalize_inputs=True, dropout=(.2,.2), activation='ReLU'):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.act1 = getattr(nn, activation)()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = getattr(nn, activation)()
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.lin3 = nn.Linear(n_hidden, n_code)
        self.normalize_inputs = normalize_inputs
        if final_activation == 'linear' or final_activation is None:
            self.final_activation = None
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError("Final activation unknown:", activation)

    def forward(self, inp):
        """ Forward method implementation of 3-layer encoder """
        if self.normalize_inputs:
            inp = F.normalize(inp, 1)
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # third layer
        act = self.lin3(act)
        if self.final_activation:
            act = self.final_activation(act)
        return act


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, n_code, n_hidden, n_output, dropout=(.2,.2), activation='ReLU'):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_output)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward implementation of 3-layer decoder """
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # final layer
        act = self.lin3(act)
        act = torch.sigmoid(act)
        return act


class Discriminator(nn.Module):
    """ Discriminator """
    def __init__(self, n_code, n_hidden, dropout=(.2,.2), activation='ReLU'):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, 1)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward of 3-layer discriminator """
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)

        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)



        # act = F.dropout(self.lin1(inp), p=self.dropout[0], training=self.training)
        # act = F.relu(act)
        # act = F.dropout(self.lin2(act), p=self.dropout[1], training=self.training)
        # act = F.relu(act)
        return torch.sigmoid(self.lin3(act))


TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

class AutoEncoder():
    ### DONE Adapt to generic condition ###
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 lr=0.001,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2,.2),
                 conditions=None,
                 verbose=True):

        self.enc, self.dec = None, None
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.n_epochs = n_epochs
        self.optimizer = optimizer.lower()
        self.normalize_inputs = normalize_inputs
        self.verbose = verbose
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation

        self.conditions = conditions

    def eval(self):
        """ Put all NN modules into eval mode """
        ### DONE Adapt to generic condition ###
        self.enc.eval()
        self.dec.eval()
        if self.conditions:
            self.conditions.eval()

    def train(self):
        """ Put all NN modules into train mode """
        ### DONE Adapt to generic condition ###
        self.enc.train()
        self.dec.train()
        if self.conditions:
            self.conditions.train()


    def ae_step(self, batch, condition_data=None):
        """
        Perform one autoencoder training step
            :param batch: np.array, the base data from Bag class
            :param condition: condition_matrix: np.array, feature space of side_info
            :return: binary_cross_entropy for this step
            """
        ### DONE Adapt to generic condition ###

        # why is this double to AdversarialAutoEncoder? Lukas: it's likely the two models will diverge
        # what is relationship to train in DecodingRecommender? Train only uses Condition. Those are implementet seperately
        # assert_condition_callabilities(condition_matrix)
        z_sample = self.enc(batch)

        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z_sample = self.conditions.encode_impose(z_sample, condition_data)

        x_sample = self.dec(z_sample)
        x_sample = torch.nan_to_num(x_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        if self.conditions:
            self.conditions.zero_grad()
        recon_loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()

        if self.conditions:
            self.conditions.step()
        return recon_loss.item()

    def partial_fit(self, X, y=None, condition_data=None):
        """
            Performs reconstrction, discimination, generator training steps
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        _check_conditions(self.conditions, condition_data)


        if y is not None:
            raise ValueError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            X = X.cuda()

        # if condition_matrix is not None:
        #     condition_matrix = condition_matrix.astype('float32')
        #     if sp.issparse(condition_matrix):
        #         condition_matrix = condition_matrix.toarray()
        #     condition_matrix = Variable(torch.from_numpy(condition_matrix))
        #     if torch.cuda.is_available():
        #         condition_matrix = condition_matrix.cuda()

        # Make sure we are in training mode and zero leftover gradients
        self.train()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition_data=condition_data)
        if self.verbose:
            log_losses(recon_loss, 0, 0)
        return self

    def fit(self, X, y=None, condition_data=None):
        """
        :param X: np.array, the base data from Bag class
        :param y: dummy variable, throws Error if used
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        # TODO: check how X representation and numpy.array work together
        # TODO: adapt combining X and new_conditions_name
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        if use_condition:
            code_size = self.n_code + self.conditions.size_increment()
        else:
            code_size = self.n_code

        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation='linear',
                           normalize_inputs=self.normalize_inputs,
                           dropout=self.dropout, activation=self.activation)
        self.dec = Decoder(code_size, self.n_hidden,
                           X.shape[1], dropout=self.dropout, activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.lr)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            if use_condition:
                # shuffle(*arrays) takes several arrays and shuffles them so indices are still matching
                X_shuf, *condition_data_shuf = sklearn.utils.shuffle(X, *condition_data)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end].toarray()
                # condition may be None
                if use_condition:
                    # c_batch = condition_shuf[start:(start+self.batch_size)]
                    c_batch = [c[start:end] for c in condition_data_shuf]
                    self.partial_fit(X_batch, condition_data=c_batch)
                else:
                    self.partial_fit(X_batch)

            if self.verbose:
                # Clean up after flushing batch loss printings
                print()
        return self

    def predict(self, X, condition_data=None):
        """

        :param X: np.array, the base data from Bag class
        :param condition_matrix: np.array, feature space of side_info
        :return:
        """
        ### DONE Adapt to generic condition ###
        use_condition = _check_conditions(self.conditions, condition_data)
        self.eval()  # Deactivate dropout
        if self.conditions:
            self.conditions.eval()
        pred = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                # batched predictions, yet inclusive
                end = start + self.batch_size
                X_batch = X[start:end].toarray()
                X_batch = torch.FloatTensor(X_batch)
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                X_batch = Variable(X_batch)

                if use_condition:
                    c_batch = [c[start:end] for c in condition_data]

                z = self.enc(X_batch)
                if use_condition:
                    z = self.conditions.encode_impose(z, c_batch)
                # reconstruct
                X_reconstuction = self.dec(z)
                # shift
                X_reconstuction = X_reconstuction.data.cpu().numpy()
                pred.append(X_reconstuction)
        return np.vstack(pred)

    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.enc.zero_grad()
        self.dec.zero_grad()

    def reset_parameters(self):
        if self is not None:
            attrs_to_call = [getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and
                             hasattr(getattr(self, attr), 'reset_parameters')]
            for attr in attrs_to_call:
                attr.reset_parameters()
                attr.zero_grad()
        self.dec.lin1.reset_parameters()
        self.dec.lin2.reset_parameters()
        self.dec.lin3.reset_parameters()
        self.dec.lin1.reset_parameters()
        self.dec.lin2.reset_parameters()
        self.dec.lin3.reset_parameters()
        self.enc.lin1.reset_parameters()
        self.enc.lin2.reset_parameters()
        self.enc.lin3.reset_parameters()
        self.enc.lin1.reset_parameters()
        self.enc.lin2.reset_parameters()
        self.enc.lin3.reset_parameters()
        if self.optimizer is not None and hasattr(self, 'parameters'):
            self.optimizer = torch.optim.Adam(self.parameters(), self.optimizer.param_groups[0]['lr'])


class DecodingRecommender(Recommender):
    """ Only the decoder part of the AAE, basically 2-MLP """
    ### DONE Adapt to generic condition ###
    def __init__(self, conditions, n_epochs=100, batch_size=100, optimizer='adam',
                 n_hidden=100, lr=0.001, verbose=True, **mlp_params):
        ### DONE Adapt to generic condition ###
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.model_params = mlp_params
        self.verbose = verbose
        self.n_hidden = n_hidden
        assert len(conditions), "Minimum 1 condition is necessary for MLP"
        self.conditions = conditions

        self.mlp, self.mlp_optim, self.vect = None, None, None

    def __str__(self):
        ### DONE Adapt to generic condition ###
        desc = "MLP-2 Decoder with " + str(self.n_hidden) + " hidden units"
        desc += " training for " + str(self.n_epochs)
        desc += " optimized by " + self.optimizer
        desc += " with learning rate " + str(self.lr)
        desc += " with %d conditions: %s " % (len(self.conditions), ', '.join(self.conditions.keys()))
        desc += "\n MLP Params: " + str(self.model_params)
        return desc

    def partial_fit(self, condition_data, y):
        ### DONE Adapt to generic condition ###
        self.mlp.train()
        self.conditions.train()
        # Encode ALL condition data with respective condition
        encoded_cdata = self.conditions.encode(condition_data)
        remaining_conditions = list(self.conditions.values())[1:]
        # Start with first encoded condition (since we need one)
        inputs = encoded_cdata[0]
        if remaining_conditions:
            for cond, cdata in zip(remaining_conditions, encoded_cdata[1:]):
                # Impose all remaining conditions
                inputs = cond.impose(inputs, cdata)


        if torch.cuda.is_available():
            inputs, y = inputs.cuda(), y.cuda()
        y_pred = self.mlp(inputs)
        y_pred = torch.nan_to_num(y_pred)
        loss = F.binary_cross_entropy(y_pred + TINY, y + TINY)
        self.mlp_optim.zero_grad()
        self.conditions.zero_grad()
        loss.backward()
        self.mlp_optim.step()
        self.conditions.step()
        if self.verbose:
            print("\rLoss: {}".format(loss.data.item()), flush=True, end='')
        return self

    def fit(self, condition_data, Y):
        ### DONE Adapt to generic condition ###
        self.mlp = Decoder(self.conditions.size_increment(),
                           self.n_hidden,
                           Y.shape[1],
                           **self.model_params)
        if torch.cuda.is_available():
            self.mlp = self.mlp.cuda()
        optimizer_cls = TORCH_OPTIMIZERS[self.optimizer]
        self.mlp_optim = optimizer_cls(self.mlp.parameters(), lr=self.lr)
        for __epoch in range(self.n_epochs):
            Y_shuf, *condition_data_shuf = sklearn.utils.shuffle(Y, *condition_data)
            for start in range(0, Y.shape[0], self.batch_size):
                end = start + self.batch_size
                Y_batch = Y_shuf[start:end]
                C_batch = [c[start:end] for c in condition_data_shuf]
                if sp.issparse(Y_batch):
                    Y_batch = Y_batch.toarray()
                self.partial_fit(C_batch, torch.FloatTensor(Y_batch))

            if self.verbose:
                print()

        return self

    def train(self, training_set):
        ### DONE Adapt to generic condition ###
        # Fit function from condition to X
        Y = training_set.tocsr()
        condition_data_raw = training_set.get_attributes(self.conditions.keys())
        condition_data = self.conditions.fit_transform(condition_data_raw)
        self.fit(condition_data, Y)


    def predict(self, test_set):
        ### DONE Adapt to generic condition ###
        n_users = test_set.size(0)
        condition_data_raw = test_set.get_attributes(self.conditions.keys())
        condition_data = self.conditions.transform(condition_data_raw)
        self.mlp.eval()
        self.conditions.eval()
        batch_results = []
        with torch.no_grad():
            for start in range(0, n_users, self.batch_size):
                end = start + self.batch_size
                c_batch = [c[start:end] for c in condition_data]
                encoded_cdata = self.conditions.encode(c_batch)
                remaining_conditions = list(self.conditions.values())[1:]
                # Start with first encoded condition (since we need one)
                inputs = encoded_cdata[0]
                if remaining_conditions:
                    for cond, cdata in zip(remaining_conditions, encoded_cdata[1:]):
                        # Impose all remaining conditions
                        inputs = cond.impose(inputs, cdata)

                # Shift data to gpu
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                res = self.mlp(inputs)
                # Shift results back to cpu
                batch_results.append(res.cpu().numpy())
        
        y_pred = np.vstack(batch_results)
        return y_pred




class AdversarialAutoEncoder(AutoEncoderMixin):
    """ Adversarial Autoencoder """
    ### DONE Adapt to generic condition ###
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 gen_lr=0.001,
                 reg_lr=0.001,
                 prior='gauss',
                 prior_scale=None,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2, .2),
                 conditions=None,
                 verbose=True):
        # Build models
        self.prior = prior.lower()
        self.prior_scale = prior_scale

        # Encoder final activation depends on prior distribution
        self.prior_sampler = PRIOR_SAMPLERS[self.prior]
        self.encoder_activation = PRIOR_ACTIVATIONS[self.prior]
        self.optimizer = optimizer.lower()

        self.n_hidden = n_hidden
        self.n_code = n_code
        self.gen_lr = gen_lr
        self.reg_lr = reg_lr
        self.batch_size = batch_size
        self.verbose = verbose
        self.gen_lr, self.reg_lr = gen_lr, reg_lr
        self.n_epochs = n_epochs

        self.enc, self.dec, self.disc = None, None, None
        self.enc_optim, self.dec_optim = None, None
        self.gen_optim, self.disc_optim = None, None
        self.normalize_inputs = normalize_inputs

        self.dropout = dropout
        self.activation = activation

        self.conditions = conditions

    def __str__(self):
        desc = "Adversarial Autoencoder"
        n_h, n_c = self.n_hidden, self.n_code
        gen, reg = self.gen_lr, self.reg_lr
        desc += " ({}, {}, {}, {}, {})".format(n_h, n_h, n_c, n_h, n_h)
        desc += " optimized by " + self.optimizer
        desc += " with learning rates Gen, Reg = {}, {}".format(gen, reg)
        desc += ", using a batch size of {}".format(self.batch_size)
        desc += "\nMatching the {} distribution".format(self.prior)
        desc += " by {} activation.".format(self.encoder_activation)
        if self.conditions:
            desc += "\nConditioned on " + ', '.join(self.conditions.keys())
        return desc

    def eval(self):
        """ Put all NN modules into eval mode """
        ### DONE Adapt to generic condition ###
        self.enc.eval()
        self.dec.eval()
        self.disc.eval()
        if self.conditions:
            # Forward call to condition modules
            self.conditions.eval()

    def train(self):
        """ Put all NN modules into train mode """
        ### DONE Adapt to generic condition ###
        self.enc.train()
        self.dec.train()
        self.disc.train()
        if self.conditions:
            # Forward call to condition modules
            self.conditions.train()

    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.enc.zero_grad()
        self.dec.zero_grad()
        self.disc.zero_grad()

    def reset_parameters(self):
        if self is not None:
            attrs_to_call = [getattr(self, attr) for attr in dir(self) if not attr.startswith("__") and
                             hasattr(getattr(self, attr), 'reset_parameters')]
            for attr in attrs_to_call:
                attr.reset_parameters()

            self.disc.lin1.reset_parameters()
            self.disc.lin2.reset_parameters()
            self.disc.lin3.reset_parameters()
            self.disc.lin1.reset_parameters()
            self.disc.lin2.reset_parameters()
            self.disc.lin3.reset_parameters()

            self.dec.lin1.reset_parameters()
            self.dec.lin2.reset_parameters()
            self.dec.lin3.reset_parameters()
            self.dec.lin1.reset_parameters()
            self.dec.lin2.reset_parameters()
            self.dec.lin3.reset_parameters()

            self.enc.lin1.reset_parameters()
            self.enc.lin2.reset_parameters()
            self.enc.lin3.reset_parameters()
            self.enc.lin1.reset_parameters()
            self.enc.lin2.reset_parameters()
            self.enc.lin3.reset_parameters()

        if self.optimizer is not None and hasattr(self, 'parameters'):
            self.optimizer = torch.optim.Adam(self.parameters(), self.optimizer.param_groups[0]['lr'])

    def ae_step(self, batch, condition_data=None):
        z_sample = self.enc(batch)

        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z_sample = self.conditions.encode_impose(z_sample, condition_data)

        x_sample = self.dec(z_sample)
        x_sample = torch.nan_to_num(x_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)

        self.enc.zero_grad()
        self.dec.zero_grad()
        if use_condition:
            self.conditions.zero_grad()
    
        # Compute gradients
        recon_loss.backward()

        # Update parameters
        self.enc_optim.step()
        self.dec_optim.step()
        if use_condition:
            self.conditions.step()

        return recon_loss.data.item()

    def disc_step(self, batch):
        """ Perform one discriminator step on batch """
        self.enc.eval()

        z_real = Variable(self.prior_sampler((batch.size(0), self.n_code)))
        if self.prior_scale is not None:
            z_real = z_real * self.prior_scale
        if torch.cuda.is_available():
            z_real = z_real.cuda()
        z_fake = self.enc(batch)
        # Compute discrimnator outputs and loss
        disc_real_out, disc_fake_out = self.disc(z_real), self.disc(z_fake)
        disc_loss = -torch.mean(torch.log(disc_real_out + TINY)
                                + torch.log(1 - disc_fake_out + TINY))
        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        return disc_loss.data.item()

    def gen_step(self, batch):
        self.enc.train()
        z_fake_dist = self.enc(batch)
        disc_fake_out = self.disc(z_fake_dist)
        gen_loss = -torch.mean(torch.log(disc_fake_out + TINY))
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()

        return gen_loss.data.item()

    def partial_fit(self, X, y=None, condition_data=None):
        """ Performs reconstrction, discimination, generator training steps """
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = torch.FloatTensor(X)
        if torch.cuda.is_available():
            # Put batch on CUDA device!
            X = X.cuda()
        # Make sure we are in training mode and zero leftover gradients
        self.train()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition_data=condition_data)
        disc_loss = self.disc_step(X)
        gen_loss = self.gen_step(X)
        if self.verbose:
            log_losses(recon_loss, disc_loss, gen_loss)
        return self

    def fit(self, X, y=None, condition_data=None):
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        if use_condition:
            code_size = self.n_code + self.conditions.size_increment()
            print("Using condition, code size:", code_size)
        else:
            code_size = self.n_code
            print("Not using condition, code size:", code_size)

        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation=self.encoder_activation,
                           normalize_inputs=self.normalize_inputs,
                           activation=self.activation,
                           dropout=self.dropout)
        self.dec = Decoder(code_size, self.n_hidden, X.shape[1],
                           activation=self.activation, dropout=self.dropout)

        self.disc = Discriminator(self.n_code, self.n_hidden,
                                  dropout=self.dropout,
                                  activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.disc = self.disc.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.gen_lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.gen_lr)
        # Regularization
        self.gen_optim = optimizer_gen(self.enc.parameters(), lr=self.reg_lr)
        self.disc_optim = optimizer_gen(self.disc.parameters(), lr=self.reg_lr)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            # Shuffle on each new epoch
            if use_condition:
                # shuffle(*arrays) takes several arrays and shuffles them so indices are still matching
                X_shuf, *condition_data_shuf = sklearn.utils.shuffle(X, *condition_data)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X_shuf.shape[0], self.batch_size):
                end = start + self.batch_size

                # Make the batch dense!
                X_batch = X_shuf[start:end].toarray()

                # condition may be None
                if use_condition:
                    # c_batch = condition_shuf[start:(start+self.batch_size)]
                    c_batch = [c[start:end] for c in condition_data_shuf]
                    self.partial_fit(X_batch, condition_data=c_batch)
                else:
                    self.partial_fit(X_batch)

            if self.verbose:
                # Clean up after flushing batch loss printings
                print()
        return self


    def predict(self, X, condition_data=None):
        self.eval()  # Deactivate dropout
        # In case some of the conditions has dropout
        use_condition = _check_conditions(self.conditions, condition_data)
        if self.conditions:
            self.conditions.eval()
        pred = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                # batched predictions, yet inclusive
                X_batch = X[start:(start+self.batch_size)]
                if sp.issparse(X_batch):
                    X_batch = X_batch.toarray()
                X_batch = torch.FloatTensor(X_batch)
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()

                if use_condition:
                    c_batch = [c[start:end] for c in condition_data]
                # reconstruct
                z = self.enc(X_batch)
                if use_condition:
                    z = self.conditions.encode_impose(z, c_batch)
                X_reconstuction = self.dec(z)
                # shift
                X_reconstuction = X_reconstuction.data.cpu().numpy()
                pred.append(X_reconstuction)
        return np.vstack(pred)


class AAERecommender(Recommender):
    """
    Adversarially Regularized Recommender
    =====================================

    Arguments
    ---------
    n_input: Dimension of input to expect
    n_hidden: Dimension for hidden layers
    n_code: Code Dimension

    Keyword Arguments
    -----------------
    n_epochs: Number of epochs to train
    batch_size: Batch size to use for training
    verbose: Print losses during training
    normalize_inputs: Whether l1-normalization is performed on the input
    """
    def __init__(self, adversarial=True, conditions=None, **kwargs):
        ### DONE Adapt to generic condition ###
        """ tfidf_params get piped to either TfidfVectorizer or
        EmbeddedVectorizer.  Remaining kwargs get passed to
        AdversarialAutoencoder """
        super().__init__()
        self.verbose = kwargs.get('verbose', True)

        # self.use_side_info = kwargs.pop('use_side_info', False)

        self.conditions = conditions

        # assert_condition_callabilities(self.use_side_info)
        # Embedding is now part of a condition
        # self.embedding = kwargs.pop('embedding', None)
        # Vectorizer also...
        # self.vect = None
        self.model_params = kwargs
        # tfidf params now need to be in the respective *condition* of condition_list
        # self.tfidf_params = tfidf_params
        self.adversarial = adversarial

    def __str__(self):
        ### DONE Adapt to generic condition ###
        if self.adversarial:
            desc = "Adversarial Autoencoder"
        else:
            desc = "Autoencoder"

        if self.conditions:
            desc += " conditioned on: " + ', '.join(self.conditions.keys())
        desc += '\nModel Params: ' + str(self.model_params)
        # TODO: is it correct for self.tfidf_params to be an EMPTY dict
        # DONE: Yes it is only the *default*!
        # desc += '\nTfidf Params: ' + str(self.tfidf_params)
        # Anyways, this kind of stuff goes into the condition itself
        return desc


    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                copy.deepcopy(self.adversarial, memo),
                copy.deepcopy(self.conditions, memo))
            _copy.model_params = self.model_params
            _copy.verbose = self.verbose
            memo[id_self] = _copy
        return _copy

    def train(self, training_set):
        """
        1. get basic representation
        2. ? add potential side_info in ??? representation
        3. initialize a (Adversarial) Autoencoder variant
        4. fit based on Autoencoder
        :param training_set: ???, Bag Class training set
        :return: trained self
        """
        X = training_set.tocsr()
        if self.conditions:
            condition_data_raw = training_set.get_attributes(self.conditions.keys())
            condition_data = self.conditions.fit_transform(condition_data_raw)
        else:
            condition_data = None

        if self.adversarial:
            # Pass conditions through along with hyperparams
            self.model = AdversarialAutoEncoder(conditions=self.conditions, **self.model_params)
        else:
            # Pass conditions through along with hyperparams!
            self.model = AutoEncoder(conditions=self.conditions, **self.model_params)

        print(self)
        print(self.model)
        print(self.conditions)

        # gives (Adversarial) Autoencoder BaseData (--> X: <???> representation) and side_info (attr_vect: numpy)
        self.model.fit(X, condition_data=condition_data)

    def predict(self, test_set):
        ### DONE Adapt to generic condition ###
        X = test_set.tocsr()
        if self.conditions:
            condition_data_raw = test_set.get_attributes(self.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = self.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        pred = self.model.predict(X, condition_data=condition_data)

        return pred
    def zero_grad(self):
        """ Zeros gradients of all NN modules """
        self.model.zero_grad()
    def reset_parameters(self):
        if hasattr(self, 'model') and self.model is not None:
            self.model.reset_parameters()
        if hasattr(self, 'conditions') and self.conditions is not None:
            self.conditions.reset_parameters()
