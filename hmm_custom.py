import numpy as np
import string
import logging
from scipy.stats import poisson
from sklearn.utils import check_random_state
#from sklearn.mixture import (
    #GMM, sample_gaussian,
    #log_multivariate_normal_density,
    #distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn import cluster
from hmmlearn.base import _BaseHMM
from scipy.stats import nbinom
import hmmlearn
import copy


# Copied from scikit-learn 0.19.
def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values."""
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")

#!/usr/bin/python

import copy


#from scipy.stats import nbinom
from scipy.special import digamma



# dl / dr = sum_t digamma(x_t + r) * post - sum_t digamma(r) * post + sum_t ln(p) * post
# dl / dr = sum_t post * (digamma(x_t + r) - digamma(r) + ln(p))

def calculate_derivative(posteriors, data, r, p):
    def _digamma(array):
        return digamma(array.astype("float64")).astype("float128")
    n_comp, n_var = r.shape
    n_obs, _ = data.shape

    derivative = np.zeros(r.shape)

    for state in range(n_comp):
        r_j, p_j = r[state], p[state]
        posteriors_j = posteriors[:, state][:, np.newaxis]
        in_brackets = _digamma(data + r_j) - _digamma(r_j) + np.log(p_j)
        derivative[state] = np.sum(posteriors_j * in_brackets, axis=0)

    #print("derivative:")
    #print(derivative)
    return derivative

    # it's the same, just different implementation
    n_comp, n_var = r.shape
    n_obs, _ = data.shape
    desired_shape = (n_comp, n_obs, n_var)
    data_repeated = np.concatenate([data] * n_comp).reshape(desired_shape)
    r_repeated = np.repeat([r], n_obs, axis=1).reshape(desired_shape)
    digamma_of_sum = _digamma(data_repeated + r_repeated)
    digamma_of_r = _digamma(r_repeated)
    log_p = np.log(p)
    log_p_repeated = np.repeat([log_p], n_obs, axis=1).reshape(desired_shape)
    sum_ = digamma_of_sum - digamma_of_r + log_p_repeated
    posteriors_repeated = np.repeat(posteriors.T, n_var).reshape(desired_shape)
    product = posteriors_repeated * sum_
    derivative = np.sum(product, axis=1)
    return derivative

    # another implementation
    r_conc = np.concatenate([r] * n_obs, axis=0)
    r_conc = r_conc.reshape(n_obs, n_comp, n_var)
    X_repeat = np.repeat(data, n_comp, axis=0)
    X_repeat = X_repeat.reshape(n_obs, n_comp, n_var)
    suma = X_repeat + r_conc
    suma = suma.reshape(n_obs, n_comp, n_var)
    pstwa_repeat = np.repeat(posteriors, n_var)
    pstwa_repeat = pstwa_repeat.reshape(n_obs, n_comp, n_var)
    a = np.sum(pstwa_repeat * _digamma(suma), axis=0)
    a = a.reshape(n_comp, n_var)
    b = np.sum(pstwa_repeat * _digamma(r_conc), axis=0)
    b = b.reshape(n_comp, n_var)
    p_conc = np.concatenate([p] * n_obs, axis=0)
    p_conc = p_conc.reshape(n_obs, n_comp, n_var)
    c = np.sum(pstwa_repeat * np.log(p_conc), axis=0)
    c = c.reshape(n_comp, n_var)
    derivative = a - b + c
    return derivative

def update_r(r, derivative, delta, stop):
    # mozna by jakos sprytniej skakac
    # np teraz mam ciag ktory skacze od kilkudziesieciu milionow
    #  na plusie i minusie (dwa razy minus, raz plus, i tak w kolko),
    #  zakres sie zaciesnia, ale powoli
    #  moze mozna by jakos uzaleznic od tego na ile duza co do modulu
    #  byla poprzednia i jeszcze poprzednia pochodna
    # albo jakos wazyc rejony czy co
    # najgorzej jak np jest caly czas ta sama dodatnia
    #  przeplatana dwoma ujemnymi, coraz blizszymi zera, ale wciaz odleglymi
    #  czlowiek by zobaczyl ze mozna tu skakac szybciej,
    #  zamiast ladowac w tym samym miejscu
    #  wiec powinno sie to tez dac zaimplementowac...
    # inna sprawa ze dla poczatkowych iteracji,
    #  kiedy estymacja p jest tez bardzo zgrubna,
    #  nie potrzebuje chyba bardzo precyzyjnej estymacji r;
    #  wystarczy mi nieduza pochodna, niekoniecznie bardzo bliska zero.
    #  Ale to juz bardziej skomplikowane do zaimplementowania.
    n_comp, n_var = r.shape
    for i in range(n_comp):
        for j in range(n_var):
            if stop[i, j] is True:
                continue
            if abs(derivative[i, j]) <= 1e-10:
                continue
            if delta[i, j] == 0:
                if derivative[i, j] < 0:
                    delta[i, j] = r[i, j] * -0.5
                elif derivative[i, j] > 0:
                    delta[i, j] = r[i, j]
                else:
                    print("cos nie tak, pewnie nan")
            elif delta[i, j] < 0:
                if derivative[i, j] < 0:
                    pass
                elif derivative[i, j] > 0:
                    delta[i, j] *= -0.5
                else:
                    print("cos nie tak, pewnie nan")
            elif delta[i, j] > 0:
                if derivative[i, j] < 0:
                    delta[i, j] *= -0.5
                elif derivative[i, j] > 0:
                    pass
                else:
                    print("cos nie tak, pewnie nan")
            if r[i, j] + delta[i, j] <= 0:
                delta[i, j] = -0.5 * r[i, j]
            r[i, j] += delta[i, j]
            if r[i, j] <= 0:
                if derivative[i, j] < 0:
                    logging.warning("r mle < 0, derivative < 0")
                    stop[i, j] = True
                r[i, j] = 0.0001
                delta[i, j] = 0
    return r, delta, stop

def find_r(r_initial, dane, pstwa, p, threshold=1e-3):
    r = r_initial.copy()
    r_not_found = True
    counter = 0
    delta = np.zeros(shape=r.shape)
    stop = np.zeros(r.shape, dtype=bool)
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, copy.deepcopy(r), p)
        #print(derivative)
        if np.any(np.isnan(derivative)):
            print("Derivative is nan, stop this madness")
            print("That's the r, p and derivative:")
            print(r)
            print(p)
            print(derivative)
            break
        if np.all((abs(derivative) < threshold) + stop):
            r_not_found = False
            break
        stop[abs(derivative) < threshold] = True
        r, delta, stop = update_r(copy.deepcopy(r), derivative, delta, stop)
        counter += 1
        #if counter % 10 == 0:
        #    print("%i iterations" % counter)
        #    print(derivative)
        #    print(r)
        if counter == 2000:
            # jesli idzie tak dlugo to pewnie i tak cos jest nie tak.
            # to niech juz beda te estymacje ktore sa.
            break
    #print("r estimated:")
    #print(r)
    #print("***")
    return r

# Copied from scikit-learn 0.19.
def distribute_covar_matrix_to_match_covariance_type(
        tied_cv, covariance_type, n_components):
    """Create all the covariance matrices from a given template."""
    if covariance_type == 'spherical':
        cv = np.tile(tied_cv.mean() * np.ones(tied_cv.shape[1]),
                     (n_components, 1))
    elif covariance_type == 'tied':
        cv = tied_cv
    elif covariance_type == 'diag':
        cv = np.tile(np.diag(tied_cv), (n_components, 1))
    elif covariance_type == 'full':
        cv = np.tile(tied_cv, (n_components, 1, 1))
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
    return cv

def array2str(array):
    """
    Changes an array to a readable string.
    [1, 2, 3] -> 1\t2\t3

    [[1,2], [3,4]]
    ->
    1\t2
    3\t4

    [[[1,2], [3,4]], [[5,6], [7,8]], [[9, 10], [11,12]]]
    ->
    1\t2
    3\t4

    5\t6
    7\t8

    9\t10
    11\t12

    Returns:
        resulting formatted string.
    """
    if type(array) == list:
        array = np.array(array)
    if type(array) != np._ndarray:
        return str(array)
    if len(array.shape) == 1:
        result = '\t'.join([array2str(element) for element in array])
    elif len(array.shape) > 1:
        result = ""
        for line in array:
            result += array2str(line)
            result += "\n"
    return result


class NegativeBinomialHMM(_BaseHMM):

    """
    Two possible notations:
    #####
    (I)
    p - probability of success
    r - number of failures
    X ~ NB(r, p) - number of successes before r failures occures
    mean(X) = rp / (1-p)
    var(X) = rp / (1-p)**2
    p = (var - mean) / var
    r = mean**2 / (var - mean)
    That's notation from wikipedia.
    #####
    (II)
    p - probability of success
    r - number of successes
    X ~ NB(r, p) - number of failures before r successes occures
    mean(X) = r(1-p) / p
    var(X) = r(1-p) / p**2
    p = mean / var
    r = mean**2 / (var - mean)
    That's notation used in scipy.stats.nbinom and R.
    #####
    These notations can be transformed easily into each other by setting p := 1-p.
    Current formulas assume notation (II).
    If you plan on changing that,
    find all methods with #NOTATION tag.
    """

    def __init__(self, n_components,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters
                 ):
        _BaseHMM.__init__(self,
                          n_components=n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol,
                          verbose=verbose,
                          params=params, init_params=init_params
                          )
        self._update_r_ = True
        self._update_p_ = False

    def _init(self, X, lengths=None):
        super(NegativeBinomialHMM, self)._init(X, lengths)
        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
        self.n_features = n_features
        if any([letter in self.init_params for letter in ['m', 'c', 'p', 'r']]):
            # estimate means, covars; calculate p, r
            means = self._estimate_means(X)
            covars = self._estimate_covars(X)
        if 'm' in self.init_params or not hasattr(self, "means_"):
            self.means_ = means
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            self.covars_ = covars
        p, r = self._calculate_p_r(self.means_, self.covars_)
        if 'p' in self.init_params or not hasattr(self, "p_"):
            self.p_ = p
        if 'r' in self.init_params or not hasattr(self, "r_"):
            self.r_ = r
        '''
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))
             with open("%sr_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.r_))
             with open("%sp_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.p_))
        '''
    def _estimate_means(self, X):
        """
        Estimate means with k-means.
        Based on GaussianHMM.
        I'm not sure if it's the best way,
         maybe simple quantiles would be better?
         But it's ready, so I'll go with it for now.
        Ok, it's actually not used. I leave it just in case.
        """
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
        kmeans.fit(X)
        means = kmeans.cluster_centers_
        means = np.sort(means, axis=0)
        return means

    def _estimate_covars(self, X):
        """
        Estimate covars.
        Based on GaussianHMM.
        I'm not sure what covariance type is appropriate here,
         so I went with diag.
        min_covar is the default value for GaussianHMM self.min_covar.
        """
        min_covar = 1e-3
        cv = np.cov(X.T) + min_covar * np.eye(X.shape[1])
        if not cv.shape:
            cv.shape = (1, 1)
        covars = \
            distribute_covar_matrix_to_match_covariance_type(
                cv, 'diag', self.n_components).copy()
        return covars

        """
        From GaussianHMM:
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            means = kmeans.cluster_centers_
            means = np.sort(means, axis = 0)
            self.means_ = means
        logging.debug("Initial means:")
        logging.debug(self.means_)
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()
        """

    def _calculate_p_r(self, means=None, covars=None):
        """
        Calculate p and r parameters from means and covars estimations.
        If covars < means, calculated p and r make no sense.
        We need to correct it, so that 0 < p < 1 and r > 0.
        If indeed covars < means for every state MLE cannot be obtained,
        but it might be that it happened just in the initialisation step,
        and after an iteration it won't happen.
        So no need to raise alarm here, I guess.
        #NOTATION
        To see current notation go to class description.
        """
        if means is None:
            means = self.means_
        if covars is None:
            covars = self.covars_
        #p = (covars - means) / covars
        p = means / covars
        r = means ** 2 / (covars - means)
        if np.any(p > 1):
            p[p > 1] = 0.99
        if np.any(p <= 0):
            p[p <= 0] = 0.000001
        if np.any(r <= 0):
            r[r <= 0] = 0.001
        return p, r

    def _calculate_means_covars(self, p=None, r=None):
        """
        Calculate means and covars from p and r EM estimations.
        #NOTATION
        To see current notation go to class description.
        """
        if p is None:
            p = self.p_
        if r is None:
            r = self.r_
        #means = r * p / (1 - p)
        #covars = r * p / (1 - p)**2
        means = r * (1-p) / p
        covars = r * (1-p) / p**2
        return means, covars

    def _check(self):
        super(NegativeBinomialHMM, self)._check()
        # maybe we could check here whether variance > mean?
        # though I'm not sure if it can be checked a priori

    def _compute_log_likelihood(self, X):
        """
        #NOTATION
        To see current notation go to class description.
        """
        def _logpmf(X, r, p):
            # jesli musze zmieniac tu typ na 64, to czy w ogole jest jakis zysk z uzywania 128?
            return nbinom.logpmf(X.astype('int'), r.astype('float64'),
                                 p.astype('float64')).astype('float128')
        n_observations, n_features = X.shape
        log_likelihood = np.ndarray((n_observations, self.n_components))
        for observation in range(n_observations):
            for state in range(self.n_components):
                log_likelihood[observation, state] = \
                    np.sum(_logpmf(X[observation, :],
                                   self.r_[state, :], self.p_[state, :]))
        return log_likelihood

    def _generate_sample_from_state(self, state, random_state=None):
        """
        #NOTATION
        To see current notation go to class description.
        """
        random_state = check_random_state(random_state)
        return [nbinom(self.r_[state][feature],
                       self.p_[state][feature]).rvs()
                for feature in range(self.n_features)]

    def _initialize_sufficient_statistics(self):
        stats = super(NegativeBinomialHMM, self)._initialize_sufficient_statistics()
        # observations
        stats['X'] = np.ndarray((0, self.n_features))
        # sum_t posteriors * observations
        # I leave name 'obs' for consistency with Gaussian
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        # sum_t posteriors
        # I leave name 'post' for consistency with Gaussian
        stats['post'] = np.zeros(self.n_components)
        # posteriors
        stats['posteriors'] = np.ndarray((0, self.n_components))

        #stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        #if self.covariance_type in ('tied', 'full'):
        #    stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
        #                                   self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(NegativeBinomialHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        stats['X'] = np.append(stats['X'], X, axis=0)
        stats['post'] += posteriors.sum(axis=0)
        stats['obs'] += np.dot(posteriors.T, X)
        stats['posteriors'] = np.append(stats['posteriors'], posteriors, axis=0)
        """
        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)
        """

    def _do_mstep(self, stats):
        logging.debug("Start of m step; p, r, means, covars:")
        logging.debug(self.p_)
        logging.debug(self.r_)
        logging.debug(self.means_)
        logging.debug(self.covars_)
        super(NegativeBinomialHMM, self)._do_mstep(stats)
        # update:
        #self.p_, self.r_, self.means_, self.covars_
        if np.any((stats['post']) == 0):
            raise RuntimeError("stats['post'] has zeros."
                               " It means for at least one state the following is true:"
                               " for every window, there is zero posterior probability"
                               " of this window being in this state."
                               " It might be needed to lower number of states."
                               " Maybe different initial means will help."
                               " No reason to continue now,"
                               " from here only errors await."
                               " Here are current values of some parameters,"
                               " for debugging / replicating purposes:\n"
                               " means:\n%s\n covars:\n%s\n p:\n%s\n r:\n%s\n"
                               " stats['obs'] (posteriors.T * X):\n%s\n"
                               " stats['post'] (sum posteriors):\n%s\n"
                               " Aaand we finish here. Bye."
                               % (str(self.means_),
                                  str(self.covars_),
                                  str(self.p_),
                                  str(self.r_),
                                  str(stats['obs']),
                                  str(stats['post'])))
        if np.any((stats['obs']) == 0):
            logging.warning("stats['obs'] has zeros."
                            "Probably sth will go wrong now."
                            " Try different initial means"
                            " or lower number of states.")
        if self._update_p_:
            self.p_ = self._update_p(stats)
            self._update_p_ = False
            self._update_r_ = True
        elif self._update_r_:
            self.r_ = self._update_r(stats)
            self._update_p_ = True
            self._update_r_ = False
        self.means_, self.covars_ = self._calculate_means_covars()
        logging.debug("End of m step; p, r, means, covars:")
        logging.debug(self.p_)
        logging.debug(self.r_)
        logging.debug(self.means_)
        logging.debug(self.covars_)
        '''
        if self.debug_prefix is not None:
             with open("%smeans_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.means_))
             with open("%scovars_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.covars_))
             with open("%sr_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.r_))
             with open("%sp_%d" %
                       (self.debug_prefix, self.iteration), 'w') as output:
                output.write(array2str(self.p_))
        '''

    def _update_p(self, stats):
        """
        ML estimation of p parameter.
        #NOTATION
        To see current notation go to class description.
        p_j = sum_t (posteriori_j,t * r_j) / sum_t posteriori_j_t(x_t + r_j)
        """
        post_times_r = stats['post'][:, np.newaxis] * self.r_
        #print(stats['post'].shape)
        #print(self.r_.shape)
        #print(stats['obs'].shape)
        #print((stats['post']*self.r_).shape)
        #print(post_times_r.shape)
        p_mle = post_times_r / (stats['obs'] + post_times_r)
        if np.any(np.isnan(p_mle)):
            print("Warning: p MLE is nan")
            if np.any(np.isnan(post_times_r)):
                print("...specifically post_times_r")
            if np.any(np.isnan(stats['obs'])):
                print("...specifically stats['obs']")
            if np.any((stats['obs'] + post_times_r) == 0):
                print("stats obs + post_times_r ma zera:")
                print(stats['obs'] + post_times_r)
                print("stats obs, post times r osobno:")
                print(stats['obs'])
                print(post_times_r)
                print("stats post:")
                print(stats['post'])

        if np.any(p_mle > 1):
            print("Warning: your p MLE is bigger than 1")
            p_mle[p_mle > 1] = 0.99
        if np.any(p_mle < 0):
            print("Warning: your p MLE is smaller than 0")
            p_mle[p_mle < 0] = 0.01
        #X = stats['X']
        #n_samples = X.shape[0]
        #X_sum = np.sum(X, axis=0)
        #p_mle = X_sum / (n_samples * self.r_ + X_sum)
        #p_mle = n_samples * self.r_ / (n_samples * self.r_ + X_sum)
        logging.debug("p MLE:")
        logging.debug(p_mle)
        if p_mle.shape != self.p_.shape:
            raise ValueError('p MLE has different shape than p in previous iteration.'
                             ' Expected %s, got %s.'
                             ' Check your MLE calculations.'
                             % (str(self.p_.shape), str(p_mle.shape)))
        return p_mle

    def _update_r(self, stats):
        """
        ML estimation of r parameter.
        #NOTATION
        To see current notation go to class description.
        """
        r_mle = self.r_
        r_mle = find_r(self.r_, stats['X'], stats['posteriors'], self.p_)
        if np.any(r_mle < 0):
            print("Warning: your r MLE is smaller than 0")
            r_mle[r_mle < 0] = 0.5
        logging.debug("r MLE:")
        logging.debug(r_mle)
        if r_mle.shape != self.r_.shape:
            raise ValueError('r MLE has different shape than r in previous iteration.'
                             ' Expected %s, got %s.'
                             ' Check your MLE calculations.'
                             % (str(self.r_.shape), str(r_mle.shape)))
        return r_mle

'''
model = NegativeBinomialHMM(n_components=2)
X = np.array([1,23,1,1,1,3,5,7,9,1,2,3,4])
X = X.reshape(-1,1)
model = model.fit(X)
print(model.decode(X))
'''
