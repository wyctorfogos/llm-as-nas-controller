
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
from itertools import combinations


def statistical_test(data, alg_names, pv_friedman=0.05, pv_wilcoxon=0.05, verbose=True):
    """
    This function performs the Friedman's test followd by the Wilcoxon's test (if applicable)

    Parameters
    ----------
    data: np.ndarray
        The  algorithms results to perform the tests. The order is `algorithms x samples`. For example, a 3 x 10 matrix
        contains 3 algorithms with 10 results for each of them
    alg_names: list
        A list with algorithms names. Ex: n = ['Alg1', 'Alg2', 'Alg3']. Obviously, it must match with the number of
        rows in data
    pv_friedman: float
        The p value reference for the Friedman test. If it's None, the test is not performed
    pv_wilcoxon: float
        The p value reference for the Wilcoxon test. If it's None, the test is not performed
    verbose: boolean
        Set it as true to print the result on the screen

    Return
    ------
    out: string
        A string describing the result of the test
    """

    data = np.asarray(data)
    _perform_wilcoxon = True
    if data.shape[0] != len(alg_names):
        raise Exception('data and alg_names rows must have the same length')

    out = '-' * 50
    out += '\nStatistical test\n'
    out += '-' * 50
    if pv_friedman is not None:
        out += f'\n- Performing the Friedman\'s test with pv_ref = {pv_friedman} ...'
        s, pv = friedmanchisquare(*[data[i, :] for i in range(data.shape[0])])
        out += '- p_value = ' + str(np.round(pv, 6)) + '\n'

        if pv > pv_friedman:
            out += '- There is no need to perform pairwise comparison because pv > pv_friedman'
            _perform_wilcoxon = False
        else:
            out += '- There is at least one algorithm that is statistically different from the others'
        out += '\n'

    if _perform_wilcoxon and pv_wilcoxon is not None:
        out += f'\n- Performing the Wilcoxon\'s test with pv_ref = {pv_wilcoxon} ...\n'
        combs = list(combinations(range(data.shape[0]), 2))
        for c in combs:
            s, pv = wilcoxon(data[c[0], :], data[c[1], :])
            out += '-- Comparing ' + alg_names[c[0]] + ' - ' + alg_names[
                c[1]] + ': p_value = ' + str(np.round(pv, 9))
            if pv < pv_wilcoxon:
                out += ' | They are statistically different!'
            else:
                out += ' | They are NOT statistically different!'
            out += '\n'

    out += '-' * 50
    out += '\n'

    if verbose:
        print(out)

    return out


class AVGMetrics(object):
    """
    This is a simple class to control the average for a given value. It's useful to control loss and accuracy for a
    mini-batch during the training phase. Essentially, it keeps track of a given value and compute the average when the
    __call__ method is called.
    """

    def __init__(self):
        """
        It starts the method's attributes
        """
        self.sum_value = 0
        self.avg = 0
        self.count = 0
        self.values = list()

    def __call__(self):
        """
        It returns the average when the method is called
        """
        return self.avg

    def std(self):
        """
            It returns the std
        """
        if len(self.values) > 0:
            return np.std(self.values)
        else:
            return 0

    def update(self, val):
        """
        Updates the attributes according to the given value
        """
        self.sum_value += val
        self.count += 1
        self.avg = self.sum_value / float(self.count)
        self.values.append(val)

    def print(self, title='Summary'):
        """
        It prints the class' attributes
        """
        print("-" * 50)
        print(f"- {title}")
        print("-" * 50)
        print('- MEDIAN: {:.3f}'.format(np.median(self.values)))
        print('- AVG: {:.3f}'.format(self.avg))
        print('- STD: {:.3f}'.format(np.std(self.values)))
        print('- MAX: {:.3f}'.format(max(self.values)))
        print('- MIN: {:.3f}'.format(min(self.values)))
        print("-" * 50)