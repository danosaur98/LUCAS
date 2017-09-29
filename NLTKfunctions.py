from nltk.metrics import agreement  # skipped for now
from nltk.metrics import association  # skipped for now
from nltk.metrics import confusionmatrix
from nltk.metrics import distance
from nltk.metrics import scores
from nltk.metrics import segmentation
from nltk.metrics import spearman
from nltk.misc import sort
from nltk.sem import glue


def confusion_matrix(reference, test, sort_by_count=False):
    return confusionmatrix.ConfusionMatrix(reference, test, sort_by_count)


def binary_distance(label1, label2):
    """Simple equality test.

    0.0 if the labels are identical, 1.0 if they are different.

    >>> from nltk.metrics import binary_distance
    >>> binary_distance(1,1)
    0.0

    >>> binary_distance(1,3)
    1.0
    """
    return distance.binary_distance(label1, label2)


def jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity.

    """
    return distance.jaccard_distance(label1, label2)


def masi_distance(label1, label2):
    """Distance metric that takes into account partial agreement when multiple
    labels are assigned.

    >>> from nltk.metrics import masi_distance
    >>> masi_distance(set([1, 2]), set([1, 2, 3, 4]))
    0.335

    Passonneau 2006, Measuring Agreement on Set-Valued Items (MASI)
    for Semantic and Pragmatic Annotation.
    """
    return distance.masi_distance(label1, label2)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    return distance.edit_distance(s1, s2, substitution_cost, transpositions)


def interval_distance(label1, label2):
    """Krippendorff's interval distance metric

    >>> from nltk.metrics import interval_distance
    >>> interval_distance(1,10)
    81

    Krippendorff 1980, Content Analysis: An Introduction to its Methodology
    """
    return distance.interval_distance(label1, label2)


def presence(label):
    """Higher-order function to test presence of a given label
    """
    return distance.presence(label)


def fractional_presence(label):
    return distance.fractional_presence(label)


def custom_distance(file):
    return distance.custom_distance(file)


def accuracy(reference, test):
    """
    Given a list of reference values and a corresponding list of test
    values, return the fraction of corresponding values that are
    equal.  In particular, return the fraction of indices
    ``0<i<=len(test)`` such that ``test[i] == reference[i]``.

    :type reference: list
    :param reference: An ordered list of reference values.
    :type test: list
    :param test: A list of values to compare against the corresponding
        reference values.
    :raise ValueError: If ``reference`` and ``length`` do not have the
        same length.
    """
    return scores.accuracy(reference, test)


def precision(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of test values that appear in the reference set.
    In particular, return card(``reference`` intersection ``test``)/card(``test``).
    If ``test`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    return scores.precision(reference, test)


def recall(reference, test):
    """
    Given a set of reference values and a set of test values, return
    the fraction of reference values that appear in the test set.
    In particular, return card(``reference`` intersection ``test``)/card(``reference``).
    If ``reference`` is empty, then return None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    return scores.recall(reference, test)


def f_measure(reference, test, alpha=0.5):
    """
    Given a set of reference values and a set of test values, return
    the f-measure of the test values, when compared against the
    reference values.  The f-measure is the harmonic mean of the
    ``precision`` and ``recall``, weighted by ``alpha``.  In particular,
    given the precision *p* and recall *r* defined by:

    - *p* = card(``reference`` intersection ``test``)/card(``test``)
    - *r* = card(``reference`` intersection ``test``)/card(``reference``)

    The f-measure is:

    - *1/(alpha/p + (1-alpha)/r)*

    If either ``reference`` or ``test`` is empty, then ``f_measure``
    returns None.

    :type reference: set
    :param reference: A set of reference values.
    :type test: set
    :param test: A set of values to compare against the reference set.
    :rtype: float or None
    """
    return scores.f_measure(reference, test, alpha)


def log_likelihood(reference, test):
    """
    Given a list of reference values and a corresponding list of test
    probability distributions, return the average log likelihood of
    the reference values, given the probability distributions.

    :param reference: A list of reference values
    :type reference: list
    :param test: A list of probability distributions over values to
        compare against the corresponding reference values.
    :type test: list(ProbDistI)
    """
    return scores.log_likelihood(reference, test)


# def approxrand(a, b, **kwargs):
# return scores.approxrand(a, b, kwargs)

def windowdiff(seg1, seg2, k, boundary="1", weighted=False):
    """
    Compute the windowdiff score for a pair of segmentations.  A
    segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

        >>> s1 = "000100000010"
        >>> s2 = "000010000100"
        >>> s3 = "100000010000"
        >>> '%.2f' % windowdiff(s1, s1, 3)
        '0.00'
        >>> '%.2f' % windowdiff(s1, s2, 3)
        '0.30'
        >>> '%.2f' % windowdiff(s2, s3, 3)
        '0.80'

    :param seg1: a segmentation
    :type seg1: str or list
    :param seg2: a segmentation
    :type seg2: str or list
    :param k: window width
    :type k: int
    :param boundary: boundary value
    :type boundary: str or int or bool
    :param weighted: use the weighted variant of windowdiff
    :type weighted: boolean
    :rtype: float
    """
    return segmentation.windowdiff(seg1, seg2, k, boundary, weighted)


def ghd(ref, hyp, ins_cost=2.0, del_cost=2.0, shift_cost_coeff=1.0, boundary='1'):
    """
    Compute the Generalized Hamming Distance for a reference and a hypothetical
    segmentation, corresponding to the cost related to the transformation
    of the hypothetical segmentation into the reference segmentation
    through boundary insertion, deletion and shift operations.

    A segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

    Recommended parameter values are a shift_cost_coeff of 2.
    Associated with a ins_cost, and del_cost equal to the mean segment
    length in the reference segmentation.

        >>> # Same examples as Kulyukin C++ implementation
        >>> ghd('1100100000', '1100010000', 1.0, 1.0, 0.5)
        0.5
        >>> ghd('1100100000', '1100000001', 1.0, 1.0, 0.5)
        2.0
        >>> ghd('011', '110', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('1', '0', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('111', '000', 1.0, 1.0, 0.5)
        3.0
        >>> ghd('000', '111', 1.0, 2.0, 0.5)
        6.0

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the hypothetical segmentation
    :type hyp: str or list
    :param ins_cost: insertion cost
    :type ins_cost: float
    :param del_cost: deletion cost
    :type del_cost: float
    :param shift_cost_coeff: constant used to compute the cost of a shift.
    shift cost = shift_cost_coeff * |i - j| where i and j are
    the positions indicating the shift
    :type shift_cost_coeff: float
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """
    return segmentation.ghd(ref, hyp, ins_cost, del_cost, shift_cost_coeff, boundary)


def pk(ref, hyp, k=None, boundary='1'):
    """
    Compute the Pk metric for a pair of segmentations A segmentation
    is any sequence over a vocabulary of two items (e.g. "0", "1"),
    where the specified boundary value is used to mark the edge of a
    segmentation.

    >>> '%.2f' % pk('0100'*100, '1'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0100'*100, 2)
    '0.00'

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the segmentation to evaluate
    :type hyp: str or list
    :param k: window size, if None, set to half of the average reference segment length
    :type boundary: str or int or bool
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """
    return segmentation.pk(ref, hyp, k, boundary)


def setup_module(module):
    return segmentation.setup_module(module)


def spearman_correlation(ranks1, ranks2):
    """Returns the Spearman correlation coefficient for two rankings, which
    should be dicts or sequences of (key, rank). The coefficient ranges from
    -1.0 (ranks are opposite) to 1.0 (ranks are identical), and is only
    calculated for keys in both rankings (for meaningful results, remove keys
    present in only one list before ranking)."""
    return spearman.spearman_correlation(ranks1, ranks2)


def ranks_from_sequence(seq):
    """Given a sequence, yields each element with an increasing rank, suitable
    for use as an argument to ``spearman_correlation``.
    """
    return spearman.ranks_from_sequence(seq)


def ranks_from_scores(scores, rank_gap=1e-15):
    """Given a sequence of (key, score) tuples, yields each key with an
        increasing rank, tying with previous key's rank if the difference between
        their scores is less than rank_gap. Suitable for use as an argument to
        ``spearman_correlation``.
        """
    return spearman.ranks_from_scores(scores, rank_gap)


def selection(a):
    """
    Selection Sort: scan the list to find its smallest element, then
    swap it with the first element.  The remainder of the list is one
    element smaller; apply the same method to this list, and so on.
    """
    return sort.selection(a)


def bubble(a):
    """
    Bubble Sort: compare adjacent elements of the list left-to-right,
    and swap them if they are out of order.  After one pass through
    the list swapping adjacent items, the largest item will be in
    the rightmost position.  The remainder is one element smaller;
    apply the same method to this list, and so on.
    """
    return sort.bubble(a)


def merge(a):
    """
    Merge Sort: split the list in half, and sort each half, then
    combine the sorted halves.
    """
    return sort.merge(a)


def quick(a):
    return sort.quick(a)


def glue_formula(obj):
    return glue.GlueFormula(obj)



confusionmatrix.demo()
distance.demo()
scores.demo()
sort.demo()
