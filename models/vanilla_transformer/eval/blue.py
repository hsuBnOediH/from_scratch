# bleu short for BiLingual Evaluation Understudy is a metric for evaluating a generated sentence to a reference
# sentence. the value of bleu is between 0 and 1, the higher, the better. But usually people will report the bleu
# score in percentage.
import collections
import math
from collections import defaultdict
from time import time

# 0 - 10: Almost useless
# 10 - 20: Hard to get the gist
# 20 - 30: The gist is clear, but still hard to understand
# 30 - 40: Understandable, but not fluent
# 40 - 50: Fluent, but not perfect
# 50 - 60: Almost perfect
# > 60: better than human

# BLEU contains 2 main parts: n-gram overlap and brevity penalty.


#   a really naive way of evaluating the translation is precision:
#   for each word is the predicted sentence, if the reference sentence contains the word, then it is a correct
#   prediction
#   precision = number of correct predictions / number of total predictions
#   but the limitation of this method is obvious

#   For example:
predicted_sentence = ["the", "cat", "the", "the", "the", "the"]
reference_sentence = ["the", "cat", "is", "on", "the", "mat"]


def precision(predicted_sentence, reference_sentence):
    predicted_frequency = defaultdict(int)
    for word in predicted_sentence:
        predicted_frequency[word] += 1
    correct_predictions = 0
    for word, frequency in predicted_frequency.items():
        if word in reference_sentence:
            correct_predictions += frequency
    return correct_predictions / len(predicted_sentence)


#   To mitigate this limitation, one came up with modified precision:
def modified_precision(predicted_sentence, reference_sentence):
    predicted_frequency = defaultdict(int)
    reference_frequency = defaultdict(int)
    for word in predicted_sentence:
        predicted_frequency[word] += 1
    for word in reference_sentence:
        reference_frequency[word] += 1
    correct_predictions = 0
    for word, frequency in predicted_frequency.items():
        if word in reference_sentence:
            correct_predictions += min(frequency, reference_frequency[word])
    return correct_predictions / len(predicted_sentence)


#   this is a better metric than precision, but still has its limitations:
#   this metric does not consider the order of the words

# to consider the order of the words, the really intuitive way is to use n-gram, beside only using single word(
# unigram) as the unit of the prediction, one could use bigram, trigram, 4-gram, etc. as the unit of the prediction
# now the calculation of the multiple-gram modified precision is:
def multi_gram_modified_precision(predicted_sentence, reference_sentence, n=4):
    def i_gram_modified_precision(predicted_sentence, reference_sentence, i):
        predicted_frequency = defaultdict(int)
        reference_frequency = defaultdict(int)
        for j in range(len(predicted_sentence) - i + 1):
            predicted_frequency[tuple(predicted_sentence[j:j + i])] += 1
        for j in range(len(reference_sentence) - i + 1):
            reference_frequency[tuple(reference_sentence[j:j + i])] += 1
        correct_predictions = 0
        for word, frequency in predicted_frequency.items():
            if word in reference_sentence:
                correct_predictions += min(frequency, reference_frequency[word])
        return correct_predictions

    res = 0
    for i in range(1, n + 1):
        res += i_gram_modified_precision(predicted_sentence, reference_sentence, i)
    return res / n

# this metric is better than the modified precision, but still has two limitations:
# 1. compute the avg over all n-gram, but not all n-gram are equally important, for example,  4 gram is much more
#    important than 1 gram, so we should weight the n-gram by their importance
#  by using the log and exp
def weighted_multi_gram_modified_precision(predicted_sentence, reference_sentence, n=4):
    def i_gram_modified_precision(predicted_sentence, reference_sentence, i):
        predicted_frequency = defaultdict(int)
        reference_frequency = defaultdict(int)
        for j in range(len(predicted_sentence) - i + 1):
            predicted_frequency[tuple(predicted_sentence[j:j + i])] += 1
        for j in range(len(reference_sentence) - i + 1):
            reference_frequency[tuple(reference_sentence[j:j + i])] += 1
        correct_predictions = 0
        for word, frequency in predicted_frequency.items():
            if word in reference_sentence:
                correct_predictions += min(frequency, reference_frequency[word])
        return math.log(correct_predictions / len(predicted_sentence))

    res = 0
    for i in range(1, n + 1):
        res += i_gram_modified_precision(predicted_sentence, reference_sentence, i) * (1 / n)
    return math.exp(res)


# this metric does not consider the length of the sentence, the method could be easily cheated by generating a
# sentence with really short length, for example:
# predicted_sentence = ["the"]
# reference_sentence = ["the", "cat", "is", "on", "the", "mat"]
# since the predicted sentence is really short, the modified precision will be 1, but actually the predicted sentence
# is not even close to the reference sentence. so we have to consider the length of the sentence, this is where brevity
# penalty comes in.

# if the predicted sentence is longer than the reference sentence, then the brevity penalty is 1, otherwise, the
# brevity penalty is exp(1 - reference_length / predicted_length)

# the n-gram modified precision range is from 0 to 1,
def brevity_penalty(predicted_sentence, reference_sentence):
    predicted_length = len(predicted_sentence)
    reference_length = len(reference_sentence)
    if predicted_length > reference_length:
        return 1
    # since we want to penalize the predicted sentence that is shorter than the reference sentence,
    # 1- reference_length / predicted_length will be negative, so we have to use exp to make it positive
    return math.exp(1 - reference_length / predicted_length)





# Geometric mean of the n-gram modified precision, since there is strong correlation between the n-gram modified
# directly multiply them will make the result underflow, since if you multiply a number between 0 and 1 multiple times
# the result will be really small, so we use the  exp the sum of the log of the n-gram modified precision to avoid





def _get_ngrams(segment, max_order):
    """
    Extracts all n-grams up to a given maximum order from an input segment.

    :param segment:
        text segment from which n-grams will be extracted
        list of tokens
        ["token1", "token2", "token3", "token4", "token5"]
    :param max_order:
        maximum length of n-grams
    :return:
        a Counter with n-gram counts
    """
    # create a counter to store the n-gram counts
    ngram_counts = collections.Counter()
    # run through all the n-gram from 1 to max_order
    for order in range(1, max_order + 1):
        # run through all the n-gram in the segment
        for i in range(len(segment) - order + 1):
            # get the n-gram, need to convert the n-gram to tuple since list is not hashable
            ngram = tuple(segment[i:i + order])
            # increment the n-gram count
            ngram_counts[ngram] += 1
    # return the n-gram counts, in which the key is the form 1 to max_order n-gram, the value is the frequency of the
    # n-gram
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False, smooth_value=0.0):
    """
     Implementation of BLEU score.
    :param reference_corpus:
        list of list of reference sentences, each sentence is a list of tokens
        1 st level: number of "list of reference sentences", len is the number of translations
        [
            [
                ["token1", "token2", "token3", "token4", "token5"],
                ["token1", "token2", "token3"]
            ],
            [
                ["token1", "token2 ", "token3", "token4", "token5"],
                ["token1", "token2", "token3"]
            ]
        ]
        2 nd level: number of "reference sentences" for a single translation, len is the number of references
        [
            ["token1", "token2", "token3", "token4", "token5"],
            ["token1", "token2", "token3"]
        ]
        3 rd level: number of tokens in a single reference sentence, len is the number of tokens in a single sentence
        ["token1", "token2", "token3", "token4", "token5"]
    :param translation_corpus:
        list of translated sentences, each sentence is a list of tokens, those sentences are the predicted sentences
        that we want to evaluate
        1 st level: number of "list of translated sentences", len is the number of translations
        [
            ["token1", "token2", "token3", "token4", "token5"],
            ["token1", "token2", "token3"]
        ]
        2 nd level: number of tokens in a single translated sentence, len is the number of tokens in a single sentence
        ["token1", "token2", "token3", "token4", "token5"]
    :param max_order:
        the maximum n-gram order to use when computing BLEU score, usually 4
    :param smooth:
        whether to apply smoothing, default is False, if do not apply smoothing, then the n-gram modified
        precision will be 0 if there is no n-gram overlap, that will make the log of 0, which is undefineda
    :param smooth_value:
        the value to use when applying smoothing, default is 0.0
    :return:
        the BLEU score, the value is between 0 and 1, the higher, the better
    """

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for (references, translation) in zip(reference_corpus, translation_corpus):
        # when compute the brevity penalty, we have to consider the shortest reference sentence
        # for the translation sentences, we need to add them up
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        # create a counter to store the n-gram counts of the merged reference sentences
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            # compute the n-gram counts of every reference sentence, and merge them
            # the merge is not accumulative, it is the keep the maximum count of the n-gram
            # for counter, the + operator will sum the count of the same key, where the | operator will keep the maximum
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        # get the n-gram counts of the translation sentence
        translation_ngram_counts = _get_ngrams(translation, max_order)
        # get the n-gram overlap of the translation sentence and the merged reference sentences
        # the & operator will return the minimum count of the n-gram, if the n-gram is not in the merged reference
        # sentences, then the count will be 0
        overlap = translation_ngram_counts & merged_ref_ngram_counts

        for ngram in overlap:
            # increment the n-gram overlap count
            # len(ngram) is to calculate the order of the n-gram, since the n-gram is a tuple, the length of the tuple
            # minus 1 will be the order of the n-gram
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            # when calculate precision of the n-gram, we have to consider the possible matches as the denominator
            # if the sentence len is x, then the possible n-gram is x - order + 1
            possible_matches = len(translation) - order + 1
            # to avoid the division by 0, we have to check if the possible matches is greater than 0,
            # that only happens when the order is greater than the length of the sentence
            # for example, if the sentence is ["the", "cat"], the possible bigram is 1, the possible trigram is 0
            if possible_matches > 0:
                # increment the possible n-gram matches count
                possible_matches_by_order[order - 1] += possible_matches
        precision = [0] * max_order
        for i in range(0,max_order):
            if smooth:
                # if one of the n-gram order has no possible matches, then the precision will be 0
                # but we have to avoid the division by 0, so we have to add a smooth value
                precision[i] = (matches_by_order[i] + smooth_value) / (possible_matches_by_order[i] + smooth_value)
            else:
                if possible_matches_by_order[i] > 0:
                    precision[i] = matches_by_order[i] / possible_matches_by_order[i]
                else:
                    precision[i] = 0

        if min(precision) > 0:
            p_log_sum = sum((1 / max_order) * math.log(p) for p in precision)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        # compute the brevity penalty
        ratio =  float(translation_length) / reference_length
        if ratio > 1.0:
            bp = 1
        else:
            bp = math.exp(1 - 1.0 / ratio)
        bleu = geo_mean * bp
    return bleu, precision, bp, ratio, translation_length, reference_length



