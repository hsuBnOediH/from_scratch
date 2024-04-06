# bleu short for BiLingual Evaluation Understudy is a metric for evaluating a generated sentence to a reference
# sentence. the value of bleu is between 0 and 1, the higher the better. But usually people will report the bleu
# score in percentage.
import math
from collections import defaultdict

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
    return 1 - reference_length / predicted_length
