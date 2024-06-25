# BLEU

## 1. Introduction
In this article, I will try to explain the intonation behind BLEU score. Most of the posts online briefly explain the formula 
and the two main parts of the BLEU score, precision and brevity penalty. However, none of them explain the intuition behind the
formula, especially the exponential part. The [original paper](https://www.aclweb.org/anthology/P02-1040.pdf) of BLEU score mentioned
the reason why they use the exponential function, but did not explain the intuition behind it. In this article, I will try to write out 
my understanding of the BLEU score and the intuition behind the formula.

## 2. Define the Problem
Before we dive into the BLEU score, let's jump back to the problem we are trying to solve. 
Translation Quality is a subjective measure, it is hard to quantify, since language is ambiguous. 
The process of directly compare two translations is merely impossible without human involve. 
Therefore, we need to find a workaround to quantify the quality of the translation, compare the similarity between the 
machine translation and the human translation could be a good start.
If we treat the human translation as the golden standard, we can calculate the similarity between the machine translation and the human translation.
The higher the similarity, the better the translation quality.

## 3. Precision
A really naive way of evaluating the translation is precision, 
which is the ratio of the number of words in the machine translation that appear in the human translation
to the total number of words in the machine translation.
The ratio could tell us how many words in the machine translation are "correct".
the formula is as follows:
$`
P = \frac{\sum_{i=1}^{n} \min(Count_{\text{clip}}(w_i), Count(w_i))}{\sum_{i=1}^{n} Count(w_i)}
`$
where $n$ is the length of the machine translation, $w_i$ is the $i$-th word in the machine translation,

`$
where $n$ is the length of the machine translation, $w_i$ is the $i$-th word in the machine translation,

