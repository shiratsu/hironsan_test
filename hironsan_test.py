# -*- coding: utf-8 -*-
#
# 訓練されたもののテストを実行する
#

from itertools import chain
import pycrfsuite
import sklearn
import hironsan_train 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

import sys
import codecs


if __name__ == '__main__':

    args = sys.argv
    
    c = hironsan_train.CorpusReader('hironsan.txt')
    train_sents = c.iob_sents('train')
    test_sents = c.iob_sents('test')

    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')

    example_sent = test_sents[0]
    if len(args) == 2:
        example_sent = test_sents[int(args[1])]

    print(example_sent)

    print(' '.join(hironsan_train.sent2tokens(example_sent)))

    print("Predicted:", ' '.join(tagger.tag(hironsan_train.sent2features(example_sent))))
    print("Correct:  ", ' '.join(hironsan_train.sent2labels(example_sent)))
