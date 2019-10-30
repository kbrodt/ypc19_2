import argparse
import functools as f
import itertools as it
import logging
import multiprocessing
from pathlib import Path
import re

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict


N_TEST_SAMPLES = 31513
logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                    level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./topics_data',
                        help='Path to data')
    parser.add_argument('--save-path', type=str, default='./submission.tsv',
                        help='Path to save')

    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds')

    # Vectorizers' parameters
    parser.add_argument('--max-features-word', type=int, default=250_000,
                        help='Number of features for word level TfIdf')
    parser.add_argument('--ngram-range-word', type=str, default='1,3',
                        help='Number of n-grams for word level TfIdf')
    parser.add_argument('--max-features-char', type=int, default=350_000,
                        help='Number of features for char level TfIdf')
    parser.add_argument('--ngram-range-char', type=str, default='3,5',
                        help='Number of n-grams for char level TfIdf')

    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')

    return parser.parse_args()


def get_score(threshold, y_true, y_pred):
    return f1_score(y_true, (y_pred > threshold).astype('int'), average='samples')


def read_csv(path, columns):
    with open(path) as f:
        return pd.read_csv(f, sep='\t', header=None, names=columns)


def read_train(path):
    data = read_csv(path, ['idx', 'title', 'content', 'labels'])
    data.dropna(subset=['labels', 'content'], inplace=True)

    return data


def read_test(path):
    columns = ['idx', 'title', 'content']
    data = read_csv(path, columns)
    data['content'] = data['content'].str.strip()
    new_data = []
    row_iter = iter(data.iterrows())
    for _, row in row_iter:
        if row['content'] == '':
            new_row = row
            _, row = next(row_iter)
            text = ''
            while row['content'] != row['content']:
                text = text + ' ' + row['idx']
                _, row = next(row_iter)
            new_row['content'] = text
            new_data.append(new_row)
            new_data.append(row)
            continue
        if row['content'] == row['content']:
            new_data.append(row)
    new_data = pd.DataFrame(new_data, columns=columns)
    new_data['idx'] = new_data['idx'].apply(int)

    return new_data


def read_all_test(path):
    def _has_idx(x):
        if x != x:
            return False

        h = re.findall(r'(1[2345]\d{4})', x)
        if len(h) == 0:
            return False

        assert len(h) == 1

        return True

    columns = ['idx', 'title', 'content']
    data = read_csv(path, columns)

    data['has_id'] = data['idx'].apply(_has_idx)

    mask = ~data['has_id'] & ~data['title'].isna()
    data.loc[mask, 'content'] = data.loc[mask, 'title']
    data.loc[mask, 'title'] = np.nan

    mask = ~data['has_id'] & ~data['idx'].isna()
    data.loc[mask, 'content'] = data.loc[mask, 'idx']
    data.loc[mask, 'idx'] = np.nan

    data.loc[data.title.isna(), 'title'] = ''
    data.loc[data.content.isna(), 'content'] = ''

    data.fillna(method='ffill', inplace=True)

    data['corpus'] = data['title'] + ' ' + data['content']
    data = data.groupby('idx')['corpus'].apply(lambda xs: ' '.join(xs)).reset_index()

    assert len(data) == N_TEST_SAMPLES, len(data)

    return data


def make_folds(df, args):
    n_classes = len(set(it.chain(*df['labels'].str.split(',').values)))
    logging.info(f'Number of classes: {n_classes}')

    y = np.zeros((len(df), n_classes), dtype='int64')
    for i, row in enumerate(df.labels.str.split(',').apply(lambda xs: [int(x) for x in xs])):
        for x in row:
            y[i, x] = 1

    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, random_state=args.seed)
    df['fold'] = -1
    for i, (train_index, dev_index) in enumerate(mskf.split(range(len(df)), y)):
        df.iloc[dev_index, 4] = i

    return mskf, y


def main():
    args = parse_args()
    logging.info(args)

    data_path = Path(args.data_path)
    logging.info(f'Reading train from {data_path}..')
    train = read_train(data_path / 'train.tsv')

    logging.info(f'Reading test from {data_path}..')
    test = read_test(data_path / 'test.tsv')

    logging.info(f'Making {args.n_folds} folds..')
    mskf, y_train = make_folds(train, args)

    logging.info(
        f'Creating word level TfIdf vectorizer with params: ngram_range={args.ngram_range_word},'
        f' max_features={args.max_features_word}')
    tfidf_word_vec = TfidfVectorizer(ngram_range=tuple(map(int, args.ngram_range_word.split(','))),
                                     analyzer='word',
                                     max_features=args.max_features_word)

    logging.info(
        f'Creating char level TfIdf vectorizer with params: ngram_range={args.ngram_range_char},'
        f' max_features={args.max_features_char}')
    tfidf_char_vec = TfidfVectorizer(ngram_range=tuple(map(int, args.ngram_range_char.split(','))),
                                     analyzer='char',
                                     max_features=args.max_features_char)

    logging.info('Training word tfidf..')
    tfidf_word_vec.fit(it.chain(train.title, train.content, test.title, test.content))

    logging.info('Training char tfidf..')
    tfidf_char_vec.fit(it.chain(train.title, train.content, test.title, test.content))

    train['corpus'] = train['title'] + ' ' + train['content']

    logging.info('Transforming train..')
    X_train = sps.hstack([tfidf_word_vec.transform(train['corpus']),
                          tfidf_char_vec.transform(train['corpus'])])

    logging.info(f'Reading test from {data_path}..')
    test = read_all_test(data_path / 'test.tsv')

    logging.info('Transforming test..')
    X_test = sps.hstack([tfidf_word_vec.transform(test['corpus']),
                         tfidf_char_vec.transform(test['corpus'])])

    clf = OneVsRestClassifier(
        SGDClassifier(loss='modified_huber', random_state=args.seed),
        n_jobs=-1,
    )

    logging.info(f'{args.n_folds} cross-validation..')
    y_train_hat = cross_val_predict(clf, X_train, y_train, method='predict_proba', cv=mskf, n_jobs=-1)
    logging.info(f'Score without normalization: {get_score(0.5, y_train, y_train_hat):.4}')

    y_train_hat = y_train_hat / y_train_hat.max(1, keepdims=True)
    logging.info(f'Score with normalization: {get_score(0.5, y_train, y_train_hat):.4}')

    logging.info('Threshold search..')
    thresholds = np.linspace(0.0, 1., 100)
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as p:
        scores = list(p.imap(f.partial(get_score, y_true=y_train, y_pred=y_train_hat), thresholds))

    thresh_ind = np.argmax(scores)
    thresh = thresholds[thresh_ind]
    logging.info(f'Max score: {scores[thresh_ind]} with thresh {thresh}')

    logging.info(f'Training on full train {X_train.shape}..')
    clf.fit(X_train, y_train)

    logging.info(f'Predicting on full test {X_test.shape}..')
    y_test_hat = clf.predict_proba(X_test)
    y_test_hat = y_test_hat / y_test_hat.max(1, keepdims=True)

    logging.info('Making submission..')
    submission = []
    for idx, row in zip(test['idx'].values, y_test_hat):
        row = (row > thresh) + 0.0
        els = ','.join([str(x) for x in list(np.nonzero(row)[0])])
        submission.append([idx, els])
    submission = pd.DataFrame(submission)

    save_path = Path(args.save_path)
    logging.info(f'Saving results to {save_path}..')
    if not save_path.parent.exists():
        save_path.mkdir(parents=True)

    submission.to_csv(save_path, sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()
