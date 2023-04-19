import wordninja
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
from os.path import join, exists
import re
from os.path import dirname, abspath, join, exists


def get_root_path():
    """获取项目根路径"""
    return dirname(abspath(__file__))


def clean_data(dataset):
    clean_text_path = join(get_root_path(), 'corpus', dataset + '_sentences_clean.txt')
    docs_list = []
    old_name = dataset
    if "no_hashtag" in dataset:
        dataset = '_'.join(dataset.split('_')[:-2])
    with open(join(get_root_path(), 'corpus', dataset + '_sentences.txt')) as f:
        for line in f.readlines():
            docs_list.append(line.strip())
    dataset = old_name
    word_counts = defaultdict(int)
    for doc in docs_list:
        words = clean_doc(doc)
        for word in words:
            word_counts[word] += 1
    clean_docs = clean_documents(docs_list, word_counts=word_counts)
    corpus_str = '\n'.join(clean_docs)
    f = open(clean_text_path, 'w')
    f.write(corpus_str)
    f.close()


    f = open(clean_text_path, 'r')
    lines = f.readlines()
    min_len = 10000
    aver_len = 0
    max_len = 0
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
            print(line)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))


def clean_documents(docs, word_counts=None):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    ret = []
    for doc in docs:
        if doc == docs[8]:
            print(doc)
        words = clean_doc(doc)
        if doc == docs[8]:
            print(words)
        # 保留不在停用词词表且词频大于等于5的词
        if word_counts != None:
            words = [word for word in words if word not in stop_words and len(word) >= 3]
        else:
            words = [word for word in words if word not in stop_words]

        if doc == docs[8]:
            print(words)
        doc = ' '.join(words).strip()
        if doc != '':
            ret.append(' '.join(words).strip())
        else:
            ret.append(' ')
    return ret


def clean_doc(string):
    string = string.lower()
    string = re.sub(r"#semst", "", string)
    string = re.sub(r"@\S*", " ", string)

    # ??   U.S.
    string = re.sub(r"[^a-z\']", " ", string)

    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"[\W\d_]", " ", string)
    string = re.sub(r" [a-z] ", " ", string)
    string = re.sub(r" [a-z] ", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = "".join(string)
    string = wordninja.split(string)

    return string


if __name__ == "__main__":
    dataset = 'twitter_target'

    clean_data(dataset)
    # 清除target文件中的停用词
    target_path = join(get_root_path(), 'corpus', dataset + '_targets.txt')
    clean_target_path = join(get_root_path(), 'corpus', dataset + '_targets_clean.txt')
    if not exists(clean_target_path):
        targets_list = []
        with open(target_path) as f:
            for line in f.readlines():
                targets_list.append(line.strip())
            targets_list.append('\n')

        clean_targets = clean_documents(targets_list, None)
        corpus_str = '\n'.join(clean_targets)
        f = open(clean_target_path, 'w')
        f.write(corpus_str)
        f.close()
