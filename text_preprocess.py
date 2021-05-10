import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


strip_tweets = re.compile(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r"[^\w\s]", "", word)
        if new_word != "":
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    pattern = "[0-9]"
    new_words = [re.sub(pattern, "", i) for i in words]
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words("english"):
            new_words.append(word)
    return new_words


def clean_spec_dataset(text, dataset_name):
    if dataset_name == "conv_ai_3":
        return "".join(text.split("|")[1:])
    elif dataset_name == "ted_talks_iwslt":
        return "".join(text.split(":")[1:])
    elif dataset_name == "tweet_eval":
        return strip_tweets.sub("", context)
    else:
        return text


def normalize_raw(text, dataset_name="", raw_text=True):
    text = clean_spec_dataset(text, dataset_name)
    return text.split(" ")


def normalize(text, dataset_name="", raw_text=True):
    text = clean_spec_dataset(text, dataset_name)
    words = nltk.word_tokenize(text)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words    


def keep_sentence(words):
    if len(words) < 2:
        return False
    return True
