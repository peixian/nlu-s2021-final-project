import spacy
nlp = spacy.load('en_core_web_sm')
from experiment_script import dataset_types

def get_sentences(paragraph):
    result = []
    try:
        doc = nlp(paragraph)
        for sentence in doc.sents:
            result.append(str(sentence))
    except Exception:
        print("This paragraph could not be converted", paragraph)
    return result


def split_long_text(list_paragraphs):
    results = []
    for paragraph in tqdm(list_paragraphs):
        results.append(get_sentences(paragraph))

    results = [item for sublist in results for item in sublist]
    return results

def concate(dataset_name, data, cache_dir):
    if dataset_name in dataset_types:
        all_datasets_downloaded = [
            load_dataset(dataset_name, sub_dataset, cache_dir=cache_dir)
            for sub_dataset in dataset_types[dataset_name]
        ]
        combined_datasets = [
            concatenate_datasets(list(sub_dataset.values()))
            for sub_dataset in all_datasets_downloaded
        ]
        data = concatenate_datasets(combined_datasets)
        return DatasetDict({"train": data})
    data = concatenate_datasets(
        list(load_dataset(dataset_name, cache_dir=cache_dir).values())
    )
    return DatasetDict({"train": data})	