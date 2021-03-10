Potential Training Dataset List
===========
| Name                      | Type         | Description                                          | Link                                                                    | License       |
|---------------------------|--------------|------------------------------------------------------|-------------------------------------------------------------------------|---------------|
| Equity Evaluation Corpus  | Gender       | Derived from semeval-2018 tweets                     | https://www.aclweb.org/anthology/S18-2005.pdf                           | ?             |
| RtGender                  | Gender       | 25M comments from FB, TED, Fitocracy, and Reddit     | https://nlp.stanford.edu/robvoigt/rtgender/                             | Research Only |
| Social Bias Frames        | Gender/Mixed | Annotated social media posts, aims at implications   | https://huggingface.co/datasets/social_bias_frames                      | cc-by-4.0     |
| Md Gender Bias            | Gender       | Annotated version of eight different datasets        | https://huggingface.co/datasets/md_gender_bias                          | mit           |
| Jigaw                     | Toxicity     | Labeled from Wikipedia comments                      | https://huggingface.co/datasets/jigsaw_toxicity_pred                    | cc0-1.0       |
| SOLID                     | Toxicity     | 9M tweets labeled by toxicity                        | https://sites.google.com/site/offensevalsharedtask/solid                | ?             |
| OLID                      | Toxicity     | 142k labeled tweets                                  | https://sites.google.com/site/offensevalsharedtask/olid                 | ?             |
| Veiled Toxicity Detection | Toxicity     | Potential toxicity counterpart to Social Bias Frames | https://github.com/xhan77/veiled-toxicity-detection/tree/main/resources | ?             |
| Winograd                  | Gender (?)   | Windograd scheme, note that this is not drop in      | https://huggingface.co/datasets/wino_bias                               | mit           |

Potential Usages
------
- 2 classifiers, one for gender bias and one for toxicity
- each classifier fine tuned on a subset of the gender/toxicity datasets
- training input is a (sentence/label) pair, output is a softmax
- evaluation is input of a single sentence
- output example is:

| Sentence | Gender Bias Score | Toxicity Score |
|----------|-------------------|----------------|
| S_n      | GB_n              | T_n            |
| S_n+1    | GB_n+1            | T_n+1          |
| ...      | ...               | ...            |

- eventual goal is to find whether there is correlation between gender bias and toxicity
Multi-label model
----------
It's possible the independent evaluation of both classifiers leaves out some amount of information. In reality, we'd want a database that has *both* gender bias and toxicity, although that seems to be relatively rare.

However, using the two independent classifiers (toxicity and gender bias), we could potentially supplement the datasets with additional labels, such as labeling the toxicity dataset with the gender bias classifer, resulting in something like:


| Sentence | Original Dataset Toxicity Score | Predicted Gender Bias Score           |
|----------|---------------------------------|---------------------------------------|
| S_n      | T_n                             | Generated from gender bias classifier |


While this data is not perfect, it is better than synthetic data. Using this supplemented dataset, we could finetune a third multi-label model that attempts to capture the correlation of both toxicity and gender bias. It's also possible that what we want want is not strictly the correlation, but rather the covariance of the two.

## Open Questions
- How will we handle passages? Do we want to handle passages?
