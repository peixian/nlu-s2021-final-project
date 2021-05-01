from torch.nn import Softmax


def _softmax_and_relabel(predictions, categorical_labels):
    # takes in a torch.Tensor of predictions and a list of categorical labels, returns a tuple of (softmax tensor, categorical label)
    m = Softmax(dim=0)
    sm = m(predictions)
    return sm, categorical_labels[sm.argmax().item()]


# -----------------------Social Bias Frames-------------------

# Each dataset should have two functions, one that relabels the dataset, and another that turns the vector into a single score
def relabel_social_bias_frames(toxicity):
    if offensive:
        if offensive == "0.0":
            return 0
        elif offensive == "0.5":
            return 1
        else:
            return 2
    else:
        return 0


def return_social_bias_frames(predictions):
    # predictions for social bias frames is a 3 size vector with [not offensive, maybe offensive, offensive]
    # returns a tuple of a score and a categorical variable
    categorical_labels = ["Not", "Maybe", "Yes"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    # if maybe has the highest value, score is 0
    # otherwise, take the difference between Yes and No
    if category == "Maybe":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[0], category


# -----------------------rtGender-------------------

# Each dataset should have two functions, one that relabels the dataset, and another that turns the vector into a single score
def relabel_rt_gender(toxicity):
    # rtGender has 4 types: ['Negative', 'Positive', 'Neutral', 'Mixed']
    if sentiment == "Negative":
        return 0
    elif sentiment == "Neutral":
        return 1
    elif sentiment == "Mixed":
        return 2
    else:
        return 3


def return_rt_gender(predictions):
    # predictions for social bias frames is a 3 size vector with [not offensive, maybe offensive, offensive]
    # returns a tuple of a score and a categorical variable
    categorical_labels = ["Negative", "Neutral", "Mixed", "Positive"]
    softmax_preds, category = _softmax_and_relabel(predictions, categorical_labels)
    # if maybe has the highest value, score is 0
    # otherwise, take the difference between Yes and No
    if category == "Maybe":
        return 0, category
    else:
        return softmax_preds[2] - softmax_preds[0], category
