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
