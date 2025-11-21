from typing import List


def print_eval(classifier, exs):
    eval = evaluate(classifier=classifier, exs=exs)
    print(eval["output_str"])


def evaluate(classifier, exs):
    """
    Evaluates a given classifier on the given examples

    Args:
        classifier (MultilabelClassifier): classifier to evaluate
        exs: the list of examples to evaluate on

    Returns:
        (float, float, float, float, float): micro, macro, and weighted average F1, exact match ratio, and hamming loss
    """
    return _calculate_eval(
        [ex[1] for ex in exs],
        classifier.predict_all([ex[0] for ex in exs]),
        len(exs[0][1]),
    )


def _calculate_eval(
    golds: List[List[int]], predictions: List[List[int]], num_labels: int
) -> dict[str, float | str]:
    """
    Prints evaluation statistics comparing golds and predictions, each of which is a sequence of 0/1 labels.
    Prints accuracy as well as micro, macro, and weighted average F1 of the positive class in addition to the number of exact matches.

    Args:
        golds (List[List[int]]): gold labels
        predictions (List[List[int]]): pred labels

    Returns:
        dict[str,float]: micro, macro, and weighted average F1, exact match ratio, and hamming loss
    """
    num_correct = [0] * num_labels
    num_pos_correct = [0] * num_labels
    num_pred = [0] * num_labels
    num_gold = [0] * num_labels
    num_total = [0] * num_labels
    num_match = 0
    if len(golds) != len(predictions):
        raise Exception(
            "Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions))
        )
    for idx in range(0, len(golds)):  ## just doing micro-f1
        gold_labels = golds[idx]
        prediction_labels = predictions[idx]
        exact_match = True
        for i in range(0, len(gold_labels)):
            gold = gold_labels[i]
            prediction = prediction_labels[i]
            if prediction == gold:
                num_correct[i] += 1
            else:
                exact_match = False
            if prediction == 1:
                num_pred[i] += 1
            if gold == 1:
                num_gold[i] += 1
            if prediction == 1 and gold == 1:
                num_pos_correct[i] += 1
            num_total[i] += 1
        if exact_match:
            num_match += 1

    # output_str = "Total Accuracy: %i / %i = %f;\n" % (
    #     sum(num_correct),
    #     sum(num_total),
    #     float(sum(num_correct)) / sum(num_total),
    # )
    # label_acc = [
    #     float(num_correct[i]) / num_total[i] if num_total[i] > 0 else 0.0
    #     for i in range(num_labels)
    # ]
    # for i, label in enumerate(processed_labels):
    #     output_str += "%s Accuracy: %i / %i = %f;\n" % (
    #         label,
    #         num_correct[i],
    #         num_total[i],
    #         label_acc[i],
    #     )
    # output_str += "\n"

    prec = [
        float(num_pos_correct[i]) / num_pred[i] if num_pred[i] > 0 else 0.0
        for i in range(num_labels)
    ]
    rec = [
        float(num_pos_correct[i]) / num_gold[i] if num_gold[i] > 0 else 0.0
        for i in range(num_labels)
    ]
    f1 = [
        2 * prec[i] * rec[i] / (prec[i] + rec[i]) if prec[i] > 0 and rec[i] > 0 else 0.0
        for i in range(num_labels)
    ]
    micro_prec = (
        float(sum(num_pos_correct)) / sum(num_pred) if sum(num_pred) > 0 else 0.0
    )
    micro_rec = (
        float(sum(num_pos_correct)) / sum(num_gold) if sum(num_gold) > 0 else 0.0
    )

    macro_f1 = sum(f1) / num_labels
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if micro_prec > 0 and micro_rec > 0
        else 0.0
    )
    weighted_f1 = sum([f1[i] * num_gold[i] / sum(num_gold) for i in range(num_labels)])
    exact_match = num_match / len(golds)
    hamming_loss = float(sum(num_total) - sum(num_correct)) / sum(num_total)

    output_str = "Micro F1 (harmonic mean of precision and recall): %f;\n" % micro_f1
    output_str += "Macro F1 (F1 averaged over each label): %f;\n" % macro_f1
    output_str += "Weighted F1 (F1 weighted by label occurrences): %f;\n" % weighted_f1
    output_str += "Exact Match: %i / %i = %f;\n" % (
        num_match,
        len(golds),
        exact_match,
    )
    output_str += "Hamming Loss: %i / %i = %f;" % (
        sum(num_total) - sum(num_correct),
        sum(num_total),
        hamming_loss,
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "exact_match_ratio": exact_match,
        "hamming_loss": hamming_loss,
        "output_str": output_str,
    }
