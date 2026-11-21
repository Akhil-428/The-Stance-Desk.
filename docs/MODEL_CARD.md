# Model card: `bert-fnc1` (BERT fine-tuned for headline-article stance)

Format after [Mitchell et al., "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993). All figures come from `models/bert-fnc1/model_info.json` and `outputs/metrics/ours/bert_fnc1.json`.

## Model details

| | |
|---|---|
| Model | `BertForSequenceClassification` (`bert-base-uncased`, 109,485,316 parameters), 4 output classes |
| Base weights | [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) @ `86b5e0934494bd15c9632b12f734a8a67f723594` (Apache-2.0) |
| Fine-tuned by | This project, `train_bert.py`, one run (seed 42) |
| Input | A headline and an article body, encoded as a sentence pair, truncated to 512 tokens |
| Output | One of `agree`, `disagree`, `discuss`, `unrelated` and a softmax probability for each |
| Format | `model.safetensors` (not stored in git; see [models/README.md](../models/README.md)) |
| License | Code: Apache-2.0. The fine-tuned weights derive from an Apache-2.0 model and were trained on FNC-1, whose own terms apply to the data. |

## Intended use

- Studying and demonstrating stance detection between a news headline and an article body.
- A teaching and research baseline on FNC-1, with a transparent evaluation protocol.
- As one signal inside a human-in-the-loop workflow, e.g. surfacing articles whose body appears to contradict their headline for a person to read.

## Out-of-scope use

- **Fact-checking or truth labelling.** The model says how a body relates to a headline, not whether either is true. An article can agree with a false claim.
- Automated moderation, ranking or takedown decisions without human review.
- Languages other than English, long documents (only the first 512 tokens are read), or domains far from 2017 English news (social media, scientific text, …); behaviour there is untested.
- Any setting where a confident wrong answer causes harm to a person.

## Training data

FNC-1 training set: 49,972 headline-body pairs built from 1,683 article bodies. Class balance: unrelated 73.1 %, discuss 17.8 %, agree 7.4 %, disagree 1.7 %. 90 % (44,973 pairs) were used for training and 10 % (4,999 pairs) for validation, split **by article body** so that no body is in both. See [data/README.md](../data/README.md).

## Evaluation data and results

Official FNC-1 competition test set: 25,413 pairs (unrelated 72.2 %, discuss 17.6 %, agree 7.5 %, disagree 2.7 %), scored **once**, after the checkpoint had been selected on validation macro-F1.

| Metric | Test set | Validation set |
|---|---|---|
| Accuracy | 0.9256 | 0.9584 |
| Macro precision / recall | 0.758 / 0.733 | 0.852 / 0.860 |
| Macro-F1 | 0.744 | 0.855 |
| Relative FNC-1 score | 87.77 % (10,226.50 / 11,651.25) | 93.00 % |
| "Always unrelated" reference | accuracy 0.722, relative FNC 39.37 % | accuracy 0.732, relative FNC 40.52 % |

Per class on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| agree | 0.676 | 0.679 | 0.678 | 1,903 |
| disagree | 0.536 | 0.429 | 0.476 | 697 |
| discuss | 0.834 | 0.830 | 0.832 | 4,464 |
| unrelated | 0.985 | 0.993 | 0.989 | 18,349 |

## Caveats and known limitations

- **`disagree` is the weak class** (recall 0.43 on test): it is rare, and refutation is the hardest relation to learn. Class weighting was not applied, to keep the original recipe.
- **Validation-to-test gap** (macro-F1 0.855 vs 0.744): the official test set covers topics not seen in training, so the validation split, although grouped by body, still shares topics with the training data.
- **Over-confidence.** Predicted probabilities are not calibrated; the model often reports above 99 % even when wrong. Treat the probabilities as scores, not as likelihoods.
- **Truncation.** About a third of training pairs exceed 512 tokens, so evidence late in a long article is invisible to the model.
- **Single run.** The results come from one training run with one seed; run-to-run variance and confidence intervals were not measured. Differences of a few tenths of a point should not be over-read.
- **Data age and bias.** FNC-1 was collected for a 2017 challenge. It reflects the topics, sources and annotation conventions of that data and may carry their biases.
- **Not the original model.** This is a new training run, not a reproduction of the original project's 90.37 % checkpoint, which was never published. The two are not directly comparable (different input encoding, training subset and selection rule).

## Ethical considerations

Stance labels can be misread as verdicts about truth. The interface states "not fact-checking" prominently, and every model output carries its engine, device and confidence caveats. Use the model to support human judgement, not to replace it.
