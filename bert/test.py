import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import re

model_dir = "bert/saved_model"

model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_amp(input_seq):

    input_seq_spaced = " ".join(
        [input_seq[i : i + 1] for i in range(0, len(input_seq), 1)]
    )
    input_seq_spaced = re.sub(r"[UZOB]", "X", input_seq_spaced)
    input_seq_tok = tokenizer(
        input_seq_spaced,
        return_tensors="pt",
        padding="max_length",
        max_length=200,
        truncation=True,
    )

    input_seq_tok = {key: val.to(device) for key, val in input_seq_tok.items()}

    # Perform inference
    with torch.no_grad():
        output = model(**input_seq_tok)

    logits = output.logits
    y_prob = torch.sigmoid(logits[:, 1]).item()

    y_pred = y_prob > 0.5
    input_class = "AMP" if y_pred else "non-AMP"

    # Output the result
    print("Input peptide sequence: " + input_seq)
    print("Class prediction: " + input_class)
    print("Probability of AMP class: {:.2f}".format(y_prob))


if __name__ == "__main__":
    input_seq = "FNRGGYNFGKSVRHVVDAIGSVAGIRGILKSIR"
    predict_amp(input_seq)
