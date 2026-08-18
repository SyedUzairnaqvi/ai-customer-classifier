# AI Customer Insight Analyzer

A hybrid NLP application for financial customer-support analysis.

## Live Demo

**Streamlit App:** https://ai-customer-classifier-v6yuiq2aqqdhooajzs3eum.streamlit.app/

**Current release:** FINAL — this repository is using the `ai_customer_insight_analyzer_FINAL` project version as the main/root version.

## What it does

Given a customer message, the system produces:

1. **Intent** — routes the message into four business-level categories: `account`, `billing`, `card`, or `transfer`.
2. **Sentiment** — uses a pretrained 3-class RoBERTa sentiment model to classify the message as negative, neutral, or positive.
3. **Urgency** — applies transparent business rules to identify high/medium/low priority signals.
4. **Routing recommendation** — suggests the support team that should handle the message.
5. **Confidence and alternatives** — exposes the intent model probability and top three predictions. Low-confidence intent predictions are marked **Needs Review**.

## ML architecture

```text
Customer message
       |
       +----------------------+------------------+
       |                      |                  |
       v                      v                  v
TF-IDF + Logistic        RoBERTa           Business rules
Regression               Sentiment          for urgency
       |                      |                  |
       v                      v                  v
    Intent               Sentiment          Urgency
       |                      |                  |
       +----------------------+------------------+
                              |
                              v
                   Routing + confidence
                              |
                              v
                          Streamlit
```

### Why this is a hybrid system

- **Logistic Regression** handles intent classification from sparse TF-IDF text features.
- **RoBERTa** is used only for three-class sentiment analysis.
- **Rules** handle operational urgency because business priority is not the same thing as sentiment probability.
- **Confidence thresholding** prevents low-confidence intent predictions from being automatically routed.

## Dataset

The intent model is trained and evaluated on **BANKING77**, a public banking customer-service intent dataset containing 13,083 queries across 77 fine-grained intents: 10,003 training examples and 3,080 test examples.

The project maps the 77 original intents into four business-level routing groups for a simpler support workflow.

## Model training

If the dataset files are missing:

```bash
python download_data.py
```

Train and evaluate:

```bash
python train.py
```

This creates:

- `model.pkl` — trained TF-IDF + Logistic Regression pipeline
- `metrics.json` — held-out test metrics and confusion matrix

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the pretrained RoBERTa sentiment model from Hugging Face.

## Current offline evaluation

The included intent model was trained on 10,003 BANKING77 training examples and evaluated on the untouched 3,080-example test set.

Current four-class intent-routing performance:

- **Accuracy: 92.89%**
- **Macro F1: 92.89%**

These are **intent-routing metrics**, not sentiment metrics and not production performance guarantees.

## Limitations

- The four business categories are a project-level mapping of the original 77 intents.
- The sentiment model is a general-purpose pretrained model, not a financial-domain sentiment model.
- Urgency is a transparent rule layer, not a learned urgency model.
- Real production deployment would require monitoring, human review for low-confidence cases, privacy controls, and domain-specific validation.
- Sentiment predictions should be treated as model-assisted signals rather than ground truth.

## Project improvements

The current version improves the prototype by:

- using a substantially larger labeled intent dataset;
- evaluating on a separate held-out test set;
- using actual model probabilities rather than hard-coded 100% confidence;
- supporting a neutral sentiment class;
- recognizing time-sensitive urgency phrases such as `soon`, `ASAP`, `today`, and `as soon as possible`;
- providing a clear-conversation control for repeated testing.
