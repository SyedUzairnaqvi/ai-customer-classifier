"""Train the customer-intent classifier on BANKING77.

The project uses BANKING77 as the real labeled training/test dataset and maps
its 77 fine-grained banking intents into four business-level routing groups.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
TRAIN_FILE = BASE / "banking_train.csv"
TEST_FILE = BASE / "banking_test.csv"
MODEL_FILE = BASE / "model.pkl"
METRICS_FILE = BASE / "metrics.json"

# Business-level groups used by the Streamlit app.
# The original dataset contains 77 fine-grained intents.
CARD_EXACT = {
    "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire",
    "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "change_pin",
    "compromised_card", "contactless_not_working", "get_physical_card", "getting_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "order_physical_card", "pin_blocked", "supported_cards_and_currencies",
    "virtual_card_not_working", "visa_or_mastercard",
}

TRANSFER_EXACT = {
    "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed",
    "cancel_transfer", "declined_transfer", "failed_transfer", "receiving_money",
    "pending_transfer", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
}

BILLING_EXACT = {
    "automatic_top_up", "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
    "declined_cash_withdrawal", "declined_card_payment", "direct_debit_payment_not_recognised",
    "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
}

ACCOUNT_EXACT = {
    "age_limit", "atm_support", "balance_not_updated_after_cheque_or_cash_deposit",
    "country_support", "disposable_card_limits", "edit_personal_details",
    "fiat_currency_support", "get_disposable_virtual_card", "lost_or_stolen_phone",
    "passcode_forgotten", "terminate_account", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "why_verify_identity",
}


def map_intent(intent: str) -> str:
    if intent in CARD_EXACT:
        return "card"
    if intent in TRANSFER_EXACT:
        return "transfer"
    if intent in BILLING_EXACT:
        return "billing"
    if intent in ACCOUNT_EXACT:
        return "account"
    raise ValueError(f"Unmapped BANKING77 intent: {intent}")


def main():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "BANKING77 files are missing. Run download_data.py first."
        )

    train = pd.read_csv(TRAIN_FILE).dropna(subset=["text", "category"])
    test = pd.read_csv(TEST_FILE).dropna(subset=["text", "category"])

    train["label"] = train["category"].map(map_intent)
    test["label"] = test["category"].map(map_intent)

    # TF-IDF captures informative words/phrases; Logistic Regression handles
    # the resulting high-dimensional sparse text features efficiently.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])

    labels = ["account", "billing", "card", "transfer"]
    accuracy = accuracy_score(test["label"], predictions)
    macro_f1 = f1_score(test["label"], predictions, average="macro")
    report = classification_report(
        test["label"], predictions, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(test["label"], predictions, labels=labels).tolist()

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "dataset": "BANKING77",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "original_intents": int(train["category"].nunique()),
        "business_classes": labels,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    print("Model saved:", MODEL_FILE)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(test["label"], predictions, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
