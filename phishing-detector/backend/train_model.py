"""
train_model.py
Generates a synthetic-but-realistic phishing/legitimate URL dataset,
extracts features, trains a Random Forest classifier, evaluates it,
and saves the trained model + metadata to disk.

Run:
    python train_model.py
"""

import json
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from feature_extractor import extract_features, features_to_vector, FEATURE_NAMES

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Generate synthetic dataset
# ---------------------------------------------------------------------------

LEGIT_DOMAINS = [
    "amazon.com", "google.com", "facebook.com", "wikipedia.org",
    "github.com", "microsoft.com", "apple.com", "netflix.com",
    "linkedin.com", "twitter.com", "reddit.com", "stackoverflow.com",
    "yahoo.com", "instagram.com", "whatsapp.com", "ebay.com",
    "paypal.com", "bankofamerica.com", "chase.com", "wellsfargo.com",
    "nytimes.com", "bbc.com", "cnn.com", "spotify.com", "dropbox.com",
    "adobe.com", "salesforce.com", "zoom.us", "office.com", "outlook.com",
]

LEGIT_PATHS = [
    "", "/", "/home", "/products", "/products/electronics",
    "/account/settings", "/articles/2024/tech-news", "/search?q=shoes",
    "/blog/post-123", "/help/support", "/about-us", "/careers",
    "/news/world", "/store/category/shoes", "/profile/johndoe",
]

PHISHING_BRANDS = [
    "amazon", "paypal", "apple", "netflix", "facebook", "instagram",
    "bankofamerica", "chase", "wellsfargo", "microsoft", "google",
    "ebay", "linkedin", "outlook", "whatsapp",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "bank",
    "confirm", "signin", "password", "wallet", "billing", "suspend",
]

SUSPICIOUS_TLDS = [
    "com-secure.info", "com-verify.xyz", "account-update.tk",
    "secure-login.ml", "com.security-check.ga", "net.online-verify.cf",
    "login-confirm.top", "id-verify.club", "support-center.live",
]

RANDOM_TLDS = ["xyz", "top", "club", "info", "tk", "ml", "ga", "cf", "online", "site"]


def random_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    use_www = random.random() < 0.5
    host = ("www." + domain) if use_www else domain
    path = random.choice(LEGIT_PATHS)
    scheme = "https" if random.random() < 0.92 else "http"
    return f"{scheme}://{host}{path}"


def random_phishing_url():
    """Build a URL using common phishing patterns."""
    pattern = random.random()
    brand = random.choice(PHISHING_BRANDS)
    keyword = random.choice(SUSPICIOUS_KEYWORDS)

    if pattern < 0.25:
        # brand + keyword + hyphen + suspicious tld
        suffix = random.choice(SUSPICIOUS_TLDS)
        host = f"{brand}-{keyword}.{suffix}"
    elif pattern < 0.45:
        # multiple subdomains impersonating brand
        rand_tld = random.choice(RANDOM_TLDS)
        host = f"{keyword}.{brand}.account-verification.{rand_tld}"
    elif pattern < 0.65:
        # raw IP address
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        host = ip
    elif pattern < 0.80:
        # @ symbol trick
        rand_tld = random.choice(RANDOM_TLDS)
        host = f"{brand}.com@{keyword}-{brand}.{rand_tld}"
    else:
        # long random subdomain chain
        rand_tld = random.choice(RANDOM_TLDS)
        sub1 = random.choice(SUSPICIOUS_KEYWORDS)
        sub2 = random.choice(SUSPICIOUS_KEYWORDS)
        host = f"{sub1}.{sub2}.{brand}-{random.randint(100,999)}.{rand_tld}"

    path_extra = random.choice([
        "", "/login.php", "/signin/account/verify",
        "/wp-content/update/secure", "//redirect/account",
        f"/{keyword}/{random.randint(1000,9999)}/confirm",
    ])
    scheme = "https" if random.random() < 0.35 else "http"
    return f"{scheme}://{host}{path_extra}"


def build_dataset(n_per_class=1500):
    rows = []
    for _ in range(n_per_class):
        url = random_legit_url()
        feats = extract_features(url)
        feats["label"] = 0
        feats["url"] = url
        rows.append(feats)

    for _ in range(n_per_class):
        url = random_phishing_url()
        feats = extract_features(url)
        feats["label"] = 1
        feats["url"] = url
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Train model
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic dataset...")
    df = build_dataset(n_per_class=1500)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset saved to dataset.csv ({len(df)} rows)")

    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])

    print(f"\nTest Accuracy: {accuracy:.4f}\n")
    print(report)

    # Feature importance
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Feature importances:")
    for name, imp in importances:
        print(f"  {name}: {imp:.4f}")

    # Save model + metadata
    joblib.dump(model, "phishing_model.pkl")
    metadata = {
        "feature_names": FEATURE_NAMES,
        "accuracy": round(accuracy, 4),
        "n_estimators": 200,
        "trained_on_samples": len(df),
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel saved to phishing_model.pkl")
    print("Metadata saved to model_metadata.json")


if __name__ == "__main__":
    main()
