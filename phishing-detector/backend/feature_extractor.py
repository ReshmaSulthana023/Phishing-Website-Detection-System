"""
feature_extractor.py
Extracts numerical features from a URL for phishing detection,
and generates human-readable "risk factor" explanations.

These features are used both for training the Random Forest model
and for real-time prediction.
"""

import re
from urllib.parse import urlparse

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "bank",
    "confirm", "signin", "password", "wallet", "paypal", "billing",
    "suspend", "limited", "alert", "ebayisapi", "webscr"
]

# Well-known brand names frequently impersonated in phishing
KNOWN_BRANDS = [
    "amazon", "paypal", "google", "facebook", "apple", "microsoft",
    "netflix", "bankofamerica", "chase", "wellsfargo", "instagram",
    "whatsapp", "outlook", "linkedin", "ebay"
]

# Trusted / official second-level domains for the brands above
OFFICIAL_DOMAINS = {
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amazon.de"],
    "paypal": ["paypal.com"],
    "google": ["google.com"],
    "facebook": ["facebook.com"],
    "apple": ["apple.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com"],
    "netflix": ["netflix.com"],
    "bankofamerica": ["bankofamerica.com"],
    "chase": ["chase.com"],
    "wellsfargo": ["wellsfargo.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "outlook": ["outlook.com", "live.com"],
    "linkedin": ["linkedin.com"],
    "ebay": ["ebay.com"],
}

IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$"
)

# The exact ordered list of feature names used by the model.
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_subdomains",
    "has_at_symbol",
    "has_https",
    "has_ip",
    "num_digits",
    "suspicious_keyword_count",
    "url_depth",
    "has_double_slash_redirect",
    "brand_impersonation",
]


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    return url


def _get_domain(parsed) -> str:
    domain = parsed.netloc or ""
    if "@" in domain:
        domain = domain.split("@")[-1]
    if ":" in domain:
        domain = domain.split(":")[0]
    return domain.lower()


def extract_features(raw_url: str) -> dict:
    """Return a dict of {feature_name: numeric_value} for a given URL."""
    url = _normalize_url(raw_url)
    parsed = urlparse(url)
    domain = _get_domain(parsed)
    full_url_lower = url.lower()

    # Basic counts
    url_length = len(url)
    domain_length = len(domain)
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_digits = sum(c.isdigit() for c in url)

    # Subdomain count: domain parts minus the registered domain (sld + tld)
    domain_parts = [p for p in domain.split(".") if p]
    num_subdomains = max(0, len(domain_parts) - 2)

    # Special characters / structure
    has_at_symbol = 1 if "@" in url else 0
    has_https = 1 if parsed.scheme == "https" else 0

    # IP address as hostname
    has_ip = 1 if IP_PATTERN.match(domain) else 0

    # Suspicious keyword matches
    suspicious_keyword_count = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url_lower
    )

    # Path depth
    path_segments = [seg for seg in parsed.path.split("/") if seg]
    url_depth = len(path_segments)

    # "//" appearing after the protocol indicates a possible redirect trick
    after_scheme = url.split("://", 1)[-1]
    has_double_slash_redirect = 1 if "//" in after_scheme else 0

    # Brand impersonation: brand name appears in domain but domain is not
    # one of the brand's official domains.
    brand_impersonation = 0
    registered_domain = ".".join(domain_parts[-2:]) if len(domain_parts) >= 2 else domain
    for brand in KNOWN_BRANDS:
        if brand in domain.replace("-", "").replace(".", ""):
            if registered_domain not in OFFICIAL_DOMAINS.get(brand, []):
                brand_impersonation = 1
                break

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_subdomains": num_subdomains,
        "has_at_symbol": has_at_symbol,
        "has_https": has_https,
        "has_ip": has_ip,
        "num_digits": num_digits,
        "suspicious_keyword_count": suspicious_keyword_count,
        "url_depth": url_depth,
        "has_double_slash_redirect": has_double_slash_redirect,
        "brand_impersonation": brand_impersonation,
    }


def features_to_vector(features: dict) -> list:
    """Convert a feature dict into an ordered list matching FEATURE_NAMES."""
    return [features[name] for name in FEATURE_NAMES]


def generate_reasons(raw_url: str, features: dict, prediction: int) -> list:
    """
    Produce a list of human-readable risk-factor strings explaining
    the prediction. `prediction` is 1 for phishing, 0 for legitimate.
    """
    url = _normalize_url(raw_url)
    full_url_lower = url.lower()
    reasons = []

    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in full_url_lower]
    if found_keywords:
        kw_list = ", ".join(f'"{k}"' for k in found_keywords[:3])
        reasons.append(f"Contains suspicious keyword {kw_list}")

    if features["url_length"] > 75:
        reasons.append("URL length is unusually high (exceeds 75 characters)")
    elif features["url_length"] > 54:
        reasons.append("URL length exceeds the normal range")

    if features["num_subdomains"] >= 3:
        reasons.append("Uses an excessive number of subdomains")
    elif features["num_subdomains"] >= 2:
        reasons.append("Uses multiple subdomains")

    if features["num_hyphens"] >= 1:
        reasons.append("Contains hyphen(s) in the URL/domain")

    if features["has_at_symbol"]:
        reasons.append('Contains "@" symbol, which can be used to obscure the real destination')

    if not features["has_https"]:
        reasons.append("Does not use a secure HTTPS connection")

    if features["has_ip"]:
        reasons.append("Uses a raw IP address instead of a domain name")

    if features["has_double_slash_redirect"]:
        reasons.append("Contains '//' redirection pattern after the domain")

    if features["brand_impersonation"]:
        reasons.append("Domain structure resembles a known brand but is not the official domain")

    if features["num_digits"] > 5:
        reasons.append("Contains an unusually high number of digits")

    if features["url_depth"] > 4:
        reasons.append("URL path is unusually deep (many nested segments)")

    if not reasons:
        if prediction == 0:
            reasons.append("No suspicious patterns detected")
            reasons.append("Uses standard domain structure")
            if features["has_https"]:
                reasons.append("Uses a secure HTTPS connection")
        else:
            reasons.append("Statistical pattern of URL structure resembles known phishing samples")

    return reasons
