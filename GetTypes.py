import json
from collections import Counter

# ---------- CONFIG ----------
INPUT_JSON = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\data\\merged_data_for_poc.json"   # <-- your 40k JSON file

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect whether it's a list or wrapped in dict
    if isinstance(data, dict) and "items" in data:
        products = data["items"]
    elif isinstance(data, list):
        products = data
    else:
        raise ValueError("Unexpected JSON format – should be list of products or dict with 'items' key.")

    missing_group = 0
    all_group_values = []

    for prod in products:
        feats = prod.get("productFeatures", [])
        group_values = [f["value"] for f in feats if f["featureId"] == "system.productgroup"]

        if not group_values:
            missing_group += 1
        else:
            all_group_values.extend(group_values)

    # Count unique group values
    group_counts = Counter(all_group_values)

    print("✅ Scan complete.")
    print(f"Total products: {len(products)}")
    print(f"Missing system.productgroup: {missing_group}")
    print(f"Unique system.productgroup values: {len(group_counts)}")

    print("\nTop 20 system.productgroup values:")
    for val, cnt in group_counts.most_common(20):
        print(f"{val}: {cnt}")

if __name__ == "__main__":
    main()
