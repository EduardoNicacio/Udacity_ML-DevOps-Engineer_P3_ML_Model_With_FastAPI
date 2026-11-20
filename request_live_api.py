"""Send one POST request to the deployed Census salary prediction API."""

import requests

# Replace the placeholder with the deployed Render service URL after deployment.
API_URL = "https://<your-render-service>.onrender.com/predict"

# A typical high-income profile (based on Census row with fnlgt 209642).
PAYLOAD = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


def main():
    """Call the live API and print the status code and prediction."""
    response = requests.post(API_URL, json=PAYLOAD, timeout=30)
    print(f"Status code: {response.status_code}")
    # 1 = income >50K, 0 = income <=50K
    print(f"Prediction: {response.json()['prediction']}")


if __name__ == "__main__":
    main()
