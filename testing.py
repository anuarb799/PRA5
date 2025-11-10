import time
import csv
import requests
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd

BASE_URL = "http://serve-sentiment-env.eba-qrxiqe3y.us-east-2.elasticbeanstalk.com/predict"

TEST_CASES = {
    "First True": "Canada is a nice place",
    "Second True": "Cucumbers are vegetables",
    "First Fake": "Aliens live among us",
    "Second Fake": "Aliens like us",
}

N_CALLS = 100

results = []

for label, text in TEST_CASES.items():
    latencies = []
    print(f"Running latency test for {label}...")
    for i in range(N_CALLS):
        start = time.time()
        response = requests.post(BASE_URL, json={"message": text})
        end = time.time()
        latency = (end - start) * 1000  # milliseconds
        results.append({"case": label, "iteration": i + 1, "latency_ms": latency})
        latencies.append(latency)
    print(f"Average latency for {label}: {mean(latencies):.2f} ms")

with open("latency_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case", "iteration", "latency_ms"])
    writer.writeheader()
    writer.writerows(results)

print("Done")

df = pd.DataFrame(results)

plt.figure(figsize=(10,6))
df.boxplot(column="latency_ms", by="case", grid=False)
plt.title("API Latency per Test Case (100 calls each)")
plt.suptitle("")
plt.ylabel("Latency (ms)")
plt.xlabel("Test Case")
plt.savefig("latency_boxplot.png", dpi=300)
plt.show()

avg_latencies = df.groupby("case")["latency_ms"].mean()
print("\nAverage Latency per Test Case:")
print(avg_latencies)
