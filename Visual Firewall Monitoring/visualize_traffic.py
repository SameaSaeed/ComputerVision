import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traffic.csv")

# Drop rows with missing IPs
df = df.dropna(subset=["ip.src", "ip.dst"])

# Count number of packets per source IP
top_sources = df['ip.src'].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_sources.plot(kind='bar', color='skyblue')
plt.title("Top 10 Source IPs by Packet Count")
plt.xlabel("Source IP")
plt.ylabel("Packets")
plt.tight_layout()
plt.savefig("traffic_bar_chart.png")
plt.show()

suspicious = top_sources[top_sources > 100]
if not suspicious.empty:
    print("\nâ ï¸  Potential Threats Detected:")
    print(suspicious)
else:
    print("\nâ  No suspicious IPs found.")
