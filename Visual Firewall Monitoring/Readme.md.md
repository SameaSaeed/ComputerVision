**Install tools:**

sudo apt update

sudo apt install wireshark tshark -y

pip install pandas matplotlib opencv-python



During Wireshark setup, if prompted, allow non-root users to capture packets:

sudo usermod -aG wireshark $USER

newgrp wireshark



**Steps**



**Step 1: Capture Network Traffic with Wireshark or tshark**

1. Start Wireshark GUI:

wireshark

2\.  Or use tshark CLI for faster, scriptable capture:

tshark -i any -a duration:30 -w capture.pcapng

This captures 30 seconds of traffic and saves it to capture.pcapng.



**Step 2: Convert pcap to CSV**

Use tshark to extract essential packet data:

tshark -r capture.pcapng -T fields -e frame.number -e ip.src -e ip.dst -e frame.len -e \_ws.col.Protocol -E header=y -E separator=, -E quote=d -E occurrence=f > traffic.csv

Now you have a structured traffic.csv with key fields.



**Step 3: Visualize Network Traffic**

python3 visualize\_traffic.py



**Step 4: Highlight Suspicious Activity (e.g., Flooding)**

Extend the script to flag possible anomalies:

You can also use OpenCV for custom rendering of heatmaps or visual overlays if desired.

