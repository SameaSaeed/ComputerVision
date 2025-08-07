**Installations:**


sudo apt install wget unzip



Download and install MinIO:
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/



**Steps**



**Step 1:** Set Up a Local Cloud Storage System (MinIO)



Create a data directory and run MinIO server:
mkdir -p ~/minio-data
export MINIO\_ROOT\_USER=minioadmin
export MINIO\_ROOT\_PASSWORD=minioadmin
minio server ~/minio-data --console-address ":9001"



Open your browser and access MinIO:
Console: http://localhost:9001
Login with:
 Username: minioadmin
 Password: minioadmin



Create a bucket (e.g., imagedata) and upload a few sample .jpg or .png images, including one or two unusual ones (anomalies).



**Step 2:** Download Images from MinIO for Processing



Install MinIO client:
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/



Configure client and download images:
mc alias set localminio http://localhost:9000 minioadmin minioadmin
mc ls localminio/imagedata
mc cp --recursive localminio/imagedata ~/minio-images



**Step 3**: Perform Image Similarity-Based Anomaly Detection

python3 detect\_anomalies.py

