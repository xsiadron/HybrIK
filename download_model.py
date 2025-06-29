import gdown
import os

# Create pretrained_models directory if it doesn't exist
os.makedirs('pretrained_models', exist_ok=True)

# Download HybrIK HRNet model (with 3DPW training)
print("Downloading HybrIK HRNet model...")
url = 'https://drive.google.com/uc?id=1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW'
output = 'pretrained_models/hybrik_hrnet.pth'
gdown.download(url, output, quiet=False)

print("Download completed!")
print(f"Model saved to: {output}")
