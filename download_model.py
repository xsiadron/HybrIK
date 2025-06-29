import gdown
import os
import zipfile
from tqdm import tqdm


def create_directories():
    """Create necessary directories for models and data files."""
    directories = ['pretrained_models', 'model_files']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}/")


def download_hybrik_model():
    """Download the main HybrIK HRNet model from Google Drive."""
    print("\n🔄 Downloading HybrIK HRNet model...")
    url = 'https://drive.google.com/uc?id=1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW'
    output = 'pretrained_models/hybrik_hrnet.pth'

    try:
        gdown.download(url, output, quiet=False)
        print(f"✅ HybrIK model downloaded successfully")
        print(f"   Location: {output}")
        return True
    except Exception as e:
        print(f"❌ Failed to download HybrIK model: {e}")
        return False


def download_and_extract_model_files():
    """Download and extract model_files.zip containing additional model data."""
    print("\n🔄 Downloading model files archive...")
    url_zip = 'https://drive.google.com/uc?id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV'
    zip_output = 'model_files.zip'

    try:
        print("   Downloading model_files.zip from Google Drive...")
        gdown.download(url_zip, zip_output, quiet=False)

        print("   Extracting archive contents...")
        with zipfile.ZipFile(zip_output, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files in archive")
            zip_ref.extractall('.')

        print("   Cleaning up temporary files...")
        os.remove(zip_output)

        print("✅ Model files extracted successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to download model files: {e}")
        print("   Manual download required:")
        print(
            "   URL: https://drive.google.com/file/d/1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV/view")
        print("   Extract to HybrIK root directory")
        return False


def verify_installation():
    """Verify that all required files are present."""
    print("\n🔍 Verifying installation...")

    required_files = [
        'pretrained_models/hybrik_hrnet.pth',
        'model_files/J_regressor_h36m.npy'
    ]

    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(
                file_path) / (1024 * 1024)  # Size in MB
            print(f"   ✅ {file_path} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_present = False

    return all_present


def main():
    """Main function to download all required HybrIK model files."""
    print("🚀 HybrIK Model Downloader")
    print("=" * 50)

    create_directories()

    success_hybrik = download_hybrik_model()
    success_model_files = download_and_extract_model_files()

    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("   All required files are present and ready to use.")
    else:
        print("\n⚠️  Installation incomplete!")
        print("   Some required files are missing. Please check the errors above.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
