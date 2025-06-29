import gdown
import os
import zipfile
from tqdm import tqdm


class ModelDownloader:
    
    def __init__(self):
        self.directories = ['pretrained_models', 'model_files']
        self.hybrik_model_url = 'https://drive.google.com/uc?id=1gp3549vIEKfbc8SDQ-YF3Idi1aoR3DkW'
        self.hybrik_model_path = 'pretrained_models/hybrik_hrnet.pth'
        self.model_files_url = 'https://drive.google.com/uc?id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV'
        self.model_files_zip = 'model_files.zip'
        self.required_files = [
            'pretrained_models/hybrik_hrnet.pth',
            'model_files/J_regressor_h36m.npy'
        ]

    def create_directories(self):
        print("📁 Creating necessary directories...")
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
            print(f"   Directory ready: {directory}/")
        return True

    def download_hybrik_model(self):
        print("\n🔄 Downloading HybrIK HRNet model...")
        
        try:
            gdown.download(self.hybrik_model_url, self.hybrik_model_path, quiet=False)
            print(f"✅ HybrIK model downloaded successfully")
            print(f"   Location: {self.hybrik_model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to download HybrIK model: {e}")
            return False

    def download_and_extract_model_files(self):
        print("\n🔄 Downloading model files archive...")
        
        try:
            print("   Downloading model_files.zip from Google Drive...")
            gdown.download(self.model_files_url, self.model_files_zip, quiet=False)

            print("   Extracting archive contents...")
            with zipfile.ZipFile(self.model_files_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"   Found {len(file_list)} files in archive")
                zip_ref.extractall('.')

            print("   Cleaning up temporary files...")
            os.remove(self.model_files_zip)

            print("✅ Model files extracted successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to download model files: {e}")
            print("   Manual download required:")
            print("   URL: https://drive.google.com/file/d/1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV/view")
            print("   Extract to HybrIK root directory")
            return False

    def verify_installation(self):
        print("\n🔍 Verifying installation...")

        all_present = True
        for file_path in self.required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   ✅ {file_path} ({file_size:.1f} MB)")
            else:
                print(f"   ❌ {file_path} - MISSING")
                all_present = False

        return all_present

    def download_all(self):
        print("🚀 HybrIK Model Downloader")
        print("=" * 50)

        self.create_directories()

        success_hybrik = self.download_hybrik_model()
        success_model_files = self.download_and_extract_model_files()

        if self.verify_installation():
            print("\n🎉 Model download completed successfully!")
            print("   All required files are present and ready to use.")
            return True
        else:
            print("\n⚠️  Model download incomplete!")
            print("   Some required files are missing. Please check the errors above.")
            return False


if __name__ == "__main__":
    downloader = ModelDownloader()
    downloader.download_all()
