import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.download_model import ModelDownloader
from src.make_structure import StructureCreator


def setup_hybrik():
    print("🌟 Starting HybrIK Complete Setup")
    print("=" * 60)
    
    try:
        print("\n📥 STEP 1: Downloading model files...")
        downloader = ModelDownloader()
        download_success = downloader.download_all()
        
        if not download_success:
            print("❌ Model download failed. Setup cannot continue.")
            return False
        
        print("\n" + "=" * 60)
        
        print("\n📁 STEP 2: Creating project structure...")
        creator = StructureCreator()
        structure_success = creator.create_structure()
        
        if not structure_success:
            print("❌ Structure creation failed.")
            return False
        
        print("\n" + "=" * 60)
        print("🎊 HybrIK SETUP COMPLETED SUCCESSFULLY! 🎊")
        print("=" * 60)
        print("✅ All model files downloaded")
        print("✅ Project structure created")
        print("🚀 HybrIK is ready to use!")
        print("\nNext steps:")
        print("1. Place your input videos in the 'input/' folder")
        print("2. Run your HybrIK scripts")
        print("3. Check results in the 'output/' folder")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error messages above and try again.")
        return False


def main():
    success = setup_hybrik()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
