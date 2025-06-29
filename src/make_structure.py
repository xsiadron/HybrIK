import os


class StructureCreator:
    
    def __init__(self):
        self.directories = [
            'input',
            'output'
        ]
    
    def create_directories(self):
        print("📂 Creating project directory structure...")
        
        created_dirs = []
        for directory in self.directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   ✅ Directory created/verified: {directory}/")
                created_dirs.append(directory)
            except Exception as e:
                print(f"   ❌ Failed to create directory {directory}: {e}")
                return False
        
        return True
    
    def verify_structure(self):
        print("\n🔍 Verifying directory structure...")
        
        all_present = True
        for directory in self.directories:
            if os.path.exists(directory) and os.path.isdir(directory):
                print(f"   ✅ {directory}/ - EXISTS")
            else:
                print(f"   ❌ {directory}/ - MISSING")
                all_present = False
        
        return all_present
    
    def create_structure(self):
        print("🏗️  HybrIK Structure Creator")
        print("=" * 50)
        
        if self.create_directories():
            if self.verify_structure():
                print("\n🎉 Project structure created successfully!")
                print("   All required directories are ready.")
                return True
            else:
                print("\n⚠️  Structure creation incomplete!")
                print("   Some directories are missing.")
                return False
        else:
            print("\n❌ Failed to create project structure!")
            return False


if __name__ == "__main__":
    creator = StructureCreator()
    creator.create_structure()
