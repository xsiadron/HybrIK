import os
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm


class HybrIKBatchProcessor:
    def __init__(self, input_dir="input", output_dir="output", use_stabilization=True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_stabilization = use_stabilization
        self.video_extensions = ['.mp4', '.avi',
                                 '.mov', '.mkv', '.wmv', '.flv', '.webm']
        self.processed_count = 0
        self.failed_count = 0

    def find_video_files(self):
        if not os.path.exists(self.input_dir):
            print(f"❌ Input directory '{self.input_dir}' does not exist")
            return []

        video_files = []
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in self.video_extensions):
                video_files.append(file_path)

        return sorted(video_files)

    def setup_directories(self, video_name):
        temp_dir = f"temp_results_{video_name}"
        final_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(final_dir, exist_ok=True)
        return temp_dir, final_dir

    def run_hybrik_processing(self, video_path, temp_dir, use_stabilization=True):
        # Choose stabilized or original script
        script_name = "scripts/demo_video_stabilized.py" if use_stabilization else "scripts/demo_video_simple.py"
        
        cmd = [
            "python", script_name,
            "--video-name", video_path,
            "--out-dir", temp_dir,
            "--save-pk",
            "--save-img"
        ]
        
        # Add stabilization parameters if using stabilized script
        if use_stabilization:
            cmd.extend([
                "--smoothing-alpha", "0.1",  # Bardzo mocne wygładzanie (0.1 = bardzo gładkie)
                "--gaussian-sigma", "2.5",   # Silne post-processing smoothing
                "--confidence-threshold", "0.3"  # Niski próg dla lepszego trackingu
            ])

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True)
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr

    def move_results(self, temp_dir, final_dir):
        if not os.path.exists(temp_dir):
            return False

        try:
            for item in os.listdir(temp_dir):
                source = os.path.join(temp_dir, item)
                destination = os.path.join(final_dir, item)

                if os.path.exists(destination):
                    if os.path.isdir(destination):
                        shutil.rmtree(destination)
                    else:
                        os.remove(destination)

                shutil.move(source, destination)

            shutil.rmtree(temp_dir)
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error moving results - {e}")
            return False

    def verify_results(self, final_dir, video_name):
        expected_files = {
            "res.pk": "3D pose data",
            f"res_2d_{video_name}.mp4": "2D result video",
            "raw_images": "Raw frame images",
            "res_2d_images": "2D result images"
        }

        print(f"   📋 Verifying generated files:")
        for filename, description in expected_files.items():
            filepath = os.path.join(final_dir, filename)
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    file_count = len(os.listdir(filepath))
                    print(
                        f"      ✅ {description}: {filename} ({file_count} files)")
                else:
                    file_size = os.path.getsize(filepath) / (1024*1024)
                    print(
                        f"      ✅ {description}: {filename} ({file_size:.1f} MB)")
            else:
                print(f"      ❌ Missing: {filename} ({description})")

    def process_single_video(self, video_path, video_index, total_videos):
        video_name = Path(video_path).stem
        video_filename = Path(video_path).name

        print(
            f"\n🎬 Processing video {video_index}/{total_videos}: {video_filename}")

        temp_dir, final_dir = self.setup_directories(video_name)
        print(f"   📁 Output directory: {final_dir}")

        print(f"   🚀 Running HybrIK inference...")
        if self.use_stabilization:
            print(f"   🎯 Using temporal stabilization for smooth animations")
        success, stdout, stderr = self.run_hybrik_processing(
            video_path, temp_dir, self.use_stabilization)

        if success:
            print(f"   ✅ HybrIK processing completed successfully")

            print(f"   📦 Moving results to output directory...")
            if self.move_results(temp_dir, final_dir):
                self.verify_results(final_dir, video_name)
                self.processed_count += 1
                return True
            else:
                print(f"   ❌ Failed to move results")
                self.failed_count += 1
                return False
        else:
            print(f"   ❌ HybrIK processing failed")
            if stderr:
                print(f"   Error details: {stderr[:200]}...")

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            self.failed_count += 1
            return False

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"✅ Successfully processed: {self.processed_count} videos")
        print(f"❌ Failed: {self.failed_count} videos")
        print(f"📁 Results saved in: {self.output_dir}/")

        if self.processed_count > 0:
            print(f"\n🎉 Batch processing completed!")
            print(f"Check the '{self.output_dir}' directory for results.")

    def run(self):
        print("🚀 HybrIK Batch Video Processor with Temporal Stabilization")
        print("="*70)
        
        if self.use_stabilization:
            print("🎯 Stabilization enabled - animations will be smoother!")
        else:
            print("⚠️  Stabilization disabled - using original processing")

        if not os.path.exists(self.input_dir):
            print(f"❌ Input directory '{self.input_dir}' does not exist!")
            print(
                f"Create the '{self.input_dir}' folder and place video files to process.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        video_files = self.find_video_files()

        if not video_files:
            print(f"❌ No video files found in '{self.input_dir}'")
            print(f"Supported formats: {', '.join(self.video_extensions)}")
            return

        print(f"📹 Found {len(video_files)} video files:")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {Path(video_file).name}")

        print(f"\n� Starting batch processing...")

        for i, video_path in enumerate(video_files, 1):
            success = self.process_single_video(
                video_path, i, len(video_files))

            if success:
                print(f"✅ {Path(video_path).name} - processed successfully")
            else:
                print(f"❌ {Path(video_path).name} - processing failed")

        self.print_summary()


def main():
    processor = HybrIKBatchProcessor(use_stabilization=True)  # Enable stabilization by default
    processor.run()


if __name__ == "__main__":
    main()
