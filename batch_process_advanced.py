"""
Advanced batch processor with configurable stabilization parameters.
Allows fine-tuning of stabilization settings for optimal results.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


class AdvancedHybrIKBatchProcessor:
    def __init__(self, input_dir="input", output_dir="output", 
                 smoothing_alpha=0.3, gaussian_sigma=1.5, confidence_threshold=0.4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.smoothing_alpha = smoothing_alpha
        self.gaussian_sigma = gaussian_sigma
        self.confidence_threshold = confidence_threshold
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
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

    def run_hybrik_processing(self, video_path, temp_dir):
        cmd = [
            "python", "scripts/demo_video_stabilized.py",
            "--video-name", video_path,
            "--out-dir", temp_dir,
            "--save-pk",
            "--save-img",
            "--smoothing-alpha", str(self.smoothing_alpha),
            "--gaussian-sigma", str(self.gaussian_sigma),
            "--confidence-threshold", str(self.confidence_threshold)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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
            "res_stabilized.pk": "3D pose data (stabilized)",
            f"res_2d_{video_name}_stabilized.mp4": "2D result video (stabilized)",
            "raw_images": "Raw frame images",
            "res_2d_images": "2D result images"
        }

        print(f"   📋 Verifying generated files:")
        for filename, description in expected_files.items():
            filepath = os.path.join(final_dir, filename)
            if os.path.exists(filepath):
                if os.path.isdir(filepath):
                    file_count = len(os.listdir(filepath))
                    print(f"      ✅ {description}: {filename} ({file_count} files)")
                else:
                    file_size = os.path.getsize(filepath) / (1024*1024)
                    print(f"      ✅ {description}: {filename} ({file_size:.1f} MB)")
            else:
                print(f"      ❌ Missing: {filename} ({description})")

    def process_single_video(self, video_path, video_index, total_videos):
        video_name = Path(video_path).stem
        video_filename = Path(video_path).name

        print(f"\n🎬 Processing video {video_index}/{total_videos}: {video_filename}")

        temp_dir, final_dir = self.setup_directories(video_name)
        print(f"   📁 Output directory: {final_dir}")

        print(f"   🚀 Running HybrIK inference with stabilization...")
        print(f"   🎯 Stabilization parameters:")
        print(f"      - Smoothing Alpha: {self.smoothing_alpha}")
        print(f"      - Gaussian Sigma: {self.gaussian_sigma}")
        print(f"      - Confidence Threshold: {self.confidence_threshold}")
        
        success, stdout, stderr = self.run_hybrik_processing(video_path, temp_dir)

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
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"✅ Successfully processed: {self.processed_count} videos")
        print(f"❌ Failed: {self.failed_count} videos")
        print(f"📁 Results saved in: {self.output_dir}/")
        print(f"🎯 Stabilization settings used:")
        print(f"   - Smoothing Alpha: {self.smoothing_alpha}")
        print(f"   - Gaussian Sigma: {self.gaussian_sigma}")
        print(f"   - Confidence Threshold: {self.confidence_threshold}")

        if self.processed_count > 0:
            print(f"\n🎉 Batch processing completed with temporal stabilization!")
            print(f"Check the '{self.output_dir}' directory for smooth animation results.")

    def run(self):
        print("🚀 Advanced HybrIK Batch Video Processor with Temporal Stabilization")
        print("="*80)
        print("🎯 This version provides stable, smooth animations with reduced jitter!")

        if not os.path.exists(self.input_dir):
            print(f"❌ Input directory '{self.input_dir}' does not exist!")
            print(f"Create the '{self.input_dir}' folder and place video files to process.")
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

        print(f"\n🔧 Stabilization Configuration:")
        print(f"   🎛️  Smoothing Alpha: {self.smoothing_alpha} (lower = smoother, 0.1-0.5 recommended)")
        print(f"   🌊 Gaussian Sigma: {self.gaussian_sigma} (higher = smoother, 0.5-2.0 recommended)")
        print(f"   🎯 Confidence Threshold: {self.confidence_threshold} (lower = better tracking)")

        print(f"\n🚀 Starting batch processing...")

        for i, video_path in enumerate(video_files, 1):
            success = self.process_single_video(video_path, i, len(video_files))

            if success:
                print(f"✅ {Path(video_path).name} - processed successfully with stabilization")
            else:
                print(f"❌ {Path(video_path).name} - processing failed")

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Advanced HybrIK Batch Processor with Temporal Stabilization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings (recommended)
  python batch_process_advanced.py

  # Very smooth animations (for shaky videos)
  python batch_process_advanced.py --smoothing-alpha 0.1 --gaussian-sigma 2.0

  # Faster tracking (for stable videos)
  python batch_process_advanced.py --smoothing-alpha 0.5 --gaussian-sigma 0.5

  # Custom input/output directories
  python batch_process_advanced.py --input my_videos --output my_results
        """
    )
    
    parser.add_argument('--input', default='input', 
                        help='Input directory containing video files (default: input)')
    parser.add_argument('--output', default='output', 
                        help='Output directory for results (default: output)')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3,
                        help='Temporal smoothing factor (0.1=very smooth, 0.5=less smooth, default: 0.3)')
    parser.add_argument('--gaussian-sigma', type=float, default=1.5,
                        help='Post-processing Gaussian smoothing (0.5-2.0, default: 1.5)')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                        help='Detection confidence threshold (0.3-0.7, default: 0.4)')

    args = parser.parse_args()

    # Validate parameters
    if not (0.0 <= args.smoothing_alpha <= 1.0):
        print("❌ Error: smoothing-alpha must be between 0.0 and 1.0")
        return
        
    if not (0.0 <= args.gaussian_sigma <= 5.0):
        print("❌ Error: gaussian-sigma must be between 0.0 and 5.0")
        return
        
    if not (0.1 <= args.confidence_threshold <= 1.0):
        print("❌ Error: confidence-threshold must be between 0.1 and 1.0")
        return

    processor = AdvancedHybrIKBatchProcessor(
        input_dir=args.input,
        output_dir=args.output,
        smoothing_alpha=args.smoothing_alpha,
        gaussian_sigma=args.gaussian_sigma,
        confidence_threshold=args.confidence_threshold
    )
    processor.run()


if __name__ == "__main__":
    main()
