"""
HybrIK Batch Video Processor

Automatically processes all video files from the input folder using HybrIK
and saves results to organized output directories.

Output structure:
output/
  video_name/
    raw_images/
    res_2d_images/  
    res_2d_video_name.mp4
    res.pk

Usage: python batch_process.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm


class HybrIKBatchProcessor:
    """Main class for batch processing videos with HybrIK."""
    
    def __init__(self, input_dir="input", output_dir="output"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        self.processed_count = 0
        self.failed_count = 0
    
    def find_video_files(self):
        """Find all video files in the input directory."""
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
        """Create necessary directories for processing."""
        temp_dir = f"temp_results_{video_name}"
        final_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(final_dir, exist_ok=True)
        return temp_dir, final_dir
    
    def run_hybrik_processing(self, video_path, temp_dir):
        """Execute HybrIK processing on a single video."""
        cmd = [
            "python", "scripts/demo_video_simple.py",
            "--video-name", video_path,
            "--out-dir", temp_dir,
            "--save-pk",
            "--save-img"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr
    
    def move_results(self, temp_dir, final_dir):
        """Move processing results from temporary to final directory."""
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
        """Verify and report generated files."""
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
                    print(f"      ✅ {description}: {filename} ({file_count} files)")
                else:
                    file_size = os.path.getsize(filepath) / (1024*1024)
                    print(f"      ✅ {description}: {filename} ({file_size:.1f} MB)")
            else:
                print(f"      ❌ Missing: {filename} ({description})")
    
    def process_single_video(self, video_path, video_index, total_videos):
        """Process a single video file through HybrIK pipeline."""
        video_name = Path(video_path).stem
        video_filename = Path(video_path).name
        
        print(f"\n🎬 Processing video {video_index}/{total_videos}: {video_filename}")
        
        temp_dir, final_dir = self.setup_directories(video_name)
        print(f"   📁 Output directory: {final_dir}")
        
        print(f"   🚀 Running HybrIK inference...")
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
        """Print processing summary."""
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"✅ Successfully processed: {self.processed_count} videos")
        print(f"❌ Failed: {self.failed_count} videos")
        print(f"📁 Results saved in: {self.output_dir}/")
        
        if self.processed_count > 0:
            print(f"\n🎉 Batch processing completed!")
            print(f"Check the '{self.output_dir}' directory for results.")
    
    def run(self):
        """Main processing pipeline."""
        print("🚀 HybrIK Batch Video Processor")
        print("="*60)
        
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
        
        print(f"\n� Starting batch processing...")
        
        for i, video_path in enumerate(video_files, 1):
            success = self.process_single_video(video_path, i, len(video_files))
            
            if success:
                print(f"✅ {Path(video_path).name} - processed successfully")
            else:
                print(f"❌ {Path(video_path).name} - processing failed")
        
        self.print_summary()


def main():
    """Entry point for the batch processor."""
    processor = HybrIKBatchProcessor()
    processor.run()


if __name__ == "__main__":
    main()
