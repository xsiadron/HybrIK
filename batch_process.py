import os
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import datetime


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
        script_name = "scripts/demo_video_stabilized.py" if use_stabilization else "scripts/demo_video_simple.py"

        cmd = [
            "python", script_name,
            "--video-name", video_path,
            "--out-dir", temp_dir,
            "--save-pk",
            "--save-img"
        ]

        if use_stabilization:
            cmd.extend([
                "--smoothing-alpha", "0.0001",
                "--gaussian-sigma", "10000",
                "--confidence-threshold", "1",
                "--stabilization-mode", "kalman"
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

    def verify_results(self, final_dir, video_name, elapsed_ms=None):
        expected_files = {
            "res.pk": "3D pose data",
            f"res_2d_{video_name}.mp4": "2D result video",
            "raw_images": "Raw frame images",
            "res_2d_images": "2D result images"
        }
        missing = []
        for filename in expected_files:
            filepath = os.path.join(final_dir, filename)
            if not os.path.exists(filepath):
                missing.append(filename)
        if not missing:
            msg = f'Successfully generated files for "{video_name}.mp4"'
            if elapsed_ms is not None:
                msg += f' in {elapsed_ms}ms'
            self.log(msg, level="SUCCESS")
        else:
            self.log(
                f'Missing files for "{video_name}.mp4": {", ".join(missing)}', level="ERROR")

    def log(self, message, level="INFO"):
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        print(f"{now.strftime('%d.%m.%Y')}|{timestamp}|{level}|{message}")

    def process_single_video(self, video_path, video_index, total_videos):
        video_name = Path(video_path).stem
        video_filename = Path(video_path).name
        self.log(f'Processing video "{video_filename}"', level="INFO")
        temp_dir, final_dir = self.setup_directories(video_name)
        start = datetime.datetime.now()
        success, stdout, stderr = self.run_hybrik_processing(
            video_path, temp_dir, self.use_stabilization)
        elapsed_ms = int((datetime.datetime.now() -
                         start).total_seconds() * 1000)
        if success:
            if self.move_results(temp_dir, final_dir):
                self.verify_results(final_dir, video_name,
                                    elapsed_ms=elapsed_ms)
                self.processed_count += 1
                return True
            else:
                self.log(
                    f'Failed to move results for "{video_filename}"', level="ERROR")
                self.failed_count += 1
                return False
        else:
            self.log(
                f'HybrIK processing failed for "{video_filename}"', level="ERROR")
            if stderr:
                self.log(f'{stderr.strip()}', level="ERROR")
            if stdout:
                self.log(f'{stdout.strip()}', level="ERROR")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            self.failed_count += 1
            return False

    def print_summary(self):
        self.log(
            f'Successfully processed: {self.processed_count} videos', level="INFO")
        self.log(f'Failed: {self.failed_count} videos', level="INFO")
        self.log(f'Results saved in: {self.output_dir}/', level="INFO")
        if self.processed_count > 0:
            self.log(
                f'Batch processing completed! Check the "{self.output_dir}" directory for results.', level="SUCCESS")

    def run(self):
        if not os.path.exists(self.input_dir):
            self.log(
                f'Input directory "{self.input_dir}" does not exist!', level="ERROR")
            self.log(
                f'Create the "{self.input_dir}" folder and place video files to process.', level="ERROR")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        video_files = self.find_video_files()
        if not video_files:
            self.log(
                f'No video files found in "{self.input_dir}"', level="ERROR")
            self.log('Supported formats: ' +
                     ', '.join(self.video_extensions), level="INFO")
            return
        for i, video_path in enumerate(video_files, 1):
            self.process_single_video(video_path, i, len(video_files))
        self.print_summary()


def main():
    processor = HybrIKBatchProcessor(use_stabilization=True)
    processor.run()


if __name__ == "__main__":
    main()
