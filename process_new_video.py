import sys
import os
import subprocess
from pathlib import Path


def process_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Nie znaleziono pliku: {video_path}")
        return False

    video_name = Path(video_path).stem
    results_dir = f"results_{video_name}"

    print(f"🎬 Przetwarzanie wideo: {video_path}")
    print(f"📁 Wyniki będą zapisane w: {results_dir}")

    cmd = [
        "python", "scripts/demo_video_simple.py",
        "--video", video_path,
        "--out_dir", results_dir
    ]

    print(f"🚀 Uruchamiam HybrIK...")
    print(f"Komenda: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print("✅ HybrIK zakończony pomyślnie!")

        pk_file = os.path.join(results_dir, "res.pk")
        video_2d = os.path.join(results_dir, f"res_2d_{video_name}.mp4")

        print(f"\n📋 Wygenerowane pliki:")
        if os.path.exists(pk_file):
            print(f"✅ Dane 3D: {pk_file}")
        else:
            print(f"❌ Brak pliku: {pk_file}")

        if os.path.exists(video_2d):
            print(f"✅ Wideo 2D: {video_2d}")
        else:
            print(f"❌ Brak pliku: {video_2d}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd podczas przetwarzania:")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python process_new_video.py <ścieżka_do_wideo>")
        print("Przykład: python process_new_video.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    process_video(video_path)
