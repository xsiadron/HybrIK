"""
Skrypt do automatycznego przetwarzania wszystkich plików wideo z folderu input
Każdy plik wideo zostanie przetworzony przez HybrIK i wyniki zostaną zapisane w folderze output

Struktura wyników:
output/
  video_name/
    raw_images/
    res_2d_images/  
    res_2d_video_name.mp4
    res.pk

Użycie: python batch_process.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def get_video_files(input_dir):
    """Znajdź wszystkie pliki wideo w folderze input"""
    video_extensions = ['.mp4', '.avi', '.mov',
                        '.mkv', '.wmv', '.flv', '.webm']
    video_files = []

    if not os.path.exists(input_dir):
        print(f"❌ Folder {input_dir} nie istnieje")
        return video_files

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file_path)

    return video_files


def process_single_video(video_path, output_base_dir):
    """Przetwórz jeden plik wideo"""

    # Pobierz nazwę pliku bez rozszerzenia
    video_name = Path(video_path).stem
    video_file_name = Path(video_path).name

    # Utwórz folder tymczasowy dla wyników
    temp_results_dir = f"temp_results_{video_name}"

    # Utwórz docelowy folder w output
    final_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"\n🎬 Przetwarzanie wideo: {video_file_name}")
    print(f"📁 Wyniki będą zapisane w: {final_output_dir}")

    # Uruchom demo HybrIK
    cmd = [
        "python", "scripts/demo_video_simple.py",
        "--video-name", video_path,
        "--out-dir", temp_results_dir,
        "--save-pk",
        "--save-img"
    ]

    print(f"🚀 Uruchamiam HybrIK...")
    print(f"Komenda: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        print("✅ HybrIK zakończony pomyślnie!")

        # Przenieś wyniki do docelowego folderu
        if os.path.exists(temp_results_dir):
            print(f"📦 Przenoszenie wyników do {final_output_dir}...")

            # Przenieś wszystkie pliki i foldery
            for item in os.listdir(temp_results_dir):
                source = os.path.join(temp_results_dir, item)
                destination = os.path.join(final_output_dir, item)

                if os.path.isdir(source):
                    if os.path.exists(destination):
                        shutil.rmtree(destination)
                    shutil.move(source, destination)
                else:
                    if os.path.exists(destination):
                        os.remove(destination)
                    shutil.move(source, destination)

            # Usuń folder tymczasowy
            if os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)

            # Sprawdź wygenerowane pliki
            expected_files = {
                "res.pk": "Dane 3D",
                f"res_2d_{video_name}.mp4": "Wideo 2D z wynikami",
                "raw_images": "Folder z surowymi obrazami",
                "res_2d_images": "Folder z obrazami wyników 2D"
            }

            print(f"\n📋 Sprawdzanie wygenerowanych plików:")
            for filename, description in expected_files.items():
                filepath = os.path.join(final_output_dir, filename)
                if os.path.exists(filepath):
                    if os.path.isdir(filepath):
                        file_count = len(os.listdir(filepath))
                        print(
                            f"✅ {description}: {filename} ({file_count} plików)")
                    else:
                        file_size = os.path.getsize(
                            filepath) / (1024*1024)  # MB
                        print(
                            f"✅ {description}: {filename} ({file_size:.1f} MB)")
                else:
                    print(f"❌ Brak: {filename} ({description})")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Błąd podczas przetwarzania {video_file_name}:")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")

        # Usuń folder tymczasowy w przypadku błędu
        if os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

        return False


def main():
    """Główna funkcja skryptu"""

    # Ścieżki folderów
    input_dir = "input"
    output_dir = "output"

    print("🎯 HybrIK Batch Processor")
    print("=" * 50)

    # Sprawdź czy folder input istnieje
    if not os.path.exists(input_dir):
        print(f"❌ Folder '{input_dir}' nie istnieje!")
        print(
            f"Utwórz folder '{input_dir}' i umieść w nim pliki wideo do przetworzenia.")
        return

    # Utwórz folder output jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Znajdź wszystkie pliki wideo
    video_files = get_video_files(input_dir)

    if not video_files:
        print(f"❌ Nie znaleziono plików wideo w folderze '{input_dir}'")
        print("Obsługiwane formaty: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
        return

    print(f"📹 Znaleziono {len(video_files)} plików wideo:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {Path(video_file).name}")

    print("\n🚀 Rozpoczynam przetwarzanie...")

    # Przetwórz każdy plik wideo
    successful = 0
    failed = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"\n" + "="*50)
        print(f"📽️  Plik {i}/{len(video_files)}")

        success = process_single_video(video_path, output_dir)

        if success:
            successful += 1
            print(f"✅ {Path(video_path).name} - przetworzony pomyślnie")
        else:
            failed += 1
            print(f"❌ {Path(video_path).name} - błąd przetwarzania")

    # Podsumowanie
    print("\n" + "="*50)
    print("📊 PODSUMOWANIE")
    print(f"✅ Pomyślnie przetworzono: {successful} plików")
    print(f"❌ Błędy: {failed} plików")
    print(f"📁 Wyniki zapisane w folderze: {output_dir}")

    if successful > 0:
        print(f"\n🎉 Przetwarzanie zakończone!")
        print(f"Sprawdź folder '{output_dir}' aby zobaczyć wyniki.")


if __name__ == "__main__":
    main()
