@echo off
echo ================================
echo    HybrIK Video Processor
echo ================================
echo.

REM Sprawdź czy środowisko conda jest aktywne
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo BŁĄD: Conda nie jest zainstalowana lub nie jest w PATH
    echo Zainstaluj Anaconda/Miniconda i spróbuj ponownie
    pause
    exit /b 1
)

echo Aktywuję środowisko hybrik...
call conda activate hybrik
if %errorlevel% neq 0 (
    echo BŁĄD: Nie można aktywować środowiska 'hybrik'
    echo Uruchom: conda create -n hybrik python=3.8
    pause
    exit /b 1
)

echo.
echo Dostępne pliki wideo:
echo.
for %%f in (*.mp4 *.avi *.mov) do (
    echo   - %%f
)

echo.
if "%1"=="" (
    echo Użycie: %0 nazwa_pliku_wideo.mp4
    echo Przykład: %0 dance.mp4
    echo.
    pause
    exit /b 1
)

set VIDEO_FILE=%1
echo Przetwarzam wideo: %VIDEO_FILE%

if not exist "%VIDEO_FILE%" (
    echo BŁĄD: Plik %VIDEO_FILE% nie istnieje
    echo.
    pause
    exit /b 1
)

echo.
echo Uruchamiam HybrIK...
python process_new_video.py "%VIDEO_FILE%"

echo.
echo ================================
echo       Przetwarzanie zakończone!
echo ================================
echo.
echo Następne kroki:
echo 1. Otwórz Blender 4.4+
echo 2. File → Import → HybrIK (.pk)
echo 3. Wybierz plik results_*/res.pk
echo.
pause
