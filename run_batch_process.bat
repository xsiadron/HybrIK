@echo off
echo ========================================
echo       HybrIK Batch Video Processor
echo ========================================
echo.

REM Sprawdź czy folder input istnieje
if not exist "input" (
    echo BLAD: Folder 'input' nie istnieje!
    echo Utworz folder 'input' i umiesc w nim pliki wideo.
    echo.
    pause
    exit /b 1
)

REM Sprawdź czy są pliki w folderze input
dir /b "input\*.mp4" "input\*.avi" "input\*.mov" "input\*.mkv" "input\*.wmv" "input\*.flv" "input\*.webm" >nul 2>&1
if errorlevel 1 (
    echo BLAD: Brak plikow wideo w folderze 'input'
    echo Obsługiwane formaty: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm
    echo.
    pause
    exit /b 1
)

echo Uruchamiam skrypt batch_process.py...
echo.

REM Uruchom skrypt Python
python batch_process.py

echo.
echo Nacisnij dowolny klawisz aby zakonczyc...
pause >nul
