@echo off
echo ============================================
echo  Building AirHockeyTracker.exe
echo ============================================
echo.

:: Install PyInstaller if not present
py -m pip install pyinstaller --quiet

:: Check that required data files exist
if not exist "test_video.mp4" (
    echo ERROR: test_video.mp4 not found.
    pause & exit /b 1
)

:: calibration.npy is optional - include it only if it exists
set EXTRA=
if exist "calibration.npy" set EXTRA=--add-data "calibration.npy;."

:: Build the executable
py -m PyInstaller --onefile ^
    --name "AirHockeyTracker" ^
    --add-data "test_video.mp4;." ^
    --add-data "utils.py;." ^
    %EXTRA% ^
    --hidden-import "matplotlib.backends.backend_tkagg" ^
    --hidden-import "matplotlib.backends.backend_agg" ^
    --collect-submodules cv2 ^
    main.py

echo.
if exist "dist\AirHockeyTracker.exe" (
    echo SUCCESS: dist\AirHockeyTracker.exe is ready.
) else (
    echo BUILD FAILED. Check the output above for errors.
)
echo.
pause
