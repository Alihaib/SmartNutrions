@echo off
echo Installing dependencies for Windows...
call venv\Scripts\activate.bat

echo Installing basic dependencies...
pip install -r requirements.txt

echo.
echo Installing OCR library (EasyOCR - Windows compatible)...
pip install easyocr

echo.
echo Installation complete!
echo.
echo Note: EasyOCR will download models on first use (~500MB)
pause








