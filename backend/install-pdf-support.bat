@echo off
echo Installing PDF support...
call venv\Scripts\activate.bat
pip install PyPDF2
echo.
echo PDF support installed!
echo You can now upload and process PDF files.
pause




