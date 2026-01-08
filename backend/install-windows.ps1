# PowerShell script to install dependencies for Windows
Write-Host "Installing dependencies for Windows..." -ForegroundColor Green

& ".\venv\Scripts\Activate.ps1"

Write-Host "Installing basic dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nInstalling OCR library (EasyOCR - Windows compatible)..." -ForegroundColor Yellow
pip install easyocr

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "Note: EasyOCR will download models on first use (~500MB)" -ForegroundColor Cyan








