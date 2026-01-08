# Quick install script - run this in your backend directory
Write-Host "Installing essential packages..." -ForegroundColor Green

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "`nInstalling FastAPI and core dependencies..." -ForegroundColor Yellow
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6
pip install pydantic==2.5.0

Write-Host "`nInstalling image processing..." -ForegroundColor Yellow
pip install pillow==10.1.0
pip install numpy==1.24.3

Write-Host "`n✅ Core packages installed! You can now start the server." -ForegroundColor Green
Write-Host "`nTo start: uvicorn main:app --reload --port 8000" -ForegroundColor Cyan
Write-Host "`nNote: AI models (transformers, torch) can be installed later if needed." -ForegroundColor Yellow








