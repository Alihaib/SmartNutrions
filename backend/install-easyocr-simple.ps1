# Simple EasyOCR installation - try this first
Write-Host "Installing EasyOCR (simple method)..." -ForegroundColor Green

& ".\venv\Scripts\Activate.ps1"

# Upgrade pip first
pip install --upgrade pip

# Try installing with pre-built wheels
pip install --only-binary :all: easyocr

# If that fails, try without binary requirement
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nTrying alternative installation method..." -ForegroundColor Yellow
    pip install easyocr --no-build-isolation
}

Write-Host "`n✅ Installation attempt complete!" -ForegroundColor Green







