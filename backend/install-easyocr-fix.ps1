# Fix for EasyOCR installation issues on Windows
Write-Host "Installing EasyOCR with compatible dependencies..." -ForegroundColor Green

& ".\venv\Scripts\Activate.ps1"

# Install dependencies separately to avoid conflicts
Write-Host "`nStep 1: Installing build tools..." -ForegroundColor Yellow
pip install --upgrade pip setuptools wheel

Write-Host "`nStep 2: Installing numpy and scipy first..." -ForegroundColor Yellow
pip install numpy scipy

Write-Host "`nStep 3: Installing scikit-image (this may take a while)..." -ForegroundColor Yellow
pip install scikit-image

Write-Host "`nStep 4: Installing other EasyOCR dependencies..." -ForegroundColor Yellow
pip install opencv-python-headless
pip install python-bidi
pip install pyclipper
pip install shapely
pip install lmdb
pip install tqdm

Write-Host "`nStep 5: Installing EasyOCR..." -ForegroundColor Yellow
pip install easyocr --no-deps
pip install easyocr  # Try again with dependencies

Write-Host "`n✅ Installation complete!" -ForegroundColor Green
Write-Host "Note: EasyOCR will download models on first use (~500MB)" -ForegroundColor Cyan







