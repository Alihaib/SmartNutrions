# PowerShell script to install essential dependencies
Write-Host "Installing essential dependencies..." -ForegroundColor Green

& ".\venv\Scripts\Activate.ps1"

Write-Host "`nStep 1: Installing FastAPI and core dependencies..." -ForegroundColor Yellow
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install python-multipart==0.0.6
pip install pydantic==2.5.0

Write-Host "`nStep 2: Installing image processing..." -ForegroundColor Yellow
pip install pillow==10.1.0
pip install numpy==1.24.3

Write-Host "`nStep 3: Installing AI models (this may take a while)..." -ForegroundColor Yellow
pip install transformers==4.35.0
pip install torch==2.1.0
pip install accelerate==0.24.1
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

Write-Host "`nStep 4: Installing OCR (EasyOCR for Windows)..." -ForegroundColor Yellow
pip install easyocr

Write-Host "`nEssential installation complete!" -ForegroundColor Green
Write-Host "You can now start the server with: uvicorn main:app --reload --port 8000" -ForegroundColor Cyan








