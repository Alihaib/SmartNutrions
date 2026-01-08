# PowerShell script to install PDF support
Write-Host "Installing PDF support..." -ForegroundColor Green

& ".\venv\Scripts\Activate.ps1"

Write-Host "Installing PyPDF2..." -ForegroundColor Yellow
pip install PyPDF2

Write-Host "`nPDF support installed successfully!" -ForegroundColor Green
Write-Host "You can now upload and process PDF files." -ForegroundColor Cyan




