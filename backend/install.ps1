# PowerShell script to install requirements
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green








