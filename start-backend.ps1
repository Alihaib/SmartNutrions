# PowerShell script to start the backend server
Write-Host "Starting Backend Server..." -ForegroundColor Green
Set-Location backend
& ".\venv\Scripts\Activate.ps1"
uvicorn main:app --reload --port 8000








