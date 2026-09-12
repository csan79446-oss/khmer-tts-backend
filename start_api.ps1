# Start Khmer TTS Backend API
Set-Location c:\App\Khmer_TTS\backend\api

# Create venv if not exists
if (-Not (Test-Path .venv)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies if needed
if (-Not (Get-Command uvicorn -ErrorAction SilentlyContinue)) {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
}

# Start the API server
Write-Host "Starting Khmer TTS Backend API on http://0.0.0.0:8000"
$env:RUNPOD_ENDPOINT_ID = ""
$env:RUNPOD_API_KEY = ""
$env:API_SECRET = ""
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
