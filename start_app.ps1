$ErrorActionPreference = "Continue"
cd c:\App\Khmer_TTS
$env:PYTHONIOENCODING = "utf-8"
$env:TOKENIZERS_PARALLELISM = "false"

# Ensure app module is importable
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

Write-Host "Starting KHMER TTS STUDIO..."
Write-Host "Working directory: $PWD"
Write-Host "PYTHONPATH: $env:PYTHONPATH"

python -c "
import sys
sys.path.insert(0, '.')
from app.main import main
import sys as _sys
try:
    _sys.exit(main())
except Exception as e:
    print(f'Error starting app: {e}')
    import traceback
    traceback.print_exc()
    _sys.exit(1)
"
