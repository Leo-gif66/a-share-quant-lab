$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv not found; installing it into your Python environment..."
  if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m pip install -U uv
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m pip install -U uv
  } else {
    Write-Host "Python not found. Install Python 3.12 first, then rerun this script."
    exit 1
  }
}

uv sync
uv run quant doctor
uv run quant demo
Write-Host "\nEnvironment is ready. Next command:"
Write-Host "uv run quant data-update --limit 50"
