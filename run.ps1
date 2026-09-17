<#
.SYNOPSIS
    Cross-platform automation script for the AI Customer Support Agent on Windows / PowerShell.
.DESCRIPTION
    Provides commands to set up the environment, run the agent, execute evaluation, and run tests.
.EXAMPLE
    .\run.ps1 setup
    .\run.ps1 run "How do I return an item?"
    .\run.ps1 eval
    .\run.ps1 test
#>

param (
    [Parameter(Position=0)]
    [ValidateSet("setup", "download-data", "run", "eval", "test", "lint", "help")]
    [string]$Command = "help",

    [Parameter(Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[!] Virtual environment not found. Running setup first..." -ForegroundColor Yellow
        Setup-Environment
    }
}

function Setup-Environment {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv (Join-Path $PSScriptRoot ".venv")
    
    Write-Host "[*] Upgrading pip..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip

    Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    & $VenvPip install -r (Join-Path $PSScriptRoot "requirements.txt")

    $EnvFile = Join-Path $PSScriptRoot ".env"
    $EnvExample = Join-Path $PSScriptRoot ".env.example"
    if (-not (Test-Path $EnvFile)) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "[+] Created .env from .env.example" -ForegroundColor Green
    }
    Write-Host "[OK] Setup completed successfully!" -ForegroundColor Green
}

switch ($Command) {
    "setup" {
        Setup-Environment
    }
    "download-data" {
        Ensure-Venv
        & $VenvPython -m data.loader --download @RemainingArgs
    }
    "run" {
        Ensure-Venv
        if ($RemainingArgs.Count -eq 0) {
            & $VenvPython -m agent.run --interactive
        } else {
            & $VenvPython -m agent.run --query ($RemainingArgs -join " ")
        }
    }
    "eval" {
        Ensure-Venv
        & $VenvPython -m eval.run --golden-set (Join-Path $PSScriptRoot "eval\golden_set\golden_examples.jsonl") @RemainingArgs
    }
    "test" {
        Ensure-Venv
        & $VenvPython -m pytest @RemainingArgs
    }
    "lint" {
        Ensure-Venv
        & $VenvPython -m ruff check . @RemainingArgs
    }
    "help" {
        Write-Host 'Usage: .\run.ps1 [command] [args...]' -ForegroundColor Cyan
        Write-Host 'Commands:'
        Write-Host '  setup         - Create .venv and install all dependencies'
        Write-Host '  download-data - Download / ingest dataset'
        Write-Host '  run [query]   - Run agent pipeline (interactive if no query provided)'
        Write-Host '  eval          - Run evaluation harness on golden set'
        Write-Host '  test          - Run pytest test suite'
        Write-Host '  lint          - Run ruff linter check'
    }
}
