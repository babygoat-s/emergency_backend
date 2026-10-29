param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project .venv not found. Create it and install dependencies first; see README.md.'
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $serverArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
    if ($Reload) { $serverArgs += '--reload' }
    & $projectPython @serverArgs
    $serverExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $serverExitCode
