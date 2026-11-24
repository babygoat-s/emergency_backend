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
    if (Test-Path -LiteralPath '.cache\ollama\ollama.exe') {
        $localOllamaModel = & $projectPython -c @'
from app.config import Settings
s = Settings()
if s.rag_enabled and s.rag_llm_provider == "openai_compatible" and s.rag_llm_base_url.rstrip("/") in (
    "http://127.0.0.1:11434/v1", "http://localhost:11434/v1",
):
    print(s.rag_llm_model or s.gemini_model)
'@
        if ($LASTEXITCODE -ne 0) { throw 'Could not read backend settings.' }
        if ($localOllamaModel) {
            try {
                & (Join-Path $PSScriptRoot 'tools\run_local_llm.ps1') -Background -Model $localOllamaModel
            }
            catch {
                Write-Warning $_.Exception.Message
            }
        }
    }
    $serverArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
    if ($Reload) { $serverArgs += '--reload' }
    & $projectPython @serverArgs
    $serverExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $serverExitCode
