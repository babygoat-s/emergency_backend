$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ollamaDirectory = Join-Path $projectRoot '.cache\ollama'
$ollamaBinary = Join-Path $ollamaDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaBinary -PathType Leaf)) {
    throw 'Extract the official Ollama Windows ZIP to .cache\ollama first.'
}

$ollamaProfile = Join-Path $ollamaDirectory 'profile'
New-Item -ItemType Directory -Force -Path $ollamaProfile | Out-Null

$server = [System.Diagnostics.ProcessStartInfo]::new($ollamaBinary, 'serve')
$server.WorkingDirectory = $ollamaDirectory
$server.UseShellExecute = $false
$server.EnvironmentVariables['USERPROFILE'] = $ollamaProfile
$server.EnvironmentVariables['OLLAMA_HOST'] = '127.0.0.1:11434'
$server.EnvironmentVariables['OLLAMA_MODELS'] = Join-Path $ollamaDirectory 'models'
$server.EnvironmentVariables['OLLAMA_NO_CLOUD'] = '1'
$server.EnvironmentVariables['OLLAMA_DEBUG_LOG_REQUESTS'] = '0'

$process = [System.Diagnostics.Process]::Start($server)
try {
    $process.WaitForExit()
    exit $process.ExitCode
}
finally {
    if (-not $process.HasExited) {
        $process.Kill()
    }
    $process.Dispose()
}
