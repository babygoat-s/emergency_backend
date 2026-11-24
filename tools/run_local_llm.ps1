param(
    [switch]$Background,
    [string]$Model
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ollamaDirectory = Join-Path $projectRoot '.cache\ollama'
$ollamaBinary = Join-Path $ollamaDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaBinary -PathType Leaf)) {
    throw 'Extract the official Ollama Windows ZIP to .cache\ollama first.'
}

function Get-OllamaTags {
    try {
        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2
        if ($tags.PSObject.Properties['models']) { return $tags }
    }
    catch {
        return $null
    }
}

if ($Background) {
    $tags = Get-OllamaTags
    if ($null -eq $tags) {
        $stdout = Join-Path $ollamaDirectory 'launcher.stdout.log'
        $stderr = Join-Path $ollamaDirectory 'launcher.stderr.log'
        $launcher = Start-Process -FilePath (Get-Process -Id $PID).Path `
            -ArgumentList @('-NoProfile', '-File', ('"{0}"' -f $PSCommandPath)) `
            -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        do {
            $tags = Get-OllamaTags
            if ($null -ne $tags) { break }
            Start-Sleep -Milliseconds 250
        } while ([DateTime]::UtcNow -lt $deadline)
        $launcher.Dispose()
        if ($null -eq $tags) {
            throw "Ollama did not open 127.0.0.1:11434. Check $stderr."
        }
    }
    if ($Model -and $Model -notin $tags.models.name) {
        throw "Ollama is running, but the configured model '$Model' is not installed."
    }
    Write-Host 'Local Ollama is ready at 127.0.0.1:11434.'
    return
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
$server.EnvironmentVariables['OLLAMA_KEEP_ALIVE'] = '10m'

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
