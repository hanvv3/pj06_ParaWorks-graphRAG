param([string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [string]$DatabasePath = '.tmp/paraworks-visual-smoke.db', [string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$started = $false
$oldWeb = $env:PLAYWRIGHT_BASE_URL
$oldApi = $env:PLAYWRIGHT_API_BASE_URL
$oldSkipSeed = $env:PLAYWRIGHT_SKIP_BACKEND_SEED
try {
    # Common startup refuses occupied ports and manages cleanup on partial failure.
    & (Join-Path $PSScriptRoot 'start-smoke.ps1') -HostAddress $HostAddress -BackendPort $BackendPort -FrontendPort $FrontendPort -DatabasePath $DatabasePath -PythonPath $PythonPath
    $started = $true
    $env:PLAYWRIGHT_BASE_URL = "http://${HostAddress}:$FrontendPort"
    $env:PLAYWRIGHT_API_BASE_URL = "http://${HostAddress}:$BackendPort"
    $env:PLAYWRIGHT_SKIP_BACKEND_SEED = '1'
    Push-Location (Join-Path $workspace 'frontend')
    try {
        & npm.cmd run test:visual
        if ($LASTEXITCODE -ne 0) { throw "Visual tests failed (exit $LASTEXITCODE)." }
    }
    finally { Pop-Location }
}
finally {
    $env:PLAYWRIGHT_BASE_URL = $oldWeb
    $env:PLAYWRIGHT_API_BASE_URL = $oldApi
    $env:PLAYWRIGHT_SKIP_BACKEND_SEED = $oldSkipSeed
    if ($started) { & (Join-Path $PSScriptRoot '../stop.ps1') -PythonPath $PythonPath }
}
