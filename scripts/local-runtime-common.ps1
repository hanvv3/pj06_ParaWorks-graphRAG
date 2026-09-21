$ErrorActionPreference = 'Stop'

function Invoke-LocalRuntime {
    param([string]$PythonPath, [string[]]$RuntimeArguments)
    $workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    if (-not $PythonPath) {
        $environmentPath = if ($env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT } else { '.venv' }
        if (-not [IO.Path]::IsPathRooted($environmentPath)) { $environmentPath = Join-Path $workspace $environmentPath }
        $PythonPath = Join-Path $environmentPath 'Scripts/python.exe'
    }
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw 'Python environment missing. Set UV_PROJECT_ENVIRONMENT or pass -PythonPath; install dependencies explicitly.'
    }
    & $PythonPath (Join-Path $PSScriptRoot 'local_runtime.py') @RuntimeArguments --workspace $workspace
    if ($LASTEXITCODE -ne 0) { throw "ParaWorks command failed (exit $LASTEXITCODE)." }
}
