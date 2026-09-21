param([string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'internal/local-runtime-common.ps1')
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments @('stop')
