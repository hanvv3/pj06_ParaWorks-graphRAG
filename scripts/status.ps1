param([string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'local-runtime-common.ps1')
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments @('status')
