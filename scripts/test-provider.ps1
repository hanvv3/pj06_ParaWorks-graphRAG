param([switch]$Live, [string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'local-runtime-common.ps1')
$arguments = @('provider')
if ($Live) { $arguments += '--live' }
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments $arguments
