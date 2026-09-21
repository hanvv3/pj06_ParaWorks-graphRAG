param([switch]$IncludeApi, [string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'internal/local-runtime-common.ps1')
$arguments = @('services', '--host', $HostAddress, '--backend-port', "$BackendPort")
if ($IncludeApi) { $arguments += '--include-api' }
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments $arguments
