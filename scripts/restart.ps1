param([string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [switch]$InitializeDatabase, [switch]$SkipFrontend, [string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'local-runtime-common.ps1')
$arguments = @('restart', '--host', $HostAddress, '--backend-port', "$BackendPort", '--frontend-port', "$FrontendPort")
if ($InitializeDatabase) { $arguments += '--initialize-database' }
if ($SkipFrontend) { $arguments += '--skip-frontend' }
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments $arguments
