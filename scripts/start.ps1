param([string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [switch]$InitializeDatabase, [switch]$SkipFrontend, [switch]$SkipApp, [string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'internal/local-runtime-common.ps1')
$arguments = @('start', '--host', $HostAddress, '--backend-port', "$BackendPort", '--frontend-port', "$FrontendPort")
if ($InitializeDatabase) { $arguments += '--initialize-database' }
if ($SkipFrontend) { $arguments += '--skip-frontend' }
if ($SkipApp) { $arguments += '--skip-app' }
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments $arguments
