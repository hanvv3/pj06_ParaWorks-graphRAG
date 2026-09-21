param([string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [string]$DatabasePath = '.tmp/paraworks-smoke.db', [string]$PythonPath = '')
. (Join-Path $PSScriptRoot 'local-runtime-common.ps1')
Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments @('start', '--host', $HostAddress,
    '--backend-port', "$BackendPort", '--frontend-port', "$FrontendPort", '--smoke-database', $DatabasePath)
