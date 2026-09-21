param([string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [int]$PostgresPort = 5432, [int]$RedisPort = 6379, [string]$DatabaseUrl = '',
    [switch]$NoAutoPortFallback, [switch]$SkipApp, [switch]$InitializeDatabase, [string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('PostgresPort') -or $PSBoundParameters.ContainsKey('RedisPort')) {
    throw 'Set complete service URLs in root .env or inherited environment; automatic port fallback and Docker provisioning were retired.'
}
$previous = $env:PARAWORKS_DATABASE_URL
try {
    if ($DatabaseUrl) { $env:PARAWORKS_DATABASE_URL = $DatabaseUrl }
    & (Join-Path $PSScriptRoot 'start.ps1') -HostAddress $HostAddress -BackendPort $BackendPort -FrontendPort $FrontendPort -SkipApp:$SkipApp -InitializeDatabase:$InitializeDatabase -PythonPath $PythonPath
}
finally { $env:PARAWORKS_DATABASE_URL = $previous }
