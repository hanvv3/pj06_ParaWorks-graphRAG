param([switch]$Stop, [switch]$Down, [switch]$SkipApp, [switch]$NoDockerDesktopLaunch,
    [string]$HostAddress = '127.0.0.1', [int]$BackendPort = 8000, [int]$FrontendPort = 3000,
    [int]$PostgresPort = 5432, [int]$RedisPort = 6379, [string]$DatabaseUrl = '',
    [switch]$InitializeDatabase, [string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
if ($Down) { throw 'Automatic Docker down is retired. Use stop.ps1 for owned app processes; manage selected Docker services explicitly. Data volumes are preserved.' }
if ($Stop) { & (Join-Path $PSScriptRoot 'stop.ps1') -PythonPath $PythonPath; return }
$parameters = @{ HostAddress=$HostAddress; BackendPort=$BackendPort; FrontendPort=$FrontendPort; DatabaseUrl=$DatabaseUrl; SkipApp=$SkipApp; InitializeDatabase=$InitializeDatabase; PythonPath=$PythonPath }
if ($PSBoundParameters.ContainsKey('PostgresPort')) { $parameters.PostgresPort = $PostgresPort }
if ($PSBoundParameters.ContainsKey('RedisPort')) { $parameters.RedisPort = $RedisPort }
Write-Host 'Compatibility launcher: configured services must already be running; no automatic Docker or schema changes.'
& (Join-Path $PSScriptRoot 'start-pgvector-dev.ps1') @parameters
