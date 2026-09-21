param([string]$RedisUrl = '', [string]$DatabaseUrl = '', [string]$PythonPath = '')
. (Join-Path $PSScriptRoot '../internal/local-runtime-common.ps1')
$oldRedis = $env:REDIS_URL
$oldDatabase = $env:PARAWORKS_DATABASE_URL
try {
    if ($RedisUrl) { $env:REDIS_URL = $RedisUrl }
    if ($DatabaseUrl) { $env:PARAWORKS_DATABASE_URL = $DatabaseUrl }
    Invoke-LocalRuntime -PythonPath $PythonPath -RuntimeArguments @('worker')
}
finally { $env:REDIS_URL = $oldRedis; $env:PARAWORKS_DATABASE_URL = $oldDatabase }
