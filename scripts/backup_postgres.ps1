param(
    [string]$HostName = "localhost",
    [int]$Port = 5432,
    [string]$Database = "road_detection",
    [string]$User = "road_user",
    [string]$OutputDir = "backups"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$output = Join-Path $OutputDir "road_detection_$stamp.dump"
pg_dump -h $HostName -p $Port -U $User -d $Database -F c -f $output
Write-Host "Backup created: $output"
