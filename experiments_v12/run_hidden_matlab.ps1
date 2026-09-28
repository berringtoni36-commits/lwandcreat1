param(
    [Parameter(Mandatory=$true)][string]$Script,
    [Parameter(Mandatory=$true)][string]$OutputPrefix,
    [string]$Fixture='',
    [string]$Result=''
)
$ErrorActionPreference='Stop'
$matlab='D:\software\bin\matlab.exe'
$fullScript=(Resolve-Path -LiteralPath $Script).Path.Replace('\','/')
$prefix=[System.IO.Path]::GetFullPath($OutputPrefix)
$parent=[System.IO.Path]::GetDirectoryName($prefix)
[System.IO.Directory]::CreateDirectory($parent) | Out-Null
if ($Fixture) { $env:AKT_FIXTURE=[System.IO.Path]::GetFullPath($Fixture) }
if ($Result) { $env:AKT_RESULT=[System.IO.Path]::GetFullPath($Result) }
$statement="run('$fullScript')"
$arguments=@('-wait','-batch',('"'+$statement+'"'),'-noFigureWindows','-logfile',('"'+$prefix+'.log"'))
$p=Start-Process -FilePath $matlab -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput ($prefix+'.stdout.txt') -RedirectStandardError ($prefix+'.stderr.txt')
Write-Output "MATLAB_EXIT_CODE=$($p.ExitCode)"
exit $p.ExitCode
