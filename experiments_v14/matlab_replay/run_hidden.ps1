param(
    [Parameter(Mandatory=$true)][string]$Fixture,
    [string]$Matlab = 'D:\software\bin\matlab.exe',
    [string]$Stem = 'E2_run00_provisional_short',
    [ValidateSet('replay_short','replay_full_60f6','replay_resort50k')][string]$EntryPoint = 'replay_short'
)
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fixturePath = (Resolve-Path -LiteralPath $Fixture).Path
$matlabPath = (Resolve-Path -LiteralPath $Matlab).Path
$resultPath = Join-Path $scriptDir ($Stem + '_result.mat')
$stdoutPath = Join-Path $scriptDir ($Stem + '.stdout.txt')
$stderrPath = Join-Path $scriptDir ($Stem + '.stderr.txt')
$logPath = Join-Path $scriptDir ($Stem + '.log')
$env:N14_REPLAY_SCRIPT_DIR = $scriptDir
$env:N14_REPLAY_FIXTURE = $fixturePath
$env:N14_REPLAY_RESULT = $resultPath
$env:N14_REPLAY_LOG = $logPath
$batch = "addpath(getenv('N14_REPLAY_SCRIPT_DIR')); $EntryPoint"
$arguments = '-noFigureWindows -batch "' + $batch + '"'
$process = Start-Process -FilePath $matlabPath -ArgumentList $arguments `
    -WindowStyle Hidden -Wait -PassThru `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
if ($process.ExitCode -ne 0) {
    throw "MATLAB exited with code $($process.ExitCode); see $stdoutPath and $stderrPath"
}
if (-not (Test-Path -LiteralPath $resultPath)) {
    throw "MATLAB exited without result: $resultPath"
}
Write-Output "result=$resultPath"
Write-Output "stdout=$stdoutPath"
Write-Output "stderr=$stderrPath"
Write-Output "log=$logPath"
