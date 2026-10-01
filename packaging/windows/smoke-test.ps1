param([Parameter(Mandatory = $true)][string]$Executable)
$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path -LiteralPath $Executable).Path
& $exe --version
if ($LASTEXITCODE -ne 0) { throw 'Executable version check failed' }
# Let the OS select an unused port; the GUI prints its bound address.
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('smokstak-smoke-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $scratch | Out-Null
$stdout = Join-Path $scratch 'stdout.log'
$stderr = Join-Path $scratch 'stderr.log'
$process = Start-Process -FilePath $exe -ArgumentList @('gui', '--no-open', '--port', '0') -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($process.HasExited) { throw "GUI exited early: $(Get-Content $stderr -Raw)" }
        $log = Get-Content -LiteralPath $stdout -Raw -ErrorAction SilentlyContinue
        if ($log -match 'http://127\.0\.0\.1:\d+') {
            $url = $Matches[0]
            $page = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
            if ($page.StatusCode -ne 200 -or $page.Content -notmatch '(?i)smokstak' -or $page.Content -match '__STAGES__') { throw 'Embedded UI response is invalid' }
            $status = Invoke-RestMethod -Uri "$url/status" -TimeoutSec 5
            if ($null -eq $status) { throw 'Status endpoint returned no JSON' }
            Write-Host "Portable web UI smoke test passed: $url"
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'GUI did not become ready within 30 seconds' }
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id; $process.WaitForExit() }
    # Only remove the two logs created here, without recursive deletion.
    Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $scratch -ErrorAction SilentlyContinue
}
