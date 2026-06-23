param(
    [string[]]$PytestArgs = @("tests")
)

$ErrorActionPreference = "Stop"

$Python = if ($env:RASCAL_PYTHON) {
    $env:RASCAL_PYTHON
} else {
    "$env:USERPROFILE\anaconda3\envs\rascal\python.exe"
}

if (-not (Test-Path $Python)) {
    throw "Could not find RASCAL Python at $Python. Set RASCAL_PYTHON to override."
}

& $Python -m pytest @PytestArgs
exit $LASTEXITCODE
