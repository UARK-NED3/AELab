param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [double]$Threshold,

    [string]$OutputPath,

    [switch]$InPlace
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input file not found: $InputPath"
}

if ($InPlace -and $OutputPath) {
    throw "Use either -InPlace or -OutputPath, not both."
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $OutputPath) {
    if ($InPlace) {
        $resolvedOutput = $resolvedInput
    }
    else {
        $directory = Split-Path -Path $resolvedInput -Parent
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)
        $extension = [System.IO.Path]::GetExtension($resolvedInput)
        $resolvedOutput = Join-Path -Path $directory -ChildPath ("{0}_trimmed{1}" -f $baseName, $extension)
    }
}
else {
    $resolvedOutput = $OutputPath
}

$lines = Get-Content -LiteralPath $resolvedInput
if ($lines.Count -eq 0) {
    throw "Input file is empty: $resolvedInput"
}

$dataHeaderIndex = -1
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i].TrimStart() -like 'X_Value,*') {
        $dataHeaderIndex = $i
        break
    }
}

if ($dataHeaderIndex -lt 0) {
    throw "Could not find the data header line starting with 'X_Value,'."
}

$outputLines = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -le $dataHeaderIndex; $i++) {
    $outputLines.Add($lines[$i])
}

$dataLines = @()
for ($i = $dataHeaderIndex + 1; $i -lt $lines.Count; $i++) {
    $dataLines += $lines[$i]
}

$lastLineToKeep = $dataLines.Count - 1
while ($lastLineToKeep -ge 0) {
    $candidate = $dataLines[$lastLineToKeep].Trim()

    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $lastLineToKeep--
        continue
    }

    $firstField = ($candidate -split ',', 2)[0].Trim()
    $parsedValue = 0.0
    if (-not [double]::TryParse(
        $firstField,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsedValue
    )) {
        break
    }

    if ($parsedValue -gt $Threshold) {
        $lastLineToKeep--
        continue
    }

    break
}

for ($i = 0; $i -le $lastLineToKeep; $i++) {
    $outputLines.Add($dataLines[$i])
}

[System.IO.File]::WriteAllLines($resolvedOutput, $outputLines)

$removedCount = $dataLines.Count - ($lastLineToKeep + 1)
Write-Host "Input: $resolvedInput"
Write-Host "Output: $resolvedOutput"
Write-Host "Threshold: $Threshold"
Write-Host "Removed trailing rows: $removedCount"
