# Run-Final-Checks.ps1
# Runs all final quality checks before thesis submission

param(
    [string]$ThesisFile = "thesis_fixed.md",
    [string]$DocFile = "build\thesis.docx"
)

Write-Host "======================================"
Write-Host "Thesis Final Quality Checks"
Write-Host "======================================"
Write-Host ""

$errors = @()
$warnings = @()
$checks = 0

# Read thesis content
Write-Host "[1/6] Reading thesis content..."
$content = Get-Content $ThesisFile -Raw -Encoding UTF8
$checks++

# Check 1: Placeholder text
Write-Host "[2/6] Checking for placeholder text..."
$placeholders = @('TODO', 'FIXME', '\[VERIFY\]', 'XXX', 'TBD')
foreach ($placeholder in $placeholders) {
    if ($placeholder -eq '\[VERIFY\]') {
        # Literal bracket search
        $matches = ([regex]::Matches($content, '\[VERIFY\]', [System.Text.RegularExpressions.RegexOptions]::None)).Count
    } else {
        $matches = ([regex]::Matches($content, $placeholder, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)).Count
    }
    if ($matches -gt 0) {
        $errors += "Found $matches instances of '$placeholder'"
    }
}
$checks++

# Check 2: Broken markdown links
Write-Host "[3/6] Checking for broken markdown syntax..."
if ($content -match '\]\([^\)]*main_files/') {
    $warnings += "Found references to main_files/ directory (may be broken links)"
}
if ($content -match '!\[\]\(') {
    $warnings += "Found empty image alt text ![]()"
}
$checks++

# Check 3: Figure numbering
Write-Host "[4/6] Verifying figure numbering..."
$figures = [regex]::Matches($content, '\*\*Figure (\d+)')
$figureNumbers = $figures | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object
$expectedFigures = 1..20
$missingFigures = $expectedFigures | Where-Object { $_ -notin $figureNumbers }
if ($missingFigures.Count -gt 0) {
    $warnings += "Figure numbering gaps: $($missingFigures -join ', ')"
}
$checks++

# Check 4: Table caption quality
Write-Host "[5/6] Checking table captions..."
$genericCaptions = ([regex]::Matches($content, '\*\*Table \d+\.\d+ - Summary\*\*')).Count
if ($genericCaptions -gt 0) {
    $warnings += "Found $genericCaptions generic 'Summary' table captions"
}
$checks++

# Check 5: Document file existence
Write-Host "[6/6] Verifying output file..."
if (-not (Test-Path $DocFile)) {
    $errors += "DOCX file not found: $DocFile"
} else {
    $fileSize = (Get-Item $DocFile).Length / 1MB
    $sizeStr = "{0:N2}" -f $fileSize
    Write-Host "  OK DOCX file exists: $sizeStr MB"
}
$checks++

# Report results
Write-Host ""
Write-Host "======================================"
Write-Host "Results: $checks checks completed"
Write-Host "======================================"
Write-Host ""

if ($errors.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "OK All checks passed! Thesis is ready for submission." -ForegroundColor Green
    exit 0
} else {
    if ($errors.Count -gt 0) {
        Write-Host "ERROR: $($errors.Count) issues found:" -ForegroundColor Red
        foreach ($error in $errors) {
            Write-Host "   - $error" -ForegroundColor Red
        }
        Write-Host ""
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "WARNING: $($warnings.Count) issues found:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "   - $warning" -ForegroundColor Yellow
        }
        Write-Host ""
    }
    
    if ($errors.Count -gt 0) {
        Write-Host "Fix errors before submission!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Review warnings, but thesis is acceptable" -ForegroundColor Yellow
        exit 0
    }
}
