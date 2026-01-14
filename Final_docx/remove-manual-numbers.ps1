# Remove all manual section numbers from thesis.md headings

$lines = Get-Content ".\thesis.md"
$output = @()

foreach ($line in $lines) {
    # Remove numbers from Heading 2 (## 1.1 Title -> ## Title)
    if ($line -match '^## \d+\.\d+ (.+)$') {
        $output += "## $($Matches[1])"
    }
    # Remove numbers from Heading 3 (### 1.1.1 Title -> ### Title)
    elseif ($line -match '^### \d+\.\d+\.\d+ (.+)$') {
        $output += "### $($Matches[1])"
    }
    # Keep everything else as-is
    else {
        $output += $line
    }
}

$output | Set-Content ".\thesis.md" -Encoding UTF8

Write-Host "Done! Removed all section numbers from headings" -ForegroundColor Green
Write-Host "Headings are now clean:" -ForegroundColor Green
Write-Host "  # Chapter 1: Introduction" -ForegroundColor Cyan
Write-Host "  ## Background" -ForegroundColor Cyan
Write-Host "  ### Contributions" -ForegroundColor Cyan
