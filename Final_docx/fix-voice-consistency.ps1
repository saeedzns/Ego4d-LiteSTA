# Fix-Voice-Consistency.ps1
# Standardizes voice from "This thesis" to "We" for natural research writing

param(
    [string]$FilePath = "thesis_fixed.md"
)

Write-Host "Reading thesis file..."
$content = Get-Content $FilePath -Raw -Encoding UTF8

Write-Host "Applying voice consistency fixes..."

# Count replacements
$replacements = 0

# Replace "This thesis" at sentence start with "We"
$pattern1 = '(?m)^This thesis '
$count1 = ([regex]::Matches($content, $pattern1)).Count
$content = $content -replace $pattern1, 'We '
$replacements += $count1

# Replace "The thesis" with "Our approach"
$pattern2 = 'The thesis '
$count2 = ([regex]::Matches($content, $pattern2)).Count
$content = $content -replace $pattern2, 'Our approach '
$replacements += $count2

# Replace "this thesis" mid-sentence with "our work"
$pattern3 = '(?<![A-Z])this thesis '
$count3 = ([regex]::Matches($content, $pattern3)).Count
$content = $content -replace $pattern3, 'our work '
$replacements += $count3

Write-Host "`nReplacement summary:"
Write-Host "  - 'This thesis' → 'We': $count1 replacements"
Write-Host "  - 'The thesis' → 'Our approach': $count2 replacements"
Write-Host "  - 'this thesis' → 'our work': $count3 replacements"
Write-Host "  Total: $replacements changes"

if ($replacements -gt 0) {
    Write-Host "`nSaving updated file..."
    $content | Set-Content $FilePath -Encoding UTF8 -NoNewline
    Write-Host "✓ Voice consistency improved"
    Write-Host "`nNote: Review changes manually to ensure context is preserved."
} else {
    Write-Host "`n✓ Voice already consistent - no changes needed"
}
