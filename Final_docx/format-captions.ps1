# This script formats all figure and table captions to be Word-compatible
# Figures: Simple numbering (Figure 1, Figure 2, etc.)
# Tables: Chapter.number format (Table 2.1, Table 8.1, etc.)

$content = Get-Content "thesis.md" -Raw

Write-Host "Formatting captions for Word compatibility..." -ForegroundColor Cyan

# Track figure and table numbers
$figNum = 1
$tableNums = @{}  # Track tables per chapter

# Process figures - simple sequential numbering
$figurePattern = '!\[(.*?)\]\((.*?)\)'
$content = [regex]::Replace($content, $figurePattern, {
    param($match)
    $caption = $match.Groups[1].Value
    $path = $match.Groups[2].Value
    
    # Create Word-compatible caption format
    $newCaption = "Figure $script:figNum - $caption"
    $script:figNum++
    
    "![$newCaption]($path)"
})

Write-Host "Processed $($figNum - 1) figures" -ForegroundColor Green

# Process tables - need to detect chapter context
# Tables will be numbered based on their chapter location
$lines = $content -split "`n"
$currentChapter = 0
$inTable = $false
$result = @()

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    
    # Detect chapter headings
    if ($line -match '^# Chapter (\d+)') {
        $currentChapter = [int]$matches[1]
        if (-not $tableNums.ContainsKey($currentChapter)) {
            $tableNums[$currentChapter] = 1
        }
    }
    
    # Detect table start (markdown table with headers)
    if ($line -match '^\|.*\|$' -and $i + 1 -lt $lines.Count -and $lines[$i + 1] -match '^\|[-:]+\|') {
        # This is a table header, add caption before it
        if ($currentChapter -eq 0) {
            $currentChapter = 2  # Default to chapter 2 if not in a chapter yet
            if (-not $tableNums.ContainsKey($currentChapter)) {
                $tableNums[$currentChapter] = 1
            }
        }
        
        $tableNum = $tableNums[$currentChapter]
        $caption = ": Table $currentChapter.$tableNum"
        
        # Add caption line before table
        $result += ""
        $result += $caption
        $result += ""
        
        $tableNums[$currentChapter]++
    }
    
    $result += $line
}

$content = $result -join "`n"

# Save the updated content
$content | Set-Content "thesis.md" -Encoding UTF8

Write-Host "`nDone! Updated thesis.md with Word-compatible captions" -ForegroundColor Green
Write-Host "Now rebuild the DOCX to apply changes" -ForegroundColor Yellow
