# Add section numbers to all headings in thesis.md

$content = Get-Content ".\thesis.md" -Raw

# Track current chapter number
$chapterNum = 0
$section2Num = 0
$section3Num = 0

$lines = Get-Content ".\thesis.md"
$output = @()

foreach ($line in $lines) {
    if ($line -match '^# Chapter (\d+)') {
        # Chapter heading - extract number, don't add prefix
        $chapterNum = [int]$Matches[1]
        $section2Num = 0
        $section3Num = 0
        $output += $line
    }
    elseif ($line -match '^# ') {
        # Other level 1 headings (Abstract, References, etc.) - no numbering
        $chapterNum = 0
        $section2Num = 0
        $section3Num = 0
        $output += $line
    }
    elseif ($line -match '^## (\d+\.\d+) ') {
        # Already numbered Heading 2 - skip
        $output += $line
    }
    elseif ($line -match '^## (.+)$') {
        # Heading 2 - add chapter.section numbering
        if ($chapterNum -gt 0) {
            $section2Num++
            $section3Num = 0
            $title = $Matches[1]
            $output += "## $chapterNum.$section2Num $title"
        } else {
            $output += $line
        }
    }
    elseif ($line -match '^### (\d+\.\d+\.\d+) ') {
        # Already numbered Heading 3 - skip
        $output += $line
    }
    elseif ($line -match '^### (.+)$') {
        # Heading 3 - add chapter.section.subsection numbering
        if ($chapterNum -gt 0 -and $section2Num -gt 0) {
            $section3Num++
            $title = $Matches[1]
            $output += "### $chapterNum.$section2Num.$section3Num $title"
        } else {
            $output += $line
        }
    }
    else {
        $output += $line
    }
}

$output | Set-Content ".\thesis.md" -Encoding UTF8

Write-Host "Done! Added section numbers to all headings" -ForegroundColor Green
Write-Host "Chapter sections numbered as: 1.1, 1.2, 2.1, 2.2, etc." -ForegroundColor Green
Write-Host "Subsections numbered as: 1.1.1, 1.1.2, 2.1.1, etc." -ForegroundColor Green
