$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path ".\custom-reference.docx").Path)

Write-Host "=== Checking Heading Styles for List Formatting ===" -ForegroundColor Cyan

for ($i = 1; $i -le 3; $i++) {
    try {
        $style = $doc.Styles["Heading $i"]
        Write-Host "`nHeading $i" -ForegroundColor Yellow
        Write-Host "  Has ListTemplate: $($null -ne $style.ListTemplate)"
        Write-Host "  ListLevelNumber: $($style.ListLevelNumber)"
        Write-Host "  OutlineLevel: $($style.ParagraphFormat.OutlineLevel)"
        
        # Check if linked to list
        if ($null -ne $style.ListTemplate) {
            Write-Host "  WARNING: Heading is linked to a list!" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "  Error: $_" -ForegroundColor Red
    }
}

$doc.Close([ref]$false)
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null

Write-Host "`nDone!" -ForegroundColor Green
