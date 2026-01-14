$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path ".\custom-reference.docx").Path)

Write-Host "=== Removing ALL List Formatting from Headings ===" -ForegroundColor Cyan

for ($i = 1; $i -le 9; $i++) {
    try {
        $style = $doc.Styles["Heading $i"]
        
        # Remove list template link
        $style.LinkToListTemplate($null)
        
        # Explicitly set numbering to none
        $style.ParagraphFormat.LeftIndent = 0
        $style.ParagraphFormat.FirstLineIndent = 0
        
        Write-Host "  Heading $i - List formatting removed" -ForegroundColor Green
    }
    catch {
        Write-Host "  Heading $i - Error $_" -ForegroundColor Yellow
    }
}

Write-Host "`nSaving template..." -ForegroundColor Cyan
$doc.Save()
$doc.Close()
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null

Write-Host "`nDone! All list formatting removed from heading styles." -ForegroundColor Green
Write-Host "Now rebuild your thesis with the updated template." -ForegroundColor Yellow
