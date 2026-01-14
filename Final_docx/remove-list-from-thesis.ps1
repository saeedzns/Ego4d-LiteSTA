$word = New-Object -ComObject Word.Application
$word.Visible = $false

Write-Host "Opening thesis.docx..." -ForegroundColor Cyan
$doc = $word.Documents.Open((Resolve-Path ".\build\thesis.docx").Path)

Write-Host "Removing list formatting from all headings..." -ForegroundColor Cyan

$count = 0
foreach ($para in $doc.Paragraphs) {
    $styleName = $para.Style.NameLocal
    
    # Check if paragraph is a heading
    if ($styleName -like "Heading*" -or $styleName -like "Titolo*") {
        # Remove any list formatting
        $para.Range.ListFormat.RemoveNumbers()
        $count++
    }
}

Write-Host "Processed $count heading paragraphs" -ForegroundColor Green

Write-Host "Saving document..." -ForegroundColor Cyan
$doc.Save()
$doc.Close()
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null

Write-Host "`nDone! All list formatting removed from headings in thesis.docx" -ForegroundColor Green
