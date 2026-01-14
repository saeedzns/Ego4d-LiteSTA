$word = New-Object -ComObject Word.Application
$word.Visible = $false

Write-Host "Opening custom-reference.docx..." -ForegroundColor Cyan
$doc = $word.Documents.Open((Resolve-Path ".\custom-reference.docx").Path)

Write-Host "Removing list templates from Heading 1 and Heading 3..." -ForegroundColor Cyan

# For Heading 1
try {
    $heading1 = $doc.Styles.Item("Heading 1")
    if ($null -ne $heading1.ListTemplate) {
        # Set to null using different method
        $heading1.ListLevelNumber = 0
        Write-Host "  Heading 1: Reset ListLevelNumber to 0" -ForegroundColor Green
    }
} catch {
    Write-Host "  Heading 1 error: $_" -ForegroundColor Red
}

# For Heading 3
try {
    $heading3 = $doc.Styles.Item("Heading 3")
    if ($null -ne $heading3.ListTemplate) {
        $heading3.ListLevelNumber = 0
        Write-Host "  Heading 3: Reset ListLevelNumber to 0" -ForegroundColor Green
    }
} catch {
    Write-Host "  Heading 3 error: $_" -ForegroundColor Red
}

# Alternative approach: Add a paragraph with each heading style and remove numbering
Write-Host "`nApplying styles to paragraphs and removing numbering..." -ForegroundColor Cyan

$range = $doc.Content
$range.Collapse(0) # Collapse to beginning

foreach ($headingNum in @(1, 3)) {
    try {
        $para = $doc.Content.Paragraphs.Add()
        $para.Range.Style = "Heading $headingNum"
        $para.Range.ListFormat.RemoveNumbers()
        $para.Range.Text = ""
        Write-Host "  Applied Heading $headingNum and removed list formatting" -ForegroundColor Green
    } catch {
        $errMsg = $_.Exception.Message
        Write-Host "  Heading $headingNum - $errMsg" -ForegroundColor Yellow
    }
}

# Remove all paragraphs we added
while ($doc.Paragraphs.Count -gt 0) {
    $doc.Paragraphs.Item(1).Range.Delete() | Out-Null
}

Write-Host "`nSaving..." -ForegroundColor Cyan
$doc.Save()
$doc.Close()
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null

Write-Host "`nDone! Template updated." -ForegroundColor Green
