# Remove automatic numbering from Heading 1 only (chapters)
# Keep numbering for subsection headings (Heading 2, 3, etc.)

$docPath = ".\build\thesis.docx"

Write-Host "Opening Word document..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $doc = $word.Documents.Open((Resolve-Path $docPath).Path)
    
    Write-Host "Removing automatic numbering from Heading 1 (chapter level only)..."
    
    # Only remove numbering from Heading 1 (chapters)
    try {
        $heading1Style = $doc.Styles["Heading 1"]
        $heading1Style.ParagraphFormat.NumberingStyleLink = $null
        $heading1Style.LinkToListTemplate($null)
        Write-Host "  Removed automatic numbering from Heading 1" -ForegroundColor Green
    } catch {
        Write-Host "  Error accessing Heading 1: $_" -ForegroundColor Red
    }
    
    # Keep Heading 2, 3, 4, etc. as they are (for subsections)
    Write-Host "  Subsection headings (Heading 2+) kept as-is" -ForegroundColor Green
    
    Write-Host "Saving document..."
    $doc.Save()
    $doc.Close()
    
    Write-Host "`nDone! Chapter headings will now show correctly:" -ForegroundColor Green
    Write-Host "  Before: '1 Chapter 1: Introduction'" -ForegroundColor Yellow
    Write-Host "  After:  'Chapter 1: Introduction'" -ForegroundColor Green
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
