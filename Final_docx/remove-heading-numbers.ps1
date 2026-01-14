# Remove automatic numbering from Word headings

$docPath = ".\build\thesis.docx"

Write-Host "Opening Word document..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $doc = $word.Documents.Open((Resolve-Path $docPath).Path)
    
    Write-Host "Removing automatic numbering from Heading styles..."
    
    # Remove numbering from Heading 1-9 styles
    for ($i = 1; $i -le 9; $i++) {
        try {
            $headingStyle = $doc.Styles["Heading $i"]
            $headingStyle.ParagraphFormat.NumberingStyleLink = $null
            $headingStyle.LinkToListTemplate($null)
            Write-Host "  Removed numbering from Heading $i"
        } catch {
            # Style might not exist, continue
        }
    }
    
    Write-Host "Saving document..."
    $doc.Save()
    $doc.Close()
    
    Write-Host "Done! Automatic heading numbers removed." -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
