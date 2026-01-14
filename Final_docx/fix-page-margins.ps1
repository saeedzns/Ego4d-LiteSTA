# Fix Page Margins and Layout
# Sets professional thesis margins in custom-reference.docx

$docPath = Join-Path $PSScriptRoot "custom-reference.docx"

Write-Host "Fixing page margins and layout..." -ForegroundColor Cyan

$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $doc = $word.Documents.Open($docPath)
    
    # Set page setup for all sections
    foreach ($section in $doc.Sections) {
        $pageSetup = $section.PageSetup
        
        # Set margins (in points: 1 inch = 72 points)
        $pageSetup.TopMargin = 72        # 1 inch
        $pageSetup.BottomMargin = 72     # 1 inch  
        $pageSetup.LeftMargin = 108      # 1.5 inches (for binding)
        $pageSetup.RightMargin = 72      # 1 inch
        
        # Set page size to A4
        $pageSetup.PageWidth = 595       # A4 width in points
        $pageSetup.PageHeight = 842      # A4 height in points
        
        # Orientation: Portrait
        $pageSetup.Orientation = 0       # wdOrientPortrait
        
        Write-Host "  Page margins set: Top/Bottom/Right: 1 inch, Left: 1.5 inches" -ForegroundColor Green
        Write-Host "  Page size: A4 (210mm x 297mm)" -ForegroundColor Green
    }
    
    $doc.Save()
    $doc.Close()
    
    Write-Host "`nPage setup fixed successfully!" -ForegroundColor Green
    Write-Host "Rebuild thesis to apply changes" -ForegroundColor Yellow
    
} catch {
    Write-Host "`nError: $_" -ForegroundColor Red
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
