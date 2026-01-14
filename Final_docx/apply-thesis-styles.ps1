# Professional Thesis Styling Script
# Applies academic color scheme to custom-reference.docx

$docPath = Join-Path $PSScriptRoot "custom-reference.docx"

Write-Host "Opening Word and applying professional styles..." -ForegroundColor Cyan

# Create Word COM object
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    # Open the reference document
    $doc = $word.Documents.Open($docPath)
    
    # Define academic color scheme
    $colorBlack = 0        # RGB(0, 0, 0)
    $colorDarkBlue = 8801311   # RGB(31, 71, 136) - #1F4788
    $colorMediumBlue = 9456686 # RGB(46, 80, 144) - #2E5090
    $colorDarkGray = 4210752   # RGB(64, 64, 64) - #404040
    $colorLightGray = 16119285 # RGB(245, 245, 245) - #F5F5F5
    
    Write-Host "Applying styles..." -ForegroundColor Yellow
    
    # HEADING 1 (Chapters) - Dark Blue, Bold, 18pt
    $heading1 = $doc.Styles.Item("Heading 1")
    $heading1.Font.Color = $colorDarkBlue
    $heading1.Font.Bold = $true
    $heading1.Font.Size = 18
    $heading1.Font.Name = "Times New Roman"
    $heading1.ParagraphFormat.SpaceBefore = 24
    $heading1.ParagraphFormat.SpaceAfter = 12
    $heading1.ParagraphFormat.PageBreakBefore = $true
    Write-Host "  ✓ Heading 1 (Chapters)" -ForegroundColor Green
    
    # HEADING 2 (Sections) - Medium Blue, Bold, 14pt
    $heading2 = $doc.Styles.Item("Heading 2")
    $heading2.Font.Color = $colorMediumBlue
    $heading2.Font.Bold = $true
    $heading2.Font.Size = 14
    $heading2.Font.Name = "Times New Roman"
    $heading2.ParagraphFormat.SpaceBefore = 18
    $heading2.ParagraphFormat.SpaceAfter = 6
    Write-Host "  ✓ Heading 2 (Sections)" -ForegroundColor Green
    
    # HEADING 3 (Subsections) - Dark Gray, Bold, 12pt
    $heading3 = $doc.Styles.Item("Heading 3")
    $heading3.Font.Color = $colorDarkGray
    $heading3.Font.Bold = $true
    $heading3.Font.Size = 12
    $heading3.Font.Name = "Times New Roman"
    $heading3.ParagraphFormat.SpaceBefore = 12
    $heading3.ParagraphFormat.SpaceAfter = 6
    Write-Host "  ✓ Heading 3 (Subsections)" -ForegroundColor Green
    
    # NORMAL (Body text) - Black, 12pt, Justified, 1.5 line spacing
    $normal = $doc.Styles.Item("Normal")
    $normal.Font.Color = $colorBlack
    $normal.Font.Bold = $false
    $normal.Font.Size = 12
    $normal.Font.Name = "Times New Roman"
    $normal.ParagraphFormat.Alignment = 3  # wdAlignParagraphJustify
    $normal.ParagraphFormat.LineSpacingRule = 1  # wdLineSpace1pt5
    $normal.ParagraphFormat.LineSpacing = 18  # 1.5 spacing
    $normal.ParagraphFormat.SpaceAfter = 6
    Write-Host "  ✓ Normal (Body text)" -ForegroundColor Green
    
    # CAPTION (Figure/Table captions) - Dark Gray, 10pt, Centered
    try {
        $caption = $doc.Styles.Item("Caption")
        $caption.Font.Color = $colorDarkGray
        $caption.Font.Size = 10
        $caption.Font.Name = "Times New Roman"
        $caption.ParagraphFormat.Alignment = 1  # wdAlignParagraphCenter
        $caption.ParagraphFormat.SpaceBefore = 6
        $caption.ParagraphFormat.SpaceAfter = 12
        Write-Host "  ✓ Caption" -ForegroundColor Green
    } catch {
        Write-Host "  ! Caption style not found (optional)" -ForegroundColor DarkYellow
    }
    
    # TABLE OF CONTENTS HEADING
    try {
        $tocHeading = $doc.Styles.Item("TOC Heading")
        $tocHeading.Font.Color = $colorDarkBlue
        $tocHeading.Font.Bold = $true
        $tocHeading.Font.Size = 16
        $tocHeading.Font.Name = "Times New Roman"
        $tocHeading.ParagraphFormat.Alignment = 1  # Center
        $tocHeading.ParagraphFormat.SpaceAfter = 24
        Write-Host "  ✓ TOC Heading" -ForegroundColor Green
    } catch {
        Write-Host "  ! TOC Heading style not found (optional)" -ForegroundColor DarkYellow
    }
    
    # Save and close
    $doc.Save()
    $doc.Close()
    
    Write-Host "`nProfessional styles applied successfully!" -ForegroundColor Green
    Write-Host "   File: custom-reference.docx" -ForegroundColor Cyan
    Write-Host "`nNext step: Rebuild thesis with Ctrl+Shift+B" -ForegroundColor Yellow
    
} catch {
    Write-Host "`nError: $_" -ForegroundColor Red
} finally {
    # Cleanup
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
