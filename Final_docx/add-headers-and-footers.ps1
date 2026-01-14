# Add-Headers-And-Footers.ps1
# Adds running headers (chapter names on odd pages, thesis title on even pages)
# Page numbers are already added by add-cover-and-page-numbers.ps1

param(
    [string]$DocPath = "build\thesis.docx"
)

# Create Word COM object
$word = New-Object -ComObject Word.Application
$word.Visible = $false

Write-Host "Opening Word document..."
$doc = $word.Documents.Open((Resolve-Path $DocPath).Path)

try {
    Write-Host "Setting up headers..."
    
    # Access the sections
    foreach ($section in $doc.Sections) {
        # Enable different odd and even headers
        $section.PageSetup.OddAndEvenPagesHeaderFooter = $true
        
        # Odd page header (right-aligned chapter name)
        $oddHeader = $section.Headers.Item(1)  # wdHeaderFooterPrimary
        $oddHeader.Range.Text = "Chapter "
        $oddHeader.Range.Font.Size = 10
        $oddHeader.Range.Font.Name = "Times New Roman"
        $oddHeader.Range.ParagraphFormat.Alignment = 2  # wdAlignParagraphRight
        
        # Even page header (left-aligned thesis title)
        $evenHeader = $section.Headers.Item(2)  # wdHeaderFooterEvenPages
        $evenHeader.Range.Text = "Ego4D-LiteSTA: Lightweight Short-Term Action Anticipation"
        $evenHeader.Range.Font.Size = 10
        $evenHeader.Range.Font.Name = "Times New Roman"
        $evenHeader.Range.ParagraphFormat.Alignment = 0  # wdAlignParagraphLeft
    }
    
    Write-Host "Headers configured successfully"
    Write-Host "Saving document..."
    $doc.Save()
    
    Write-Host "`nDone! Successfully added:"
    Write-Host "  - Odd pages: Chapter name (right-aligned)"
    Write-Host "  - Even pages: Thesis title (left-aligned)"
    Write-Host "`nNote: Chapter names show 'Chapter' prefix. Update manually if needed for specific chapter titles."
    
} finally {
    $doc.Close()
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
