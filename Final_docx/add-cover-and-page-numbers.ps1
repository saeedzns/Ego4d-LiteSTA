# Add cover page and page numbers to thesis

$docPath = ".\build\thesis.docx"

Write-Host "Opening Word document..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $doc = $word.Documents.Open((Resolve-Path $docPath).Path)
    
    # Add cover page at the beginning
    Write-Host "Adding cover page..."
    $doc.Range(0, 0).InsertBreak(7) # Insert page break
    $coverRange = $doc.Range(0, 0)
    
    # Add university logo placeholder (centered)
    $coverRange.ParagraphFormat.Alignment = 1 # Center
    $coverRange.InsertAfter("[INSERT UNIVERSITY LOGO HERE]")
    $coverRange.Font.Size = 14
    $coverRange.Font.Bold = $true
    $coverRange.InsertParagraphAfter()
    $coverRange.InsertParagraphAfter()
    $coverRange.InsertParagraphAfter()
    
    # Add thesis title
    $titleRange = $doc.Range($coverRange.End, $coverRange.End)
    $titleRange.ParagraphFormat.Alignment = 1 # Center
    $titleRange.Font.Size = 18
    $titleRange.Font.Bold = $true
    $titleRange.InsertAfter("Ego4D-LiteSTA: A Modular Lightweight Pipeline`nfor Short-Term Object Interaction Anticipation")
    $titleRange.InsertParagraphAfter()
    $titleRange.InsertParagraphAfter()
    $titleRange.InsertParagraphAfter()
    
    # Add author and details
    $detailsRange = $doc.Range($titleRange.End, $titleRange.End)
    $detailsRange.ParagraphFormat.Alignment = 1 # Center
    $detailsRange.Font.Size = 12
    $detailsRange.Font.Bold = $false
    $detailsRange.InsertAfter("A Thesis Presented to`nThe Faculty of [Your Department]`n[Your University]")
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertAfter("In Partial Fulfillment`nof the Requirements for the Degree`nMaster of Science in Computer Science")
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertAfter("By`n[Your Name]")
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertParagraphAfter()
    $detailsRange.InsertAfter("January 2026")
    
    Write-Host "Cover page added successfully"
    
    # Add page numbers
    Write-Host "Adding page numbers..."
    foreach ($section in $doc.Sections) {
        # Add page numbers to footer
        $footer = $section.Footers.Item(1) # Primary footer
        $footer.PageNumbers.Add(2, $false) # 2 = center alignment, false = don't add to first page
        
        # Format page numbers
        $footer.Range.ParagraphFormat.Alignment = 1 # Center
        $footer.Range.Font.Size = 11
    }
    
    Write-Host "Page numbers added successfully"
    
    # Different page numbering for front matter (optional - Roman numerals)
    Write-Host "Setting up page numbering format..."
    # First section (cover to TOC) - Roman numerals
    if ($doc.Sections.Count -gt 1) {
        $doc.Sections.Item(1).Footers.Item(1).PageNumbers.NumberStyle = 1 # Roman lowercase (i, ii, iii)
    }
    
    Write-Host "Saving document..."
    $doc.Save()
    $doc.Close()
    
    Write-Host "`nDone! Successfully added:" -ForegroundColor Green
    Write-Host "  - Cover page with thesis title" -ForegroundColor Green
    Write-Host "  - University logo placeholder" -ForegroundColor Green
    Write-Host "  - Page numbers in footer (centered)" -ForegroundColor Green
    Write-Host "`nNote: Please replace [INSERT UNIVERSITY LOGO HERE] with actual logo" -ForegroundColor Yellow
    Write-Host "      and update [Your Name], [Your Department], [Your University]" -ForegroundColor Yellow
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
