# Add Word-compatible captions to figures and tables in thesis.docx
# This enables automatic List of Figures and List of Tables generation

$docPath = ".\build\thesis.docx"

Write-Host "Opening Word document..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path $docPath).Path)

Write-Host "Processing figures and tables..."

$figureCount = 0
$tableCount = 0

# Process each paragraph to find figure and table captions
foreach ($para in $doc.Paragraphs) {
    $text = $para.Range.Text.Trim()
    
    # Check if this paragraph is a figure caption (bold text starting with "Figure")
    if ($text -match '^Figure\s+(\d+)\s*[-–—]\s*(.+)$') {
        $figNum = $matches[1]
        $caption = $matches[2].Trim()
        
        # Look for the image in previous paragraphs (within 2 paragraphs back)
        $paraIndex = $para.Range.Paragraphs(1).Range.Paragraphs(1).Range.Start
        $searchRange = $doc.Range(0, $paraIndex)
        
        # Find the last inline shape (image) before this caption
        $foundImage = $null
        for ($i = $searchRange.InlineShapes.Count; $i -ge 1; $i--) {
            $shape = $searchRange.InlineShapes.Item($i)
            if ($shape.Range.End -le $paraIndex) {
                $foundImage = $shape
                break
            }
        }
        
        if ($foundImage) {
            try {
                # Position cursor after the image
                $captionRange = $foundImage.Range
                $captionRange.Collapse([Microsoft.Office.Interop.Word.WdCollapseDirection]::wdCollapseEnd)
                $captionRange.InsertParagraphAfter()
                $captionRange.Collapse([Microsoft.Office.Interop.Word.WdCollapseDirection]::wdCollapseEnd)
                
                # Insert Word caption
                $captionRange.InsertCaption(
                    [Microsoft.Office.Interop.Word.WdCaptionLabelID]::wdCaptionFigure,
                    " - $caption",
                    $null,
                    [Microsoft.Office.Interop.Word.WdCaptionPosition]::wdCaptionPositionBelow,
                    $null
                )
                
                # Delete the old caption paragraph
                $para.Range.Delete() | Out-Null
                
                $figureCount++
            } catch {
                Write-Host "  Warning: Could not process Figure $figNum" -ForegroundColor Yellow
            }
        }
    }
    
    # Check if this paragraph is a table caption (bold text starting with "Table")
    if ($text -match '^Table\s+(\d+(?:\.\d+)?)\s*[-–—]\s*(.+)$') {
        $tableNum = $matches[1]
        $caption = $matches[2].Trim()
        
        # Look for table in next paragraphs (within 3 paragraphs ahead)
        $paraIndex = $para.Range.Paragraphs(1).Range.Paragraphs(1).Range.End
        $searchRange = $doc.Range($paraIndex, $paraIndex + 500)
        
        # Find the next table after this caption
        $foundTable = $null
        foreach ($table in $doc.Tables) {
            if ($table.Range.Start -ge $paraIndex -and $table.Range.Start -le ($paraIndex + 500)) {
                $foundTable = $table
                break
            }
        }
        
        if ($foundTable) {
            try {
                # Position cursor before the table
                $captionRange = $foundTable.Range
                $captionRange.Collapse([Microsoft.Office.Interop.Word.WdCollapseDirection]::wdCollapseStart)
                
                # Insert Word caption above table
                $captionRange.InsertCaption(
                    [Microsoft.Office.Interop.Word.WdCaptionLabelID]::wdCaptionTable,
                    " - $caption",
                    $null,
                    [Microsoft.Office.Interop.Word.WdCaptionPosition]::wdCaptionPositionAbove,
                    $null
                )
                
                # Delete the old caption paragraph
                $para.Range.Delete() | Out-Null
                
                $tableCount++
            } catch {
                Write-Host "  Warning: Could not process Table $tableNum" -ForegroundColor Yellow
            }
        }
    }
}

Write-Host "`nSaving document..."
$doc.Save()
$doc.Close()
$word.Quit()

# Release COM objects
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

Write-Host "`nDone! Processed:" -ForegroundColor Green
Write-Host "  - $figureCount figure captions converted to Word format"
Write-Host "  - $tableCount table captions converted to Word format"
Write-Host "`nYou can now use Word's Insert > Table of Figures feature"
