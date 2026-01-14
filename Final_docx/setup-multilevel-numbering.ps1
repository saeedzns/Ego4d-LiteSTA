# Configure proper multilevel numbering for headings in custom-reference.docx
# Heading 1: No numbering (chapters)
# Heading 2: 1.1, 1.2, 1.3 (sections)
# Heading 3: 1.1.1, 1.1.2 (subsections)

$docPath = ".\custom-reference.docx"

Write-Host "Configuring multilevel heading numbering..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $doc = $word.Documents.Open((Resolve-Path $docPath).Path)
    
    # Create a multilevel list template
    Write-Host "Creating multilevel list..."
    $listTemplate = $doc.ListTemplates.Add($true)
    
    # Configure Heading 1 (Chapter level - no automatic number)
    $listTemplate.ListLevels.Item(1).NumberFormat = ""
    $listTemplate.ListLevels.Item(1).NumberStyle = 0 # No numbering
    $listTemplate.ListLevels.Item(1).NumberPosition = 0
    $listTemplate.ListLevels.Item(1).Alignment = 0
    
    # Configure Heading 2 (Section level - 1.1, 1.2, etc.)
    $listTemplate.ListLevels.Item(2).NumberFormat = "%1.%2"
    $listTemplate.ListLevels.Item(2).NumberStyle = 0  # Arabic numbers
    $listTemplate.ListLevels.Item(2).NumberPosition = 0
    $listTemplate.ListLevels.Item(2).Alignment = 0
    $listTemplate.ListLevels.Item(2).TrailingCharacter = 0 # Tab
    $listTemplate.ListLevels.Item(2).StartAt = 1
    
    # Configure Heading 3 (Subsection level - 1.1.1, 1.1.2, etc.)
    $listTemplate.ListLevels.Item(3).NumberFormat = "%1.%2.%3"
    $listTemplate.ListLevels.Item(3).NumberStyle = 0  # Arabic numbers
    $listTemplate.ListLevels.Item(3).NumberPosition = 0
    $listTemplate.ListLevels.Item(3).Alignment = 0
    $listTemplate.ListLevels.Item(3).TrailingCharacter = 0
    $listTemplate.ListLevels.Item(3).StartAt = 1
    
    # Configure Heading 4 (1.1.1.1, etc.)
    $listTemplate.ListLevels.Item(4).NumberFormat = "%1.%2.%3.%4"
    $listTemplate.ListLevels.Item(4).NumberStyle = 0
    $listTemplate.ListLevels.Item(4).NumberPosition = 0
    $listTemplate.ListLevels.Item(4).Alignment = 0
    $listTemplate.ListLevels.Item(4).TrailingCharacter = 0
    $listTemplate.ListLevels.Item(4).StartAt = 1
    
    # Link styles to list levels
    Write-Host "Linking heading styles to list levels..."
    $doc.Styles["Heading 1"].LinkToListTemplate($listTemplate, 1)
    $doc.Styles["Heading 2"].LinkToListTemplate($listTemplate, 2)
    $doc.Styles["Heading 3"].LinkToListTemplate($listTemplate, 3)
    $doc.Styles["Heading 4"].LinkToListTemplate($listTemplate, 4)
    
    Write-Host "Saving template..."
    $doc.Save()
    $doc.Close()
    
    Write-Host "`nDone! Multilevel numbering configured:" -ForegroundColor Green
    Write-Host "  Heading 1: No number (Chapter 1: Introduction)" -ForegroundColor Green
    Write-Host "  Heading 2: 1.1, 1.2, 1.3 (Background, Methods, etc.)" -ForegroundColor Green
    Write-Host "  Heading 3: 1.1.1, 1.1.2, 1.1.3 (subsections)" -ForegroundColor Green
    Write-Host "  Heading 4: 1.1.1.1, 1.1.1.2 (sub-subsections)" -ForegroundColor Green
    
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
