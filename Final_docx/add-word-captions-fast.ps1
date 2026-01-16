# Fast Word Caption Converter
# Converts bold "Figure X" and "Table X" text to proper Word captions

$docPath = "build\thesis.docx"

Write-Host "Opening document..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open((Resolve-Path $docPath).Path)

Write-Host "Processing captions (this may take 2-3 minutes)..."

# Find all bold paragraphs starting with Figure or Table
$figureCount = 0
$tableCount = 0

foreach ($para in $doc.Paragraphs) {
    if ($para.Range.Bold -eq -1) {  # Bold text
        $text = $para.Range.Text.Trim()
        
        # Match Figure pattern
        if ($text -match '^Figure\s+(\d+)\s*[-–—]\s*(.+)$') {
            $num = $Matches[1]
            $caption = $Matches[2].Trim()
            
            # Replace with Word caption
            $para.Range.Delete()
            $para.Range.InsertCaption(
                "Figure",          # Label
                " - $caption",     # Text after number
                $null,            # Position
                $null,            # Use label
                $false            # Exclude label from caption
            )
            $figureCount++
            
            if ($figureCount % 10 -eq 0) {
                Write-Host "  Processed $figureCount figures..."
            }
        }
        # Match Table pattern
        elseif ($text -match '^Table\s+(\d+\.?\d*)\s*[-–—]\s*(.+)$') {
            $num = $Matches[1]
            $caption = $Matches[2].Trim()
            
            $para.Range.Delete()
            $para.Range.InsertCaption(
                "Table",
                " - $caption",
                $null,
                $null,
                $false
            )
            $tableCount++
            
            if ($tableCount % 10 -eq 0) {
                Write-Host "  Processed $tableCount tables..."
            }
        }
    }
}

Write-Host "`nSaving document..."
$doc.Save()
$doc.Close()
$word.Quit()

[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

Write-Host "`nDone! Converted:"
Write-Host "  - $figureCount figures"
Write-Host "  - $tableCount tables"
Write-Host "`nYou can now use 'Insert > Table of Figures' in Word"
