# Enhanced Retrieval System for Long Documents

這是一個增強型檢索系統，能夠從大量的 PDF 文件中提取和處理文本資料，並將處理後的資料存入 JSON 格式，以便於進一步分析和檢索。本系統運用 OCR、中文自然語言處理工具 (CKIP)、和自訂字典來支援中文文本的處理。

## 功能概述

- **PDF 文本提取**：從 PDF 中提取文字，無法提取時則使用 OCR 輔助。
- **自訂字典斷詞**：使用 CKIP 工具及自訂字典進行斷詞，提升分詞準確度。
- **問題與答案處理**：處理 FAQ 中的問題及答案，生成處理後的 JSON 文件，便於後續檢索。
- **數據轉換及存儲**：將處理過的文本分段、過濾，並儲存為 JSON 檔案。

## 安裝指南

### 1. 環境設定

請先確保安裝了 Python 3.6 或更高版本，並建議在虛擬環境中安裝依賴。

### 2. 安裝依賴套件

在專案根目錄中執行以下命令，以安裝所需的依賴套件：

```bash
pip install -r requirements.txt
```

### 3.CKIP Tagger 資料下載

請執行以下命令來下載 CKIP 的模型資料：

```bash
from ckiptagger import data_utils
data_utils.download_data_gdown("./data")
```

### 4.安裝 Tesseract OCR

請安裝 Tesseract OCR，並依情況設定以下環境變數：

```bash
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
```

## 使用方法

執行主程式的指令範例如下：

```bash
python tojson.py --source_path ./reference --cache_path ./json_reference
```


