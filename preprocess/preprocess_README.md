# AI CUP 2024 TEAM_6178

這是初賽繳交的程式碼及相關說明。

用於資料前處理。

## 功能概述

- **PDF 文本提取**：從 PDF 中提取文字，無法提取時則使用 OCR 輔助。
- **自訂字典斷詞**：使用 CKIP 工具及自訂字典進行斷詞，提升分詞準確度。
- **問題與答案處理**：處理 FAQ 中的問題及答案，生成處理後的 JSON 文件，便於後續檢索。
- **數據轉換及存儲**：將處理過的文本分段、過濾，並儲存為 JSON 檔案。

## 安裝指南

### 1. 環境設定

請先確保安裝了 Python 3.7.16，並建議在虛擬環境中安裝依賴。

### 2. 安裝依賴套件

在專案根目錄中執行以下命令，以安裝所需的依賴套件：

```bash
pip install -r requirements.txt
```

### 3.CKIP Tagger 資料下載

官方GITHUB:
https://github.com/ckiplab/ckiptagger

請執行以下命令來下載 CKIP 的模型資料：

```bash
from ckiptagger import data_utils
data_utils.download_data_gdown("./data")
```

### 4.安裝 Tesseract OCR

請安裝 Tesseract OCR :
```bash
pip install pytesseract
```
安裝繁體中文語言包 :


Tesseract 默認不包含繁體中文語言包，你需要手動安裝：

 • Linux 或 macOS
```bash
sudo apt-get install tesseract-ocr-chi-tra
```

 • Windows

 1. 下載繁體中文語言包（chi_tra.traineddata），可以在 官方語言包下載頁面(https://github.com/tesseract-ocr/tessdata) 找到。
 2. 將下載的 chi_tra.traineddata 文件放置在 Tesseract-OCR 的 tessdata 資料夾中（例如 C:\Program Files\Tesseract-OCR\tessdata）。

並依情況設定以下環境變數：

```bash
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
```

## 使用方法

執行主程式的指令範例如下：

```bash
python tojson.py --source_path ./reference --cache_path ./json_reference
```


