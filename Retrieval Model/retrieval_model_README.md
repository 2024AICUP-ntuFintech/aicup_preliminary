# AI CUP 2024 TEAM_6178

這是初賽繳交的程式碼及相關說明。

本倉庫提供一個增強的文件檢索系統，用於處理財務、保險和 FAQ 等類別的長文檔。該系統利用嵌入模型和重排序模型，根據用戶查詢精確檢索文件，並通過自定義詞典和停用詞過濾來優化文本處理。

## 目錄
- [功能](#功能)
- [模型與技術](#模型與技術)
- [項目結構](#項目結構)
- [處理流程](#處理流程)
- [環境設置](#環境設置)
- [使用方法](#使用方法)
- [依賴項](#依賴項)

## 功能
- 多語言嵌入模型進行跨語言的文檔表示。
- 重排序模型用於根據查詢優化文件的相關性。
- 基於 CKIP 的中文分詞和停用詞過濾處理。
- 數據緩存以提高處理效率並減少冗餘。

## 模型與技術

### embadding模型(針對faq)
- **`multilingual-e5-large `**
### 重排序模型
- **`BAAI/bge-reranker-large`**：對文檔進行重排序，根據句子相關性進行優化。
- **`maidalun1020/bce-reranker-base_v1`**：使用 BCE（二元交叉熵）的分類模型來判定文檔相關性。

### 其他 NLP 技術
- **CKIP 分詞**：使用自定義詞典進行中文語言分詞和命名實體識別。

## 項目結構
項目目錄結構如下：

```bash
C:.
├─preprocess
│  │  preprocess_README.md
│  │  requirements.txt
│  │  tojson.py
│  │
│  ├─data
│  │      LICENSE
│  │
│  ├─json_reference
│  │      finance_data.json
│  │      insurance_data.json
│  │
│  └─reference
│      ├─faq
│      ├─finance
│      └─insurance
└─Retrieval Model
    │  main_embadding.py
    │  requirements.txt
    │  reranker_README.md
    │  retrieval_model_README.md
    │
    ├─json_reference
    │      finance_data.json
    │      insurance_data.json
    │      
    └─競賽訓練資料集
        └─競賽資料集
            ├─dataset
            │  └─preliminary
            │          cal_points.py
            │          ground_truths_example.json
            │          pred_retrieve.json
            │          questions_example.json
            │          questions_preliminary.json
            │
            └─reference
                ├─faq
                │      pid_map_content.json
                │      processed_pid_map_content.json
                │
                ├─finance
                └─insurance

```

## 處理流程
1. **加載數據**：加載問題和財務、保險及 FAQ 的源文檔。
2. **分詞處理**：使用 CKIP 分詞，基於自定義詞典進行分割，並去除停用詞。
3. **生成嵌入**：使用 multilingual-e5 模型為查詢和文檔生成嵌入。
4. **重排序**：根據相似性得分，應用 BCE & BGE 分類器進行重排序。
5. **最終評分和選擇**：基於權重聚合分數，並為每個查詢選擇最相關的文檔。
6. **緩存和保存結果**：將結果緩存到 JSON 文件中，以便下次運行更快。

## 變量和關鍵組件

### 主要變量和函數

- **`ws`**：CKIP 的分詞和詞性標註模型，用於處理中文文本。
- **`custom_dict`**：用戶定義的自定義詞典（例如財務術語），幫助分詞器有效地進行分割。
- **`stopwords`**：常見的無意義詞（如 "的", "了"）會被過濾掉，以提高檢索相關性。
- **`get_ckip_tokenization`**：使用 CKIP 對文本進行分詞，應用自定義詞典並去除停用詞。
- **`embadding_e5_l`**：使用 multilingual-e5-large 嵌入並規範化一批文檔和查詢文本。
- **`rerank_with_bge`** 和 **`rerank_with_bce`**：基於 BGE 和 BCE 的重排序函數，用於文檔相關性評分。


### `embadding_e5_l` 函數說明

`embadding_e5_l` 函數是檢索系統中用於生成嵌入並計算相似度的核心組件之一。它使用 `multilingual-e5-large` 模型對查詢文本和候選文檔進行嵌入生成。最終，該函數計算查詢和每個文檔的相似度得分並返回一個包含文檔 ID 和相似度的字典。

#### 函數簡介

官方資料 : 
https://huggingface.co/intfloat/multilingual-e5-large


```python
def embadding_e5_l(input_texts, relevant_doc_ids):
    if not input_texts or not relevant_doc_ids:
        return {}  # Return empty dict if inputs are empty
    
    # Tokenize input texts
    batch_dict = tokenizer_l(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    # Move tensors to the GPU
    batch_dict = {key: tensor.to(device) for key, tensor in batch_dict.items()}
    
    # Use no_grad for inference
    with torch.no_grad():
        outputs = model_l(**batch_dict)
    
    # Apply average pooling
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity scores
    query_embedding = embeddings[0]  # First is the query embedding
    doc_embeddings = embeddings[1:]  # Remaining are document embeddings
    scores = (query_embedding @ doc_embeddings.T) * 100
    
    # Convert scores to a Python list
    scores = scores.tolist()

    # Store max scores for each document ID
    doc_score_dict = {doc_id: score for score, doc_id in zip(scores, relevant_doc_ids)}
    
    # Cleanup
    del outputs, batch_dict
    torch.cuda.empty_cache()

    return doc_score_dict
```


#### 參數

- `input_texts` (`List[str]`): 包含查詢文本和候選文檔的列表。第一個文本應為查詢，其餘為候選文檔。
- `relevant_doc_ids` (`List[str]`): 與候選文檔對應的文檔 ID 列表。每個 ID 將與計算得分配對。

#### 函數步驟

##### 1. Tokenize 輸入文本
使用 `tokenizer_l` 將 `input_texts` 進行分詞處理，確保文本長度不超過 512，並填充和截斷以適應批處理：
```python
batch_dict = tokenizer_l(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
```
分詞後的批次字典 `batch_dict` 包含查詢和候選文檔的嵌入所需的張量。

##### 2. 移動到 GPU
將 `batch_dict` 中的張量移動到 GPU 上以加速計算：
```python
batch_dict = {key: tensor.to(device) for key, tensor in batch_dict.items()}
```

##### 3. 返回結果字典
將每個文檔的得分與其 ID 配對，並存入字典 `doc_score_dict`，最終返回此字典：
```python
doc_score_dict = {doc_id: score for score, doc_id in zip(scores, relevant_doc_ids)}
```
#### 返回值

- `doc_score_dict` (`Dict[str, float]`): 字典形式的結果，鍵為候選文檔 ID，值為其與查詢的相似度得分。




### `rerank_with_bge` 函數說明

`rerank_with_bge` 函數是長文件檢索系統中的一個核心組件，用於根據查詢的相關性對候選文檔進行重排序。該函數利用 `FlagReranker` 模型，針對每個查詢-文檔對計算相似度得分，並返回標準化後的得分，以便後續的檢索排序。

#### 函數簡介

```python
def rerank_with_bge(valid_relevant_docs):
    # Reranker1: compute scores using FlagReranker
    try:
        # 準備 [query, passage] 的對列表
        score_inputs = [[query, doc] for doc in valid_relevant_docs]
        # 使用 FlagReranker 計算分數
        reranker_bge_scores = reranker_bge.compute_score(score_inputs)
        logging.info(f"QID: {qid}, Reranker1 scores: {reranker_bge_scores[:5]}")  # 只記錄前5個分數以簡化日誌

        # 標準化 Reranker1 分數到 -1 ~ 1
        normalized_reranker_bge_scores = [score / 10 for score in reranker_bge_scores]
        logging.info(f"QID: {qid}, Normalized Reranker1 scores: {normalized_reranker_bge_scores[:5]}")
    except Exception as e:
        logging.error(f"Error during reranking with Reranker1 for QID: {qid}: {e}")
        answer_dict['answers'].append({"qid": qid, "retrieve": None})

    return normalized_reranker_bge_scores
```

#### 參數

- `valid_relevant_docs` (`List[str]`): 候選的相關文檔列表。每個文檔是一個字符串，包含與查詢相似的內容。

#### 函數步驟

##### 1. 准備輸入數據
將查詢與候選文檔組合成查詢-文檔對的列表，每一對作為 `FlagReranker` 的輸入：
```python
score_inputs = [[query, doc] for doc in valid_relevant_docs]
```
其中 `query` 是用戶的查詢，`doc` 是候選的文檔。

##### 2. 計算相似度得分
使用 `FlagReranker` 的 `compute_score` 方法對每一對查詢-文檔對計算相似度得分：
```python
reranker_bge_scores = reranker_bge.compute_score(score_inputs)
```
此方法會返回每個查詢-文檔對的分數。這些分數表示查詢與文檔之間的相關性。

##### 3. 標準化得分
將原始得分標準化至 -1 到 1 之間，便於後續排序和處理：
```python
normalized_reranker_bge_scores = [score / 10 for score in reranker_bge_scores]
```
標準化操作會將得分縮放至一個統一範圍，使其易於比較。


#### 返回值

- `normalized_reranker_bge_scores` (`List[float]`): 標準化的重排序得分列表。每個分數對應一個查詢-文檔對，用於最終的排序。


### `rerank_with_bce` 函數說明

`rerank_with_bce` 函數是檢索系統中的一個重排序組件，用於基於 BCE（二元交叉熵）模型計算查詢與候選文檔之間的相關性分數。此函數使用 `AutoModelForSequenceClassification` 模型對每個查詢-文檔對進行評分，並通過批次處理提高計算效率。最後返回所有候選文檔的相關性分數列表。

#### 函數簡介

```python
def rerank_with_bce(query, documents, reranker_model, reranker_tokenizer, device, batch_size=32):
    try:
        reranker_model.eval()
        sentence_pairs = [[query, doc] for doc in documents]
        scores = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Computing Reranker2 Scores"):
                batch = sentence_pairs[i:i+batch_size]
                inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = reranker_model(**inputs)
                logits = outputs.logits.view(-1).float()
                # 應用 sigmoid 轉換為概率分數
                probabilities = torch.sigmoid(logits)
                batch_scores = probabilities.cpu().numpy()
                scores.extend(batch_scores)
        logging.info(f"QID: {qid}, Reranker2 scores: {scores[:5]}")  # 只記錄前5個分數以簡化日誌
    except Exception as e:
        logging.error(f"Error during reranking with Reranker2 for QID: {qid}: {e}")
        answer_dict['answers'].append({"qid": qid, "retrieve": None})
    return scores
```


#### 參數

- `query` (`str`): 用戶查詢文本。
- `documents` (`List[str]`): 候選的相關文檔列表，每個文檔為一個字符串。
- `reranker_model` (`AutoModelForSequenceClassification`): 預訓練的 BCE 重排序模型。
- `reranker_tokenizer` (`AutoTokenizer`): 與重排序模型相對應的分詞器，用於將文本轉換為模型所需的輸入格式。
- `device` (`str`): 指定模型運行的設備（如 `cpu` 或 `cuda`）。
- `batch_size` (`int`, 預設值為 32): 指定處理批次的大小，以便於批量計算提升性能。

#### 函數步驟

##### 1. 構建查詢-文檔對
將查詢與候選文檔組合成查詢-文檔對的列表，作為模型輸入：
```python
sentence_pairs = [[query, doc] for doc in documents]
```
其中 `query` 是用戶的查詢，`doc` 是候選的文檔。

##### 2. 批次處理分詞
使用 `reranker_tokenizer` 將查詢-文檔對進行批次分詞，並設定 `padding` 和 `truncation` 參數以適應批次處理：
```python
inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
```

##### 3. 運行模型並計算 logits
將批次輸入傳遞給重排序模型 `reranker_model`，並計算每個查詢-文檔對的 logits 值：
```python
outputs = reranker_model(**inputs)
logits = outputs.logits.view(-1).float()
```

##### 4. 應用 Sigmoid 函數
使用 `sigmoid` 函數將 logits 轉換為概率分數，表示每個查詢-文檔對的相關性：
```python
probabilities = torch.sigmoid(logits)
```

##### 5. 累積得分
將每個批次的分數轉換為 NumPy 格式，並將其擴展至最終的 `scores` 列表中：
```python
batch_scores = probabilities.cpu().numpy()
scores.extend(batch_scores)
```

##### 6. 返回相關性分數列表
最終返回 `scores` 列表，其中包含所有候選文檔的相似度分數。

#### 返回值

- `scores` (`List[float]`): 所有候選文檔與查詢的相關性分數列表，每個分數對應一個查詢-文檔對。




## 環境設置

### 先決條件
- **Python**：版本 3.7.16 
- **支持 CUDA 的 GPU**（可選，但推薦以加快處理速度）

### 安裝依賴項
使用以下命令安裝所需的包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 參數

- `--question_path`：問題 JSON 文件的路徑，用於指定包含查詢問題的文件位置。
- `--source_path`：源數據目錄的路徑，系統將從此目錄加載需要檢索的財務和保險數據。
- `--output_path`：保存輸出 JSON 文件的路徑，檢索結果將以 JSON 格式保存在此文件中。
- `--cache_path`：保存緩存 JSON 數據的目錄，用於加快後續的數據加載速度。

- `--model_type`：選擇 `bge`, `bce`, 或 both 作為重排序模型類型。
  - **`bge`**：僅使用 BGE（`BAAI/bge-reranker-large`）進行重排序。
  - **`bce`**：僅使用 BCE（`maidalun1020/bce-reranker-base_v1`）進行重排序。
  - **`both`**：同時使用 BGE 和 BCE 兩個模型進行重排序，並將兩者的得分加權平均後排序。


請執行以下指令:


```bash
python main_embadding.py --question_path=競賽訓練資料集/競賽資料集/dataset/preliminary/questions_example.json --source_path=../preprocess/json_reference --output_path=競賽訓練資料集/競賽資料集/dataset/preliminary/pred_retrieve.json --cache_path=json_reference/ --model_type both

