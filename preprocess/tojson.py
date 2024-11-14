import os
import json
import argparse
from tqdm import tqdm
import pdfplumber
import logging
import pytesseract
import jieba
import re
import warnings
from ckiptagger import WS, POS, NER, data_utils, construct_dictionary

# 設定 Tesseract OCR 路徑
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
warnings.simplefilter(action='ignore', category=FutureWarning)

#data_utils.download_data_gdown("./")
# 載入 CKIP 標註工具
ws = WS("./data")
pos = POS("./data")
ner = NER("./data")

#-------------functions-------------

# 將資料存為 JSON 檔
def save_to_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# 從 JSON 檔載入資料
def load_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
    
# 將數字reference裡的一些數字轉為中文
def number_to_chinese(num):
    units = ["", "十", "百", "千", "萬", "億"]
    digits = "零一二三四五六七八九"
    result = ""
    str_num = str(num)
    length = len(str_num)
    
    for i, digit in enumerate(str_num):
        digit_value = int(digit)
        if digit_value != 0:
            result += digits[digit_value] + units[length - i - 1]
        elif not result.endswith("零"):
            result += "零"
    
    # 處理 "一十" -> "十" 的情況
    result = result.replace("一十", "十")
    # 去除末尾的 "零"
    result = result.rstrip("零")
    
    return result

#自定義字典(根據reference)
custom_dict = {}
for i in range(1, 501):  # 假設需要生成「第1條」到「第500條」
    chinese_number = number_to_chinese(i)
    custom_dict[f"第{chinese_number}條"] = 1
    custom_dict[f"第{i}季"] =1
custom_dict.update({
    "刷臉": 1, "綜合對帳單": 1, "對帳單": 1, "支付寶": 1, "本期淨利": 1,
    "約當現金": 1, "要保人": 1, "資產負債表": 1, "綜合損益表": 1, 
    "權益變動表": 1, "每股盈餘": 1, "瑞昱": 1, "華碩": 1, "智邦": 1, 
    "研華": 1, "台達電子": 1
})
custom_dictionary = construct_dictionary(custom_dict)

# CKIP 斷詞工具
def get_ckip_tokenization(text):
    words = ws([text],sentence_segmentation=True,
            segment_delimiter_set={'：', '，', '\n', '。','【','】'},recommend_dictionary=custom_dictionary)
    return words[0]

# 定義停用詞表
stopwords = {"的", "了", "之", "在", "和", "也", "有", "是","於","\n","：","。","，","「","」"}
def remove_stopwords(words, stopwords):
    # 過濾掉不需要的詞
    return [word for word in words if word not in stopwords]

# 預處理文字(針對ocr轉換出的空白做處理)
def preprocess_text(text):
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    text = re.sub(r'\s+([。，；：！？])', r'\1', text)
    return text.strip()

def normalize_text(text):
    # 去除單字間的空格
    return re.sub(r'(\w)\s(\w)', r'\1\2', text)

# 讀取 PDF 並切chunks
def read_pdf(pdf_loc, page_infos: list = None, max_tokens=512, overlap_tokens=50):
    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        logging.error(f"Error opening PDF file {pdf_loc}: {e}")
        return []
    
    try:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    except IndexError:
        logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
        pages = pdf.pages
    
    pdf_text = ''
    for page_number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text()
            if text:
                logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
                pdf_text += text + "\n\n"
            else:
                logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
                image = page.to_image(resolution=300).original
                pil_image = image.convert("RGB")
                ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra', config='--psm 6')
                if ocr_text.strip():
                    ocr_text = normalize_text(preprocess_text(ocr_text))
                    logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
                    pdf_text += ocr_text + "\n\n"
                else:
                    logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
        except Exception as e:
            logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
    pdf.close()
    
    if not pdf_text.strip():
        logging.warning(f"No text extracted from {pdf_loc}. Skipping this file.")
        return []

    # 分句並分段
    word_s = get_ckip_tokenization(pdf_text)
    sentences = remove_stopwords(word_s, stopwords)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        token_length = len(sentence)
        
        if current_length + token_length > max_tokens:
            if current_chunk:
                chunk = ' '.join(current_chunk)
                chunks.append(chunk)
                # Implement overlapping
                if overlap_tokens > 0:
                    current_chunk = current_chunk[-overlap_tokens:]
                    current_length = sum(len(jieba.lcut(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
        current_chunk.append(sentence)
        current_length += token_length
    
    # Add any remaining sentences as the last chunk
    if current_chunk:
        chunk = ' '.join(current_chunk)
        chunks.append(chunk)
    
    logging.info(f"Total chunks created from {pdf_loc}: {len(chunks)}")
    return chunks


# 讀取資料並儲存為 JSON
def load_data(source_path,json_save_path):

    '''if os.path.exists(json_save_path):
        logging.info(f"Loading data from cached JSON: {json_save_path}")
        return load_from_json(json_save_path)'''
    
    # 否則，從 PDF 提取數據並保存到 JSON 文件中
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    all_documents = []
    all_doc_ids = []
    missing_pdfs = []
    
    for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
        try:
            file_id = int(file.replace('.pdf', ''))
        except ValueError:
            logging.warning(f"Skipping non-PDF or improperly named file: {file}")
            continue
        file_path = os.path.join(source_path, file)
        splits = read_pdf(file_path)
        if not splits:
            logging.warning(f"No content extracted from file: {file_path}")
            missing_pdfs.append(file)
        corpus_dict[file_id] = splits
        all_documents.extend(splits)
        all_doc_ids.extend([file_id] * len(splits))

    # Save the extracted data to JSON
    save_data = {
        "corpus_dict": corpus_dict,
        "all_documents": all_documents,
        "all_doc_ids": all_doc_ids
    }
    save_to_json(save_data, json_save_path)
    
    if missing_pdfs:
        logging.info(f"Total missing PDFs: {len(missing_pdfs)}")
        for pdf in missing_pdfs:
            logging.info(f"Missing PDF: {pdf}")
    
    return save_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Retrieval System for Long Documents.')
    parser.add_argument('--source_path', type=str, default='./reference', required=True, help='Path to the source data directory.')
    parser.add_argument('--cache_path', type=str, default='./json_reference', help='Directory to save cached JSON data.')
    args = parser.parse_args()
    answer_dict = {"answers": []}

    #-------------finance & insurance data處理----------------
    
    finance_cache = os.path.join(args.cache_path, 'finance_data.json')
    insurance_cache = os.path.join(args.cache_path, 'insurance_data.json')

    corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'), finance_cache)
    corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'), insurance_cache)

    print("資料處理完成，已儲存至 json_reference")

    #-------------faq data處理----------------
    # 讀取原始 JSON 檔案
    with open('./reference/faq/pid_map_content.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 對每個問題和答案進行斷詞和過濾
    processed_data = {}

    for pid, items in data.items():
        processed_items = []
        for item in items:
            # 處理問題
            question = item.get("question", "")
            tokenized_question = get_ckip_tokenization(question)
            filtered_question = remove_stopwords(tokenized_question, stopwords)
            question_str = ' '.join(filtered_question)
        
            # 處理答案
            processed_answers = []
            for answer in item['answers']:
                tokenized_answer = get_ckip_tokenization(answer)
                filtered_answer = remove_stopwords(tokenized_answer, stopwords)
                answer_str = ' '.join(filtered_answer)
                processed_answers.append(answer_str)
        
            # 加入處理後的問題和答案
            processed_items.append({
                "問題": question_str,
                "答案": processed_answers
            })
    
        # 將處理後的結果儲存回新字典
        processed_data[pid] = processed_items

    # 將處理後的資料寫入新的 JSON 檔案
    with open('processed_pid_map_content.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print("資料處理完成，已儲存至 processed_pid_map_content.json")

    del ws
    del pos
    del ner

    
