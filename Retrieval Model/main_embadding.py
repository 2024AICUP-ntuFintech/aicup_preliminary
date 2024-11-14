import os
os.environ["TRANSFORMERS_NO_LOGGING"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from ckiptagger import WS, POS, NER, data_utils, construct_dictionary
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from snownlp import SnowNLP
from FlagEmbedding import FlagReranker
import warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#--------------------------模型下載----------------------------------
# embadding 模型
tokenizer_s = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model_s = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

tokenizer_l = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model_l = AutoModel.from_pretrained('intfloat/multilingual-e5-large')


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

#model_jina = SentenceTransformer("jinaai/jina-embeddings-v3")

# reranker 模型
reranker_model = SentenceTransformer('BAAI/bge-reranker-base')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reranker_bge = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
reranker_bce = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1')
tokenizer_bce = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1')
reranker_bce.to(device)
reranker_bce.eval()
model_l = model_l.to(device)


#--------------------------------------------------------------------

ws = WS("./data")
pos = POS("./data")
#ner = NER("./data/data")

# Function to load data from JSON file
def load_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
    
# Function to load data from source path
def load_data(source_path,json_save_path):

    if os.path.exists(json_save_path):
        logging.info(f"Loading data from cached JSON: {json_save_path}")
        data = load_from_json(json_save_path)
        # 返回格式確認
        return data.get("corpus_dict", {}), data.get("all_documents", {}), data.get("all_doc_ids", {})

def load_questions(question_path):
    try:
        with open(question_path, 'r', encoding='utf8') as f:
            qs_ref = json.load(f)
        logging.info(f"Loaded questions from {question_path}")
    except Exception as e:
        logging.error(f"Error reading question file {question_path}: {e}")
        return None
    return qs_ref

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

custom_dict = {}
custom_dict["對帳單"]=1
custom_dict["刷臉"] = 1
custom_dict["綜合對帳單"] = 1
custom_dict["對帳單"] =1
custom_dict["支付寶"] =1
custom_dict["本期淨利"] =1
custom_dict["約當現金"] =1
custom_dict["要保人"] =1

custom_dict["資產 負債 表"] =1
custom_dict["綜合 損益 表"] =1
custom_dict["權益 變動 表"] =1
custom_dict["每股盈餘"]=1

custom_dict["瑞昱"] =1
custom_dict["華碩"] =1
custom_dict["智邦"] =1
custom_dict["研華"] =1
custom_dict["台達電子"] =1
for i in range(1, 501):  # 假設需要生成「第1條」到「第500條」
    chinese_number = number_to_chinese(i)
    custom_dict[f"第{chinese_number}條"] = 1
    custom_dict[f"第{i}季"] =1

custom_dictionary = construct_dictionary(custom_dict)

# 定義停用詞表
stopwords = {"的", "了", "之", "在", "和", "也", "有", "是","於","\n","：","。","，","「","」","【","】","、","；","「","」"}
def remove_stopwords(words, stopwords):
    # 過濾掉不需要的詞
    return [word for word in words if word not in stopwords]

def get_ckip_tokenization(text):
    words = ws([text],sentence_segmentation=True,
            segment_delimiter_set={'：', '，', '\n', '。','【','】'},recommend_dictionary=custom_dictionary)
    tokenized_query = remove_stopwords(words[0], stopwords)
    return tokenized_query

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

# 針對 FAQ 的處理
def process_faq(query, relevant_docs, relevant_doc_ids):
    tokenized_query = get_ckip_tokenization(query)
    filtered_query = remove_stopwords(tokenized_query, stopwords)
    question_str = ' '.join(filtered_query)
    print("query: ",question_str)
    input_texts = [f"query: {question_str}"] + [f"passage: {doc}" for doc in relevant_docs]
    embadding_scores = embadding_e5_l(input_texts, relevant_doc_ids)
    return embadding_scores
    
def process_finance_insurance(relevant_docs,relevant_doc_ids):
    # 進行數據驗證：確保所有相關文檔都是字符串
    valid_relevant_docs = []
    valid_relevant_doc_ids = []
    invalid_docs = []
    for doc, doc_id in zip(relevant_docs, relevant_doc_ids):
        if isinstance(doc, str) and doc.strip():
            valid_relevant_docs.append(doc)
            valid_relevant_doc_ids.append(doc_id)
        else:
            invalid_docs.append((qid, doc_id, doc))
        
    if invalid_docs:
        for invalid in invalid_docs:
            logging.error(f"QID: {invalid[0]}, Invalid Document ID: {invalid[1]}, Document: {invalid[2]}")

        # 確認所有 relevant_docs 都是字符串
    for idx, doc in enumerate(valid_relevant_docs):
        if not isinstance(doc, str):
            logging.error(f"QID: {qid}, Relevant Doc Index: {idx}, Document is not a string: {doc}")

    return valid_relevant_docs,valid_relevant_doc_ids

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

# Function to compute reranker2 scores using AutoModelForSequenceClassification
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
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Retrieval System for Long Documents.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--cache_path', type=str, default='./json_reference', help='Directory to save cached JSON data.')
    parser.add_argument('--embadding_weight', type=float, default=1.0, help='Weight for Embadding scores.')
    parser.add_argument('--reranker_weight', type=float, default=0.0, help='Weight for Reranker scores.')
    parser.add_argument('--max_sentences', type=int, default=10, help='Maximum number of sentences per chunk.')
    parser.add_argument('--overlap', type=int, default=2, help='Number of overlapping sentences between chunks.')
    parser.add_argument('--model_type', type=str, choices=['bge', 'bce', 'both'], default='both', help='Select model type to use for scoring.')


    args = parser.parse_args()

    answer_dict = {"answers": []}
    # Load reference quedata, using cached JSON izxssssxf available
    finance_cache = os.path.join(args.cache_path, 'finance_data.json')
    insurance_cache = os.path.join(args.cache_path, 'insurance_data.json')

    qs_ref = load_questions(args.question_path)
    if qs_ref is None:
        print("Failed to load questions.")
    else:
        print("Questions loaded successfully.")
    
    # Ensure the cache directory exists
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

    # Load reference data
    corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'), finance_cache)
    corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'), insurance_cache)
    #print(corpus_dict_insurance.get('620', []))
    
    # Load FAQ mapping and split
    try:
        with open(os.path.join(args.source_path, 'faq/processed_pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # Read reference data file
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []
    
    # Check for required FAQ doc_ids
    required_faq_doc_ids = set()
    for q in qs_ref.get('questions', []):
        if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
            required_faq_doc_ids.update(q.get('source', []))
    
    missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
    if missing_faq_doc_ids:
        logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
    else:
        logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

    # Prepare FAQ documents
    faq_documents = []
    faq_doc_ids = []
    for key, value in key_to_source_dict.items():
        for q in value:
            combined = f"問題：{q['問題']} 答案：{' '.join(q['答案'])}"
            faq_documents.append(combined)
            faq_doc_ids.append(key)

    faq_dict = {}
    for doc_id, doc in zip(faq_doc_ids, faq_documents):
        if doc_id not in faq_dict:
            faq_dict[doc_id] = []
        faq_dict[doc_id].append(doc)
    
    # Aggregate all documents
    all_documents = documents_finance + documents_insurance 
    all_doc_ids = doc_ids_finance + doc_ids_insurance
    #print(all_doc_ids)

    # Build a mapping from doc_id to list of indices in all_documents
    doc_id_to_indices = {}
    for idx, doc_id in enumerate(all_doc_ids):
        if doc_id not in doc_id_to_indices:
            doc_id_to_indices[doc_id] = []
        doc_id_to_indices[doc_id].append(idx)
    

    # 存放所有結果
        all_answers = []
    # Process each question
    for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
        qid = q_dict.get('qid')
        query = q_dict.get('query')
        category = q_dict.get('category')
        source = q_dict.get('source')
        
        if not all([qid, query, category, source]):
            logging.warning(f"Skipping incomplete question entry: {q_dict}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue
        
        # Select the appropriate corpus
        if category == 'finance':
            corpus_dict = corpus_dict_finance
            docs = documents_finance
            doc_ids = doc_ids_finance
            relevant_docs = []
            relevant_doc_ids = []
            for doc_id in source:
                docs_for_id = corpus_dict.get(str(doc_id), [])
                if not docs_for_id:
                    logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
                relevant_docs.extend(docs_for_id)
                relevant_doc_ids.extend([doc_id] * len(docs_for_id))
            valid_relevant_docs,valid_relevant_doc_ids = process_finance_insurance(relevant_docs,relevant_doc_ids)
            tokenized_query = get_ckip_tokenization(query)
            filtered_query = remove_stopwords(tokenized_query, stopwords)
            question_str = ' '.join(filtered_query)
            #print(question_str)

            # Compute scores based on selected model type
            if args.model_type in ['bge', 'both']:
                rerank_bge_score = rerank_with_bge(valid_relevant_docs)
            if args.model_type in ['bce', 'both']:
                rerank_bce_score = rerank_with_bce(question_str, valid_relevant_docs, reranker_bce, tokenizer_bce, device)

            # Calculate final scores based on model choice
            if args.model_type == 'bge':
                final_scores = rerank_bge_score
            elif args.model_type == 'bce':
                final_scores = rerank_bce_score
            else:  # both
                final_scores = [(bge + bce) / 2 for bge, bce in zip(rerank_bge_score, rerank_bce_score)]

            # 將文檔與平均分數及 doc_id 組合
            doc_scores = list(zip(valid_relevant_docs, final_scores, valid_relevant_doc_ids))
            # 按平均分數降序排序
            top_n = 1  # 根據需求調整
            top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]
            if not top_n_docs:
                logging.warning(f"No documents after reranking for QID: {qid}")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue
            # 選擇最佳文檔
            best_doc, best_score, best_doc_id = top_n_docs[0]
            logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Average Score: {best_score:.4f}")
            #print(best_doc_id)
            # 將結果添加到 answer_dict
            answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

        elif category == 'insurance':
            corpus_dict = corpus_dict_insurance
            docs = documents_insurance
            doc_ids = doc_ids_insurance
            relevant_docs = []
            relevant_doc_ids = []
            for doc_id in source:
                docs_for_id = corpus_dict.get(str(doc_id), [])
                if not docs_for_id:
                    logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
                relevant_docs.extend(docs_for_id)
                relevant_doc_ids.extend([doc_id] * len(docs_for_id))

            valid_relevant_docs,valid_relevant_doc_ids = process_finance_insurance(relevant_docs,relevant_doc_ids)
            tokenized_query = get_ckip_tokenization(query)
            filtered_query = remove_stopwords(tokenized_query, stopwords)
            question_str = ' '.join(filtered_query)

             # Compute scores based on selected model type
            if args.model_type in ['bge', 'both']:
                rerank_bge_score = rerank_with_bge(valid_relevant_docs)
            if args.model_type in ['bce', 'both']:
                rerank_bce_score = rerank_with_bce(question_str, valid_relevant_docs, reranker_bce, tokenizer_bce, device)

            # Calculate final scores based on model choice
            if args.model_type == 'bge':
                final_scores = rerank_bge_score
            elif args.model_type == 'bce':
                final_scores = rerank_bce_score
            else:  # both
                final_scores = [(bge + bce) / 2 for bge, bce in zip(rerank_bge_score, rerank_bce_score)]

            # 將文檔與平均分數及 doc_id 組合
            doc_scores = list(zip(valid_relevant_docs, final_scores, valid_relevant_doc_ids))
            # 按平均分數降序排序
            top_n = 1  # 根據需求調整
            top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]
            if not top_n_docs:
                logging.warning(f"No documents after reranking for QID: {qid}")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})
                continue
            # 選擇最佳文檔
            best_doc, best_score, best_doc_id = top_n_docs[0]
            logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Average Score: {best_score:.4f}")

            # 將結果添加到 answer_dict
           # print(best_doc_id)
            answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

        elif category == 'faq':
            corpus_dict = key_to_source_dict
            docs = faq_documents
            doc_ids = faq_doc_ids
            relevant_docs = []
            relevant_doc_ids = []
            for key in source:
                docs_for_key = corpus_dict.get(key, [])
                if not docs_for_key:
                    logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
                relevant_docs.extend(docs_for_key)
                relevant_doc_ids.extend([key] * len(docs_for_key))
            
            relevant_doc_ids_u = list(dict.fromkeys(relevant_doc_ids))
            source_passage = [faq_dict[doc_id] for doc_id in relevant_doc_ids_u]
            embadding_scores = process_faq(query, source_passage, relevant_doc_ids_u)
            #print(embadding_scores)
            #print(qid,": ", source_passage)
            # Combine aggregated scores with weights
            final_scores = {}
            for doc_id in embadding_scores:
                final_scores[doc_id] = (1.0 * embadding_scores[doc_id])
            # Sort documents by final_scores in descending order
            sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            # Select the top document
            if sorted_docs:
                best_doc_id, best_score = sorted_docs[0]
                logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Final Score: {best_score:.4f}")
                print(best_doc_id)
                answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})
            else:
                logging.warning(f"No documents after reranking for QID: {qid}")
                answer_dict['answers'].append({"qid": qid, "retrieve": None})

        else:
            logging.warning(f"Unknown category '{category}' for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Get indices of relevant_docs in all_documents
        relevant_indices = []
        for doc_id in source:
            indices = doc_id_to_indices.get(doc_id, [])
            if not indices:
                logging.warning(f"QID: {qid} - doc_id {doc_id} not found in corpus.")
            relevant_indices.extend(indices)

        #print(embadding_scores)

    # Save results
    try:
        with open(args.output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_path}: {e}")



