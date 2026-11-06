# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603948
- **Họ và tên:** Mai Danh Hiếu
- **SEED** (in ở ô đầu notebook): 9693

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

                                precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.832      0.903        0.788   
2. BoW + MLP (deep learning)         0.989   0.905      0.945        0.947   
3a. Encoder (MiniLM) + kNN           0.826   0.947      0.882        0.947   
3b. Encoder (MiniLM) + LogReg        0.958   0.958      0.958        1.000   
3c. Encoder (MiniLM) + MLP head      0.957   0.947      0.952        1.000   
4a. LLM nhỏ zero-shot                0.679   0.950      0.792        0.792   
4b. LLM lớn zero-shot                0.826   0.950      0.884        0.884   
4c. LLM nhỏ few-shot                 1.000   0.450      0.621        0.621   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.074      21568.143       0.054   
2. BoW + MLP (deep learning)        14.919      25181.136       0.334   
3a. Encoder (MiniLM) + kNN         141.219         27.513      49.848   
3b. Encoder (MiniLM) + LogReg      141.895         31.294      36.813   
3c. Encoder (MiniLM) + MLP head    142.332         31.776      36.624   
4a. LLM nhỏ zero-shot                0.000          1.144     815.461   
4b. LLM lớn zero-shot                0.000          1.569     520.838   
4c. LLM nhỏ few-shot                 0.000          1.953     470.717   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.110           0.0  
2. BoW + MLP (deep learning)          2.628           0.0  
3a. Encoder (MiniLM) + kNN           74.934           0.0  
3b. Encoder (MiniLM) + LogReg        55.443           0.0  
3c. Encoder (MiniLM) + MLP head      47.445           0.0  
4a. LLM nhỏ zero-shot              1281.852           0.0  
4b. LLM lớn zero-shot              1092.805           0.0  
4c. LLM nhỏ few-shot                747.682           0.0  

## 3. Dự đoán và thực tế

[14:35:27] ml_f1: F1 ~0.7, recall cao hơn vì lượng dữ liệu 'ham' lớn hơn, model học tốt hơn ở class này, dẫn đến ít bỏ sót hơn
Thực tế: F1~0.9

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là BoW. Điểm trên test quá lạc quan do dữ liệu tồn tại trùng lặp temlate, model dễ học shortcut.
BoW và LLM đều gặp khó ở nhóm né từ khóa, thao túng.
LLM bị lừa. Phỏng thủ ngoài prompt: chuẩn hóa đầu vào thay vì giao cho LLM.
Với nhóm né từ khóa, ta có thể tìm từ gần nghĩa. Rủi ro là làm phức tạp mô hình.
50 case không đại diện cho spam thật. Bộ test này nên dùng để đánh giá khả năng tổng quát hóa của model, không nên dùng làm final test.

## 5. Thử thách cá nhân

🎯 Đề #4: Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?
# ✏️ Code thử thách của bạn ở đây
embed = lambda texts: emb_model.encode(list(texts), batch_size=128)
RESULTS2 = {}
for k in [1, 5, 25]:
  pred_knn, bench_knn = benchmark(lambda texts: knn_predict(E_tr, y_tr, embed(texts), k=k), texts_te, texts_te[SUB])
  m_knn = metrics(y_te, pred_knn)

  print(f"kNN   k = {k}", fmt_bench(enc_tr_time, bench_knn))
  RESULTS2[f"Encoder (MiniLM) + kNN k = {k}"] = {**m_knn, "f1_SUB": metrics(y_te[SUB], pred_knn[SUB])["f1"], "train_s": enc_tr_time, **bench_knn, "usd_per_1k": 0.0}
pd.DataFrame(RESULTS2).T[["accuracy", "precision", "recall", "f1"]].round(3)

kNN   k = 1 ⏱ train 141.2s · inference cả lô 26 tin/s · từng tin p50 55.06 ms / p95 92.81 ms
kNN   k = 5 ⏱ train 141.2s · inference cả lô 27 tin/s · từng tin p50 52.83 ms / p95 76.16 ms
kNN   k = 25 ⏱ train 141.2s · inference cả lô 27 tin/s · từng tin p50 115.94 ms / p95 223.02 ms
accuracy	precision	recall	f1
Encoder (MiniLM) + kNN k = 1	0.970	0.846	0.926	0.884
Encoder (MiniLM) + kNN k = 5	0.969	0.826	0.947	0.882
Encoder (MiniLM) + kNN k = 25	0.959	0.794	0.895	0.842


## 6. Phương án cho sản phẩm thật

Từ kết quả đánh giá, ta có thể chọn BoW và Embedding + LogReg. Phương án tốt nhất, đảm bảo cả độ trễ và chất lượng là Embedding + LogReg.
Mô hình lai đề xuất: Embedding + LogReg với ưu điểm nhanh, rẻ; LLM cho trường hợp khó, độ tin cậy thấp. Nó tốt hơn vì kết hợp được ưu điểm của cả 2: nhanh, rẻ và khả năng suy luận từ LLM.
Dự đoán sai nhiều nhất: khả năng của LLM. Thay đổi hiểu biết: LLM chỉ sực sự tốt khi con người biết cách kiểm soát nó.

## 7. AI log và điều đã học

Phần	Bạn hỏi AI gì (prompt)	AI trả lời/viết gì	Bạn đã sửa / kiểm chứng gì
1	Một số kiến thức về Numpy	Giải thích	Yes
2	Custom API self host LLM	Hướng dẫn chi tiết	Yes
