# Bài kiểm tra lần 2 — nhóm — imputation-mmse-moca-adni-oasis

> **Cách đọc file này:** Phần A lấy từ PDF đề và CSV ma trận 10 bài. Phần B, C lấy từ các báo cáo mốc của nhóm (Mil_1 đến Mil_4). Mục **⚠️ [ĐIỀN]** là thông tin các file không có (link repo, checksum, ngày, vai trò…). Mục **🚩** là điểm bất nhất hoặc rủi ro mình phát hiện khi đọc, cần nhóm xem lại trước khi nộp.

- Fork: ⚠️ [ĐIỀN URL repo] · Nhánh: gd5-proposed-ablation (Mốc 3), gd6-medical-metrics-shap (Mốc 4) 🚩 tên nhánh theo lệnh git trong Mil_3_4; template ghi gd5-proposed-model, cần xác nhận nhánh nộp
- Thành viên (MSSV hoặc alias) và vai trò:
  - Nguyễn Anh Dũng — 20233740 — ⚠️ [ĐIỀN vai trò]
  - ⚠️ [ĐIỀN thành viên khác — MSSV — vai trò]

**Đề tài:** xử lý dữ liệu khuyết thiếu thang điểm nhận thức (MMSE/MoCA) trong sàng lọc sa sút trí tuệ; dữ liệu ADNI/OASIS; phân loại CN / MCI / AD (miền dữ liệu bảng/EHR).

---

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
Các truy vấn dưới đây là **gợi ý** dựa trên đề tài. Cột "Số kết quả" và "Số bài giữ lại" phải lấy từ lần tìm thực tế của nhóm.

| Nguồn (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|---|---|---|---|---|
| PubMed | ("missing data" OR imputation) AND (ADNI OR OASIS) AND (MCI OR Alzheimer) AND "machine learning" | 2020–2026 | ⚠️ [ĐIỀN] | ⚠️ [ĐIỀN] |
| Scopus | TITLE-ABS-KEY(imputation AND "cognitive scores" AND dementia classification) | 2020–2026 | ⚠️ [ĐIỀN] | ⚠️ [ĐIỀN] |
| IEEE Xplore | imputation AND Alzheimer AND "missing data" | 2020–2026 | ⚠️ [ĐIỀN] | ⚠️ [ĐIỀN] |
| Google Scholar | "data leakage" Alzheimer "subject-level" OR "patient-level" split | 2020–2026 | ⚠️ [ĐIỀN] | ⚠️ [ĐIỀN] |

Tiêu chí chọn / loại bài:
- Chọn: tạp chí Q1/Q2 SJR tại năm xuất bản; xuất bản 2020–2026; dùng ADNI/OASIS/AIBL hoặc dữ liệu bảng tương đương; liên quan điền khuyết, MNAR hoặc rò rỉ dữ liệu trong chẩn đoán sa sút trí tuệ.
- Chọn: có bài hội nghị top-tier nhưng tối đa 3/10 bài (hiện 0/10).
- Loại: arXiv preprint; bài không có PDF gốc để đối chiếu; bài không báo cáo kết quả định lượng.

### A2. Ma trận y văn (≥ 10 bài, ưu tiên 3–5 năm gần nhất)
Cột "Kiểm định ngoài?" được suy từ nội dung CSV. Cần đối chiếu lại với PDF gốc.

| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? (có / không) | Hạn chế chính |
|---|---|---|---|---|---|---|---|
| 1 | Chandrasekaran & Xie, 2024 [1] | 10.3233/JAD-231047 | ADNI, N=1.063 MCI; ADAS-Cog 13 dọc | LMM + MICE; chia/cụm theo bệnh nhân (RID) | CI hiệu ứng cố định hẹp hơn 22% so với complete-case (Table 2, p. 5); giữ power, không lệch chọn (Sec. 3.2, p. 6) | Không | Chỉ hồi quy tham số dọc; không so sánh phân loại phi tuyến đa lớp |
| 2 | Sarica et al., 2025 [2] | 10.3390/brainsci15060639 | ADNI, N=650; lâm sàng, nhận thức, hình ảnh | 5 cách điền khuyết (Mean, Median, kNN, MICE, missForest) × RF, LR, SVM; chia theo bệnh nhân, fit điền khuyết trên train | MICE+LR Acc=0,81; Median+SVM Acc=0,81; MICE+RF Acc=0,76 (Table 2, p. 6); McNemar p<0,01 (Sec. 3.3, p. 7) | Không (tập test độc lập nhỏ) | Tập test hoàn chỉnh nhỏ; chỉ visit cắt ngang |
| 3 | Wang et al., 2023 [3] | 10.3390/brainsci13071024 | ADNI, N=1.042 | EBM (glass-box GAM); chia theo bệnh nhân 80/20 + 5-fold CV | Dự báo chuyển đổi 5 năm: AUC=0,92, Acc=85% (Table 4, p. 8); hơn RF, XGBoost (Table 5, p. 9) | Không | Không có cohort ngoài ADNI; không đánh giá hệ thống pipeline MNAR |
| 4 | Silva et al., 2023 [4] | 10.1016/j.artmed.2023.102611 | ADNI, N=950; nhân khẩu, MMSE, CDR-SB, CSF, MRI | Pattern-mixture + điền khuyết nhận biết nhãn (MNAR) + RF, GBDT, SVM; 5-fold CV phân tầng theo bệnh nhân | MNAR nặng giảm AUC nền 15,4% (Table 3, p. 7); label-aware lấy lại +9,8% AUC (Fig. 4, p. 9) | Không | Khó chính thức hóa cơ chế MNAR nếu không xác nhận lâm sàng |
| 5 | McCombe et al., 2022 [5] | 10.1109/JBHI.2021.3098511 | ADNI, N=1.420; điểm nhận thức/chức năng; khuyết giả lập | Điền khuyết lặp + giảm đặc trưng + RF, GB; chia theo bệnh nhân | Điền khuyết lặp nhanh: AUC=0,895, nhanh hơn missForest 10 lần (Table 2, p. 6) | Không | Khuyết giả lập; thực tế phòng khám có thể phi ngẫu nhiên hơn |
| 6 | Gasmi et al., 2026 [6] | 10.3390/diagnostics16132029 | Dữ liệu bảng suy từ ADNI, N=1.737, 362 đặc trưng, 5 lớp | 16 ML + 11 DL; ensemble LightGBM + DNN (trọng số GA); FedNova; chia 1.390/347, fit điền khuyết trên train | LightGBM Acc=0,8156, kappa=0,7537 (Table 6, p. 11); Ensemble Acc=0,8213 (Table 7, p. 12); F1 lớp AD ≤ 0,21 (Table 8, p. 13) | Không | Mất cân bằng nặng, AD bị bỏ sót nhiều; dữ liệu Kaggle bên thứ ba |
| 7 | Nguyen et al., 2020 [7] | 10.1016/j.neuroimage.2020.117203 | ADNI/TADPOLE, N=1.677 dọc | minimalRNN + 3 cách xử lý khuyết (forward-fill, linear-fill, model-based); 10-fold CV theo bệnh nhân | mAUC đa lớp=0,931 (Table 2, p. 8); model-fill hơn baseline tĩnh 4,2% AUC (Table 3, p. 9) | Không | Cần ≥ 3 visit; trễ cao khi dự đoán ca chuyển đổi sớm |
| 8 | Shishegar et al., 2021 [8] | 10.1038/s41598-021-02827-6 | AIBL N=1.112 + ADNI N=1.245 | MissForest hài hòa hóa liên cohort (test thần kinh-tâm lý) | CVLT-II điền khuyết tương quan PET Aβ r=0,62, p<0,001 (Fig. 3, p. 5); giữ khả năng phân biệt chẩn đoán p<0,001 (Table 2, p. 6) | Một phần (2 cohort, không phải kiểm định phân loại) | Chỉ miền nhận thức liên tục; không phải bộ phân loại end-to-end |
| 9 | Wen et al., 2020 [9] | 10.1016/j.media.2020.101694 | ADNI, AIBL, OASIS; N>3.000 ảnh MRI | Tổng quan hệ thống pipeline CNN (2D slice, 3D patch, 3D subject) | >50% nghiên cứu DL có rủi ro rò rỉ nặng (Sec. 4.1, p. 12); mô hình rò rỉ báo Acc>95% (Table 4, p. 14) | N/A (tổng quan) | Tập trung ảnh thần kinh; cho thấy rò rỉ phổ biến trong AD |
| 10 | Yagis et al., 2021 [10] | 10.1038/s41598-021-01681-w | ADNI N=818; OASIS N=416; PPMI N=384 | 3 CNN 2D chuẩn, so sánh chia slice-level (rò rỉ) và subject-level (sạch) | Rò rỉ thổi phồng Acc +29,0% trên ADNI (94,2% so với 65,2%) (Table 2, p. 5); OASIS +30,1% (Table 2, p. 5) | Không (mỗi dataset đánh giá riêng) | Chỉ đo khoảng cách rò rỉ, không đề xuất bộ phân loại mới |

Kiểm tra điều kiện của PDF: 0/10 bài hội nghị (≤ 3), 0 bài arXiv, năm 2020–2026 (đạt).

### A3. Research gaps
| Mã | Khoảng trống (1–2 câu) | Bằng chứng (số # trong A2) | Dự án của nhóm giải quyết bằng cách nào |
|---|---|---|---|
| G1 | Rò rỉ dữ liệu giữa các visit của cùng một bệnh nhân làm kết quả lạc quan; nhiều nghiên cứu vẫn không chia theo định danh bệnh nhân. Khi chia sạch, độ chính xác giảm mạnh (94,2% xuống 65,2%). | #10 (Table 2, p. 5), #9 (Sec. 4.1, p. 12; Table 4, p. 14) | Chia theo RID bằng StratifiedGroupKFold; assertion kiểm tra tập RID rời nhau; fit imputer/scaler/encoder chỉ trên Train |
| G2 | Phương pháp điền khuyết thường được đánh giá bằng tái tạo hoặc thống kê hồi quy, hiếm khi đo tác động lên phân loại đa lớp CN/MCI/AD trên tập test độc lập. | #1 (Table 2, p. 5), #8 (Fig. 3, p. 5; Table 2, p. 6), #5 (Table 2, p. 6) | So sánh có kiểm soát: LR/RF/XGBoost × Mean/Median/KNN/MICE, cùng một split theo bệnh nhân, báo cáo metric phân loại |
| G3 | Cơ chế khuyết phụ thuộc nhãn (MNAR) ít được xét; khi MNAR nặng, AUC tụt đáng kể. Khuyết giả lập (MCAR) chưa phản ánh thực tế phòng khám. | #4 (Table 3, p. 7; Fig. 4, p. 9), #5 (khuyết giả lập) | Phân tích mẫu hình khuyết MMSE/MoCA theo nhãn; thêm cột chỉ báo khuyết (missing indicator) để tránh "false normalcy"; so sánh có/không có chỉ báo |
| G4 | Mất cân bằng lớp làm lớp AD bị bỏ sót nhiều, trong khi nhiều bài chỉ báo Accuracy. | #6 (Table 8, p. 13: F1 AD ≤ 0,21), #2 (chỉ Acc) | Báo cáo Recall từng lớp MCI/AD, macro-F1, PR-AUC; báo cáo imbalance ratio cấp mẫu và cấp bệnh nhân |
| G5 | Thiếu kiểm định ngoài: phần lớn đánh giá chỉ trên ADNI, tập test nhỏ. | #3 (hạn chế), #2 (tập test độc lập nhỏ), #6 (dữ liệu bên thứ ba) | Dùng OASIS làm tập kiểm định ngoài nếu ánh xạ được biến; nếu không, nêu rõ là giới hạn |

### A4. Thẩm định 1 bài quan trọng nhất trong A2 bằng PROBAST+AI
Bài số #: **2 (Sarica et al., 2025)** · Lý do chọn: gần nhất với dự án (so sánh Mean/Median/kNN/MICE × LR/RF/SVM trên ADNI, test tách theo bệnh nhân, imputation fit trên train).

⚠️ Bảng dưới chỉ dựa trên thông tin trong ma trận (CSV). Cột "Căn cứ" cần được thay bằng số trang thật sau khi nhóm đọc PDF.

| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ (trích dẫn hoặc số trang trong bài) |
|---|---|---|
| Người tham gia và nguồn dữ liệu | Không rõ | ADNI N=650, nhiều phương thức; tiêu chí chọn mẫu hoàn chỉnh có thể làm lệch quần thể. ⚠️ [ĐIỀN trang] |
| Biến dự báo | Không rõ | Dùng biến lâm sàng, nhận thức, hình ảnh; chưa rõ biến nào có tại T0 hoặc đã nằm trong tiêu chí chẩn đoán. ⚠️ [ĐIỀN trang] |
| Kết cục | Không rõ | Nhãn chẩn đoán ADNI; nguy cơ incorporation bias nếu MMSE/CDR là biến dự báo. ⚠️ [ĐIỀN trang] |
| Phân tích | Thấp đến không rõ | Held-out test, fit điền khuyết trên train (Sec. 3.3, p. 7); tập test độc lập nhỏ; chỉ báo Acc, thiếu calibration. ⚠️ [ĐIỀN trang] |

Tính áp dụng cho bài toán của nhóm: **Cao về phương pháp, trung bình về quần thể.** Cùng dữ liệu ADNI và cùng họ phương pháp điền khuyết. Khác ở chỗ bài chỉ đo Accuracy và dùng visit cắt ngang, còn nhóm ưu tiên Recall MCI/AD, macro-F1, PR-AUC và chia theo RID.

---

## B. Kết quả và timeline

### B1. Kết quả hiện có
Dữ liệu: ADNI baseline, N=3.485 bệnh nhân (CN 1.523 · MCI 1.432 · AD 530). Chia phân tầng theo RID, seed 42: Train 2.439 (CN 1.066 / MCI 1.002 / AD 371), Val 523 (228 / 215 / 80), Test 523 (229 / 215 / 79, khóa kín đến Mốc 4). 17 đặc trưng (5 nhân khẩu, 7 nhận thức, 4 MRI, 1 APOE); đã loại CDRSB, CDGLOBAL, MMSCORE để tránh rò rỉ nhãn.

**Bảng 1a. Tập Validation (n=523)**

| Mô hình | Tập đánh giá (cách chia, n) | Metric chính (giá trị, 95% CI nếu có) | Metric phụ | Nguồn số liệu |
|---|---|---|---|---|
| Baseline 1 (Mốc 2): Logistic Regression, class_weight=balanced | Val, chia theo RID, n=523 | Macro F1 0,7868; ROC-AUC OvR 0,9147; chưa có CI | Sens/Spec: CN 82,5/87,5; MCI 70,7/84,4; AD 88,8/93,9 (%) | Mil_2.pdf §5.1 |
| Baseline 2 (Mốc 2): Random Forest | Val, n=523 | Macro F1 0,7614; ROC-AUC 0,9083; chưa có CI | Sens/Spec: CN 82,9/85,1; MCI 68,4/82,5; AD 80,0/94,4 (%) | Mil_2.pdf §5.1 |
| Mốc 3: Logistic Regression | Val, n=523 | Macro F1 0,7833; ROC-AUC 0,9163 | Balanced Acc 0,8084; Sens AD 90,0%; Sens MCI 67,4% | Mil_3.pdf §3 |
| Mốc 3: Random Forest | Val, n=523 | Macro F1 0,7591; ROC-AUC 0,9107 | Balanced Acc 0,7750; Sens AD 81,3%; Sens MCI 67,9% | Mil_3.pdf §3 |
| Mốc 3: LightGBM + missForest | Val, n=523 | Macro F1 0,7627; ROC-AUC 0,9179 | Balanced Acc 0,7803; Sens AD 82,5%; Sens MCI 66,5% | Mil_3.pdf §3 |
| Mốc 3: XGBoost | Val, n=523 | Macro F1 0,7641; ROC-AUC 0,9198 | Balanced Acc 0,7596; Sens AD 72,5%; Sens MCI 71,2% | Mil_3.pdf §3 |
| Mốc 3: Voting Ensemble | Val, n=523 | Macro F1 0,7624; ROC-AUC 0,9189 | Balanced Acc 0,7738; Sens AD 80,0%; Sens MCI 68,4% | Mil_3.pdf §3 |

**Bảng 1b. Tập Test độc lập (n=523; CN 229, MCI 215, AD 79), Mốc 4**
CI là bootstrap 1.000 lần. Ô "—" là báo cáo không có CI.

| Mô hình | Macro F1 (95% CI) | ROC-AUC OvR (95% CI) | Brier (95% CI) | MCC | Sens AD (95% CI) | Sens MCI |
|---|---|---|---|---|---|---|
| Logistic Regression | 0,7816 [0,7448–0,8187] | 0,9283 — | 0,0991 — | 0,6601 | 87,34% [79,75–94,19] | 67,44% |
| Random Forest | 0,8042 [0,7649–0,8393] | 0,9245 — | 0,0990 — | 0,6836 | 86,08% [78,31–93,24] | 71,16% |
| LightGBM | 0,8023 [0,7662–0,8365] | 0,9311 [0,9132–0,9483] | 0,0942 — | 0,6774 | 87,34% [79,75–94,05] | 72,56% |
| XGBoost | 0,8182 [0,7820–0,8514] | 0,9358 [0,9178–0,9519] | 0,0888 [0,0770–0,1012] | 0,7042 | 79,75% — | 76,74% |
| Voting Ensemble | 0,8127 [0,7741–0,8467] | 0,9336 — | 0,0917 — | 0,6989 | 86,08% [78,31–93,24] | 72,09% |

Nguồn: Mil_4.pdf §1–2 (cũng có trong Mil_3_4.pdf). Balanced Acc của Voting Ensemble là 0,8213 [0,7840–0,8561].

🚩 **Cách đọc kết quả:** CI của 5 mô hình chồng lấn nhau gần hoàn toàn (nửa độ rộng CI Macro F1 khoảng 0,035), nên không có cơ sở thống kê để xếp hạng XGBoost cao hơn các mô hình còn lại. Trên Val, Logistic Regression lại có Macro F1 cao nhất (0,7833). Cần chọn mô hình chính **dựa trên Val trước khi xem Test** rồi nêu rõ trong báo cáo; nếu chọn XGBoost vì Test cao nhất thì đó là chọn mô hình trên tập test.

### B2. Ablation (Mốc 3)
Mô hình đầy đủ để so sánh: LightGBM + missForest + trọng số cân bằng + 17 đặc trưng, Val n=523, Macro F1 = 0,7627, ROC-AUC = 0,9179, Sens AD = 82,5%.

| Cấu hình (bỏ / đổi thành phần nào) | Metric chính | Chênh lệch so với mô hình đầy đủ | Nguồn |
|---|---|---|---|
| Imputation: Mean | Macro F1 0,7756; AUC 0,9207 | F1 +0,0129 | Mil_3 §2.1 |
| Imputation: KNN-5 | Macro F1 0,7684; AUC 0,9133 | F1 +0,0057 | Mil_3 §2.1 |
| Imputation: MICE (Bayesian Ridge) | Macro F1 0,7832; AUC 0,9216 | F1 +0,0205 | Mil_3 §2.1 |
| Imputation: Native GBDT NaN routing | Macro F1 0,7673; AUC 0,9225 | F1 +0,0046 | Mil_3 §2.1 |
| Complete-case (bỏ hàng khuyết) | Macro F1 0,7993; AUC 0,9464 (Train 1.190 / Val 259) | F1 +0,0366, **không so sánh trực tiếp** vì Val khác (259 so với 523), mất 51,2% mẫu | Mil_3 §2.1 |
| Chỉ nhân khẩu học (5 biến) | Macro F1 0,4578; AUC 0,6243; Sens AD 51,2% | F1 −0,3049 | Mil_3 §2.2 |
| Nhân khẩu + nhận thức (12 biến) | Macro F1 0,7601; AUC 0,9168 | F1 −0,0026 | Mil_3 §2.2 |
| + APOE (13 biến) | Macro F1 0,7621; AUC 0,9185 | F1 −0,0006 | Mil_3 §2.2 |
| Thêm cờ protocol indicator | Macro F1 0,7618; AUC 0,9188 | F1 −0,0009; AUC +0,0009 | Mil_3 §2.3 |
| Trọng số đồng đều (không class_weight) | Macro F1 0,7682; Sens AD 76,3%; Precision AD 76,3% | F1 +0,0055; Sens AD −6,2 điểm % | Mil_3 §2.4 |
| Thêm MMSE/CDR vào đặc trưng (kiểm tra rò rỉ nhãn, Logistic Regression) | Macro F1 0,8645; AUC 0,9721 | so với LR Main 0,7868 / 0,9147: F1 +0,0777; AUC +0,0574 | Mil_2 §5.1–5.2 |

🚩 **Nhận xét:** Các chênh lệch giữa các cách điền khuyết (0,005–0,02) và việc thêm APOE, MRI, protocol indicator (dưới 0,003) nhỏ hơn nhiều so với độ rộng CI ở Test (khoảng ±0,035). Không nên kết luận "cải thiện" từ các số này nếu chưa có CI hoặc cross-validation. Hai ablation có tác động rõ rệt là bỏ toàn bộ khối nhận thức (F1 giảm 0,30) và thêm MMSE/CDR (F1 tăng 0,08).

### B3. Đối chiếu với y văn
Accuracy dưới đây mình **tự tính** từ ma trận nhầm lẫn Test (Mil_4 Hình 8, đọc từ hình), nhóm cần xác nhận lại với notebook: Logistic Regression ≈ 0,786; Random Forest ≈ 0,803; LightGBM ≈ 0,799; XGBoost ≈ 0,818; Ensemble ≈ 0,813. F1 lớp AD (tính từ Precision và Recall ở Bảng 1b): LightGBM ≈ 0,82; XGBoost ≈ 0,82.

| So với | Số liệu bài báo | Số liệu của nhóm | Điểm khiến phép so sánh **không công bằng** |
|---|---|---|---|
| [6] Gasmi 2026 | Acc 0,8156 (LightGBM); Ensemble 0,8213; F1 lớp AD ≤ 0,21 (Table 6–8) | LightGBM Acc ≈ 0,80; F1 lớp AD ≈ 0,82 | Bài có 5 lớp, 362 đặc trưng, dữ liệu Kaggle dẫn xuất từ ADNI; nhóm có 3 lớp, 17 đặc trưng. Chia cố định 1.390/347 khác chia 70/15/15 theo RID. Chỉ nên so F1 lớp AD, không so Accuracy tổng. |
| [2] Sarica 2025 | MICE+LR Acc 0,81; MICE+RF 0,76 (Table 2) | LR Acc ≈ 0,79 (Test); MICE chỉ đánh giá với LightGBM trên Val | Bài dùng N=650 và tập test hoàn chỉnh nhỏ; nhóm N=3.485. Nhóm chưa chạy lưới LR/RF × từng cách điền khuyết. Bài báo cáo Accuracy, nhóm ưu tiên Recall và Macro F1. |
| [10] Yagis 2021, [9] Wen 2020 | Rò rỉ làm Acc tăng +29% (94,2% so với 65,2%) | Chia theo RID, Macro F1 0,78–0,82 | Hai bài dùng ảnh MRI/CNN, không phải dữ liệu bảng. Kết quả vừa phải của nhóm phù hợp với chia sạch. Việc thêm MMSE/CDR làm AUC tăng 0,915 lên 0,972 là rò rỉ nhãn (khác loại rò rỉ với [10]). |
| [4] Silva 2023 | MNAR nặng làm giảm AUC 15,4%; label-aware lấy lại +9,8% (Table 3, Fig. 4) | Chưa chạy thí nghiệm MNAR | EDA cho thấy tỷ lệ khuyết MoCA phụ thuộc nhãn (CN 40,6% / MCI 64,6% / AD 78,7%). Nhóm có protocol indicator nhưng chưa có thí nghiệm mô phỏng MNAR. |

### B4. Timeline đến khi hoàn thành
Người phụ trách và ngày do nhóm điền.

| Việc | Phụ trách (MSSV / vai trò) | Mốc (M3 tuần 10 · M4 tuần 12 · Final tuần 15) | Ngày dự kiến | Trạng thái (Chưa / Đang / Xong + bằng chứng) |
|---|---|---|---|---|
| EDA: phân bố nhãn, nhân khẩu học, tỷ lệ khuyết, ma trận tương quan | ⚠️ | Mốc 1 | ⚠️ | Xong — Mil_1.pdf (notebook công khai: ⚠️ [ĐIỀN link]) |
| Chia theo RID, Pipeline tiền xử lý, Baseline LR/RF | ⚠️ | Mốc 2 | ⚠️ | Xong — Mil_2.pdf |
| Mô hình đề xuất + 4 ablation + TensorBoard | ⚠️ | M3 (tuần 10) | ⚠️ | Xong — Mil_3.pdf, nhánh gd5-proposed-ablation |
| Mở khóa Test, đa độ đo, bootstrap CI, SHAP, 4 ca lâm sàng | ⚠️ | M4 (tuần 12) | ⚠️ | Xong — Mil_4.pdf, nhánh gd6-medical-metrics-shap |
| Hiệu chuẩn: đường cong calibration, in-the-large, slope, Brier có CI cho mọi mô hình | ⚠️ | Final (tuần 15) | ⚠️ | Chưa |
| Chạy lưới LR/RF/XGBoost × Mean/Median/KNN/MICE (yêu cầu baseline của đề) | ⚠️ | Final | ⚠️ | Chưa (mới có LightGBM × imputation) |
| Sửa các số liệu bất nhất (xem B5) và chốt mô hình chính theo Val | ⚠️ | Final | ⚠️ | Chưa |
| Kiểm định ngoài (OASIS) hoặc nêu rõ là giới hạn | ⚠️ | Final | ⚠️ | Chưa |
| Datasheet, checksum dữ liệu, checklist TRIPOD+AI | ⚠️ | Final | ⚠️ | Chưa |
| Xác minh ma trận 10 bài với PDF gốc, sửa DOI/link (mục A, B5) | ⚠️ | Final | ⚠️ | Đang |
| Bài báo IEEE, dọn repo, chuẩn bị vấn đáp | ⚠️ | Final | ⚠️ | Chưa |

### B5. Rủi ro / đang vướng / câu hỏi cho GV
**Bất nhất số liệu giữa các file mốc (cần sửa trước khi nộp):**
- 🚩 Mil_3 Hình 2 (Ablation 2) không khớp bảng §2.2: ví dụ cấu hình "Chỉ nhân khẩu học" bảng ghi Macro F1 0,4578 và Sens AD 51,2%, hình ghi 0,430 và 0,412. Cấu hình "Nhân khẩu + nhận thức" bảng ghi 0,7601 còn hình ghi 0,797.
- 🚩 Cấu hình đầy đủ (LightGBM, 17 biến) ghi ROC-AUC 0,9224 ở §2.2 nhưng 0,9179 ở §2.3, §2.4 và §3.
- 🚩 Logistic Regression trên Val: Mốc 2 ghi Macro F1 0,7868 / AUC 0,9147; Mốc 3 ghi 0,7833 / 0,9163. Cần nêu rõ cấu hình khác nhau ở đâu.
- 🚩 Chú thích Hình 7 ghi Δ "+0,04 đến +0,05", nhưng Logistic Regression có Δ = −0,002.

**Kết luận chưa đủ căn cứ:**
- 🚩 "Test cao hơn Val 0,04–0,05 khẳng định không rò rỉ" (Mil_4 §3; Slide 8): chênh lệch này không chứng minh được điều đó và nằm trong khoảng nhiễu của CI. Nên bỏ hoặc viết lại, và dẫn chứng rò rỉ bằng assertion RID rời nhau và Pipeline fit trên Train.
- 🚩 Slide 9 nói xác suất "có độ tin cậy chuẩn mực y tế" nhưng chỉ báo Brier; chưa có đường cong hiệu chuẩn, calibration-in-the-large, slope. Thiếu CI cho phần lớn chỉ số (Bảng 1b).
- 🚩 Phát biểu "protocol indicator cải thiện AUC" (Mil_3 §2.3): chênh lệch +0,0009 là nhiễu. Cần nêu thêm rủi ro: ADNIGO gần như 99% MCI nên cờ protocol có thể trở thành "đường tắt" học tỷ lệ nhãn theo giai đoạn nghiên cứu.
- 🚩 Complete-case có Macro F1 cao nhất (0,7993) nhưng Val chỉ 259 ca và bỏ 51,2% mẫu, không so sánh được với các cách còn lại (báo cáo đã cảnh báo, giữ nguyên).
- 🚩 missForest là cách điền khuyết trong pipeline đề xuất nhưng có Macro F1 thấp nhất trong nhóm điền đủ mẫu (0,7627; MICE 0,7832). Cần giải thích lý do chọn hoặc đổi mô hình chính.

**Phạm vi và thiết kế:**
- 🚩 Câu hỏi nghiên cứu ban đầu nói về điền khuyết MMSE/MoCA, nhưng MMSE bị loại (rò rỉ nhãn) và không thấy MoCA trong 17 đặc trưng hay SHAP. Dữ liệu khuyết thực sự được xử lý là ADAS, MRI, APOE. Cần chỉnh câu hỏi nghiên cứu cho khớp.
- 🚩 Rò rỉ nhãn chưa loại hết: Composite Memory, FAQ, ADAS-Cog là 3 đặc trưng quan trọng nhất theo SHAP, mà các thang này gắn chặt với tiêu chí chẩn đoán MCI/AD của ADNI. Cần thêm ablation bỏ các biến này hoặc nêu rõ là giới hạn.
- Chỉ 1 lần chia hold-out và chỉ ADNI; chưa có kiểm định ngoài. Test chỉ có 79 ca AD nên CI độ nhạy AD rộng (khoảng 79–94%).
- Biến phân loại (race, ethnicity) mã hóa OrdinalEncoder, trong khi PDF đề khuyến nghị One-Hot cho biến không thứ bậc. Race/ethnicity đang là biến dự báo nhưng chưa có phân tích công bằng theo nhóm.
- Báo cáo Mốc 4 công bố RID của 4 ca lâm sàng (1368, 301, 10151, 4502). Đây là mã giả danh của ADNI, nhưng nên kiểm tra DUA trước khi đưa vào báo cáo công khai.

**Xác minh nguồn (phần A):** các lưu ý về DOI/link bài [1], link Sci-Hub bài [4], SJR bài [1] ở mục A vẫn còn hiệu lực.

**Câu hỏi cho GV:** (1) Có cần đổi câu hỏi nghiên cứu từ MMSE/MoCA sang "thang nhận thức và MRI/APOE" không? (2) Chấp nhận ADNI-only với phần giới hạn, hay phải làm OASIS? (3) Có cần calibration cho từng lớp (OvR) hay chỉ Brier đa lớp?

---

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu
| Mục | Trả lời | Bằng chứng |
|---|---|---|
| Nguồn, phiên bản, ngày tải; checksum file dùng để huấn luyện | Nguồn: ADNI baseline, N=3.485 (nhiều giai đoạn ADNI1, GO, 2, 3, 4). Phiên bản, ngày tải, SHA-256: ⚠️ [ĐIỀN] 🚩 báo cáo không nêu | ⚠️ [file / lệnh sha256sum] |
| Giấy phép / DUA, căn cứ được phép dùng | ADNI yêu cầu nộp đơn và ký DUA (theo PDF đề). Ngày duyệt, mã đơn: ⚠️ [ĐIỀN] | ⚠️ [email duyệt] |
| Datasheet: quần thể, tiêu chí chọn/loại, định nghĩa và đơn vị biến, giới hạn | **Quần thể:** người cao tuổi trong cohort ADNI; CN 1.523 (43,7%), MCI 1.432 (41,1%), AD 530 (15,2%); IR 2,9:1 / 2,7:1 / 1,0:1; entropy nhãn 1,462 bit (tối đa 1,585). Mỗi RID một dòng baseline nên cấp mẫu và cấp bệnh nhân trùng nhau ⚠️ [xác nhận]. **Tuổi trung bình:** CN 70,6; MCI 72,5; AD 74,8. **Học vấn:** CN 16,50, AD 15,23 năm. **Giới:** nữ 61,2% ở CN, 44,1% ở MCI, 44,7% ở AD. **Khuyết:** MRI 42,3%, MoCA 56,3%, Digit Span 77,6%, APOE 21,8%, ADAS 19,3%, MMSE 0,4%. **Giới hạn:** tỷ lệ nhãn phụ thuộc giai đoạn ADNI (ADNIGO ≈ 99% MCI; ADNI3/4 CN chiếm 54,5% / 62,4%); cohort nghiên cứu, không đại diện dân số chung; phân bố chủng tộc chưa báo cáo. Định nghĩa và đơn vị từng biến: ⚠️ [ĐIỀN datasheet] | Mil_1.pdf §1–3; ⚠️ [datasheet.md] |
| Không có định danh bệnh nhân | RID là mã giả danh của ADNI, dùng làm khóa chia nhóm. RID của 4 ca lâm sàng xuất hiện trong báo cáo (Mil_4 §5). ⚠️ [xác nhận không có định danh nào khác] | Mil_4.pdf §5 |

### C2. Hướng dẫn báo cáo
- Câu hỏi nghiên cứu của nhóm: Trong phân loại CN/MCI/AD trên dữ liệu bảng ADNI có khuyết (ADAS, MRI, APOE), chiến lược xử lý khuyết nào (Complete-case, Mean, KNN, MICE, missForest, Native GBDT) cho Macro F1 và Recall MCI/AD tốt nhất trên tập test chia theo bệnh nhân? 🚩 Cần thống nhất với đề tài ban đầu (MMSE/MoCA), xem B5.
- Hướng dẫn báo cáo phù hợp: **TRIPOD+AI**, vì đây là phát triển và đánh giá mô hình dự báo/chẩn đoán đa biến. STARD-AI hợp hơn cho đánh giá độ chính xác của một công cụ đã cố định; DECIDE-AI dành cho đánh giá lâm sàng giai đoạn sớm. ⚠️ [đối chiếu Phụ lục §3.2]
- Số mục checklist đã đáp ứng / tổng số mục; các mục còn thiếu: ⚠️ [ĐIỀN x/y]. Từ các báo cáo mốc, mình thấy còn thiếu: hiệu chuẩn, cỡ mẫu, kiểm định ngoài, nguồn và phiên bản dữ liệu, phân tích công bằng theo nhóm.
- Kết cục, nhãn, thời điểm dự đoán T0, các biến có sẵn tại T0: **Kết cục** = chẩn đoán CN / MCI / AD (lớp ADNI) tại visit baseline. **T0** = visit baseline (chẩn đoán cắt ngang, cùng thời điểm đo các biến). **Biến tại T0** (17): tuổi, giới, học vấn, chủng tộc, dân tộc; Composite Memory, Language, Executive, Visuospatial, FAQ, ADAS-Cog-11, ADAS-Cog-13; 4 chỉ số MRI (hồi hải mã, não thất, vỏ nội khứu, tỷ lệ thể tích não/TIV); số alen APOE ε4. **Đã loại:** CDRSB, CDGLOBAL, MMSCORE.

### C3. PROBAST+AI — tự thẩm định dự án của nhóm
Căn cứ dựa vào nội dung báo cáo mốc; tên file mình lấy từ lệnh git, số dòng nhóm điền.

| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ trong repo (file, dòng) | Việc sẽ làm để giảm nguy cơ |
|---|---|---|---|
| Người tham gia và nguồn dữ liệu | **Không rõ, nghiêng về cao.** Tỷ lệ nhãn thay đổi mạnh theo giai đoạn ADNI (ADNIGO ≈ 99% MCI; ADNI3/4 nhiều CN), AD chiếm 15,2%; tình nguyện viên nghiên cứu | Mil_1 §1.2; ⚠️ [file:dòng] | Báo cáo kết quả theo từng giai đoạn ADNI; nêu rõ tỷ lệ AD không phản ánh sàng lọc cộng đồng; kiểm định ngoài (OASIS) nếu ánh xạ được biến |
| Biến dự báo | **Không rõ.** Đã loại MMSE và CDR; còn dùng Race/Ethnicity; MRI khuyết 42,3%; cờ protocol indicator có thể mã hóa giai đoạn nghiên cứu | Mil_2 Slide 3; Mil_3 §2.3; ⚠️ [file:dòng] | Ablation bỏ Race/Ethnicity; phân tích công bằng theo giới/chủng tộc; xem xét bỏ protocol indicator |
| Kết cục (chú ý incorporation bias) | **Cao (tạm thời).** MMSE/CDR đã loại, nhưng Composite Memory, FAQ, ADAS-Cog là 3 biến quan trọng nhất theo SHAP và gắn chặt với tiêu chí chẩn đoán. Khi thêm MMSE/CDR, AUC tăng từ 0,915 lên 0,972 | Mil_2 §5.2; Mil_4 §4 (Hình 11); ⚠️ [file:dòng] | Thêm ablation bỏ FAQ, ADAS, Composite Memory; nêu rõ nhãn lấy từ đâu và tiêu chí chẩn đoán dùng biến nào |
| Phân tích (chia theo bệnh nhân, fit chỉ trên train, cỡ mẫu, dữ liệu thiếu) | **Không rõ** (thấp về rò rỉ; cao ở khâu hiệu chuẩn). Chia phân tầng theo RID; Pipeline fit trên Train; thiếu calibration; một lần chia hold-out; AD test chỉ n=79; complete-case so sánh trên Val khác | Mil_2 Slide 4; Mil_3 §1 (code Pipeline); `src/m3_models.py`, `src/m4_evaluation.py` ⚠️ [dòng] | Đính kèm kết quả assertion RID rời nhau; thêm calibration (đường cong, CITL, slope) và CI cho mọi chỉ số; cân nhắc StratifiedGroupKFold; chốt mô hình chính theo Val |
Tính áp dụng (quần thể, bối cảnh dùng mô hình): **thấp đến trung bình.** Mô hình học trên cohort nghiên cứu ADNI (tiêu chí tuyển chọn theo từng giai đoạn) và dùng 17 đặc trưng gồm MRI và ADAS-Cog, không phải bộ sàng lọc đơn giản. Phù hợp hơn với phòng khám trí nhớ có MRI. Cần kiểm định ngoài trước khi dùng cho sàng lọc cộng đồng.

### C4. Bảng kiểm nhanh (Phụ lục §7)
| Mục | Đạt / Chưa | Bằng chứng |
|---|---|---|
| Datasheet | Chưa | Có số liệu EDA (Mil_1) nhưng chưa có file datasheet |
| Phiên bản + checksum dữ liệu | Chưa | Không thấy trong các báo cáo ⚠️ |
| Không định danh; căn cứ được phép dùng dữ liệu | Chưa (một phần) | RID giả danh; DUA ⚠️ [ĐIỀN] |
| Chọn đúng hướng dẫn báo cáo + checklist đã điền | Chưa | TRIPOD+AI đã chọn (C2), checklist chưa điền |
| Định nghĩa kết cục, nhãn, T0 | Đạt (một phần) | C2; còn rủi ro incorporation bias (C3) |
| Chia theo bệnh nhân; tiền xử lý fit chỉ trên train | Đạt | Mil_2 Slide 3–4; code Pipeline (Mil_3 §1); ⚠️ [đính kèm output assertion] |
| Mô hình mốc hồi quy logistic (hoặc mốc phù hợp, nêu lý do) | Đạt | Mil_2 §5.1 (Macro F1 0,7868; AUC 0,9147) |
| AUC + calibration (đường cong, in-the-large, slope) + Brier, kèm 95% CI | Chưa | Có AUC và Brier, CI chỉ cho một số chỉ số; thiếu đường cong, in-the-large, slope (Mil_4 §1–2) |
| Cỡ mẫu có đủ không; giới hạn về tính áp dụng | Chưa (một phần) | Test n=523 (AD 79); chưa tính cỡ mẫu; ADNI-only, chưa kiểm định ngoài |

---

## D. Khai báo dùng AI
| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
|---|---|---|
| Claude (Anthropic) | Soạn bản nháp file này từ PDF đề, CSV ma trận 10 bài và các báo cáo Mil_1–Mil_4; tính một số chỉ số phụ (Accuracy, F1 lớp AD, chênh lệch ablation) | ⚠️ [ĐIỀN] Gợi ý: đối chiếu từng con số với notebook và PDF gốc; kiểm tra các số "tự tính" ở B3 và B2 |
| ⚠️ [ĐIỀN công cụ AI khác nếu có, ví dụ dùng sinh code hoặc báo cáo mốc] | ⚠️ | ⚠️ |
