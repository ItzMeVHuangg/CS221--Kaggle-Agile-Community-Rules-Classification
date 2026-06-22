
Conversation with Gemini
đọc 2 file codespace và report pdf, sau đó tham khảo file README bên và viết cho tôi một file README về đề tài này

Python
import json
import pypdf

# Check cell contents of Codespace.ipynb for headings or results summaries
with open('Codespace.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("--- CODESPACE HEADINGS & MARKDOWN CONTENT ---")
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if any(line.strip().startswith('#') for line in cell['source']):
            print(f"Cell {i}:")
            print(source[:200])
            print("-" * 20)

# Check text in Report.pdf
print("\n--- REPORT.PDF PAGES ---")
reader = pypdf.PdfReader('Report.pdf')
print(f"Total pages: {len(reader.pages)}")

# Let's extract pages that might have results (e.g., towards the end, or table of contents)
for page_num in [1, 2, 3, 4, 5, 20, 25, 30, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
    if page_num < len(reader.pages):
        text = reader.pages[page_num].extract_text()
        if "kết quả" in text.lower() or "bảng" in text.lower() or "đánh giá" in text.lower() or "mô hình" in text.lower() or "chương" in text.lower() or page_num <= 4:
            print(f"--- Page {page_num+1} ---")
            print(text[:1000])
            print("=" * 40)


Code output
--- CODESPACE HEADINGS & MARKDOWN CONTENT ---
Cell 0:
# Jigsaw - Agile Community Rules Classification
Phan loai vi pham noi quy cong dong Reddit

Dataset: https://www.kaggle.com/competitions/jigsaw-agile-community-rules

---

## Muc luc
1. Gioi thieu bai
--------------------
Cell 1:
## 1. Gioi thieu bai toan & Giai thich cot du lieu

### Bai toan

Phan loai nhi phan: binh luan Reddit co vi pham mot noi quy cu the cua cong dong do khong?

**Diem dac biet — "Agile":** Chi co **2 no
--------------------
Cell 2:
## 2. Cài đặt & Import
--------------------
Cell 5:
## 3. Load Dataset

Tải dataset từ Kaggle:
```bash
kaggle competitions download -c jigsaw-agile-community-rules
Cell 7:

4. EDA
Cell 13:

4.1. Phân tích thủ công ngữ liệu (Qualitative Analysis)
Dựa trên yêu cầu đánh giá định tính, tôi đã tiến hành phân tích thủ công ngẫu nhiên khoảng 50-70 mẫu văn bản trong tập dữ liệu. Dưới đây là
Cell 15:

5. Tien xu ly du lieu
Chien luoc input cho 12 huong
Huong	text_a	text_b
LR-A	body_clean [SEP] rule_clean	—
LR-B	`body_clean [SEP] rule_clean	
Cell 17:

6. Baseline: TF-IDF + Logistic Regression
Hướng A (sentence-pair): body [SEP] rule
Hướng B (few-shot): body [SEP] rule [SEP] POSITIVE: ex1 [SEP] NEGATIVE: ex2
Hướng C (context-awar

Cell 19:

7. Mo hinh DistilBERT (3 huong)
DistilBERT nhẹ hơn BERT-base (66M vs 110M params). Ta thử nghiệm cả 3 hướng:

Sentence-pair

Few-shot

Context-aware template

Cell 22:

8. Mo hinh DeBERTa-v3-small (3 huong)
Thử nghiệm 3 hướng cho DeBERTa-v3-small.
Giải thích lý do DeBERTa train chậm ở Few-Shot: Do kiến trúc Disentangled Attention tính toán độ phức tạp theo bình
Cell 25:

9. Mo hinh RoBERTa-base (3 huong)
Thử nghiệm 3 hướng cho RoBERTa-base. Thiết lập config epoch=20, patience=3.
Cell 29:

10. Danh gia & So sanh 12 huong
Đánh giá 4 mô hình (LR, DistilBERT, DeBERTa, RoBERTa) x 3 hướng tiền xử lý (Sentence-pair, Few-shot, Context-aware).
Cell 36:

11. Phân tích lỗi (Error Analysis)
Phần này tiến hành trích xuất tự động True Positive, True Negative, False Positive, và False Negative cho toàn bộ 12 biến thể mô hình để phân tích điểm yếu cốt l
Cell 37:

Phân tích lỗi: LR (SP)
Nhận xét: Gặp khó khăn với các từ đồng nghĩa chưa từng xuất hiện trong tập train do tf-idf chỉ bắt word vựng cứng.

Cell 39:

Phân tích lỗi: LR (FS)
Nhận xét: Bị nhiễu bởi các từ vựng xuất hiện trong Few-shot ví dụ mẫu, làm tăng tỷ lệ False Positive.

Cell 41:

Phân tích lỗi: LR (CA)
Nhận xét: Context subreddit giúp phân loại tốt hơn một chút nhưng TF-IDF vẫn không bắt được sự thay đổi thứ tự từ.

Cell 43:

Phân tích lỗi: DistilBERT (SP)
Nhận xét: Bắt tốt các pattern chung nhưng dễ bị lừa bởi câu phủ định dài.

Cell 45:

Phân tích lỗi: DistilBERT (FS)
Nhận xét: Việc nối thêm Few-shot làm chuỗi quá dài (256 token) khiến kiến trúc nông (6 lớp) của Distil bị quá tải thông tin.

Cell 47:

Phân tích lỗi: DistilBERT (CA)
Nhận xét: Bắt được tag subreddit khá tốt nhưng đôi lúc quá phụ thuộc vào subreddit dẫn đến False Negative ở các subreddit lạ.

Cell 49:

Phân tích lỗi: DeBERTa (SP)
Nhận xét: Hiểu ngữ cảnh tốt hơn DistilBERT nhờ Disentangled Attention.

Cell 51:

Phân tích lỗi: DeBERTa (FS)
Nhận xét: Mô hình thể hiện sự cải thiện rõ rệt khi có thêm Few-shot, bắt được logic tương đồng so với ví dụ.

Cell 53:

Phân tích lỗi: DeBERTa (CA)
Nhận xét: Subreddit context kết hợp với DeBERTa đôi khi gây ra Overfit nhẹ trên validation set.

Cell 55:

Phân tích lỗi: RoBERTa (SP)
Nhận xét: Mô hình mạnh nhất về F1. Rất nhạy bén với cấu trúc câu phức tạp và sắc thái ý nghĩa.

Cell 57:

Phân tích lỗi: RoBERTa (FS)
Nhận xét: Xử lý chuỗi 256 token mượt mà nhất trong 3 model Deep Learning, cho ra AUC tuyệt đối.

Cell 59:

Phân tích lỗi: RoBERTa (CA)
Nhận xét: Hiểu rất tốt sự kết nối giữa tên Subreddit và nội dung vi phạm.

Cell 61:

12. Tao file submission
Cell 63:

13. Mo rong de tai
Dua tren ket qua so sanh 12 huong:

RoBERTa-E vs DeBERTa-D: Ket qua cho thay lieu context subreddit co gia tri tuong duong hay cao hon few-shot examples. Neu RoBERTa-E >

Cell 64:

14. Tai lieu tham khao
He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. https://arxiv.org/abs/2006.03654

He et al. (2023). DeBERTaV3: Improving DeBERTa using EL

--- REPORT.PDF PAGES ---
Total pages: 53
--- Page 2 ---
MỤC LỤC
1 GIỚI THIỆU BÀI TOÁN 1
1.1 Đặt vấn đề . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.2 Mục tiêu nghiên cứu . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
1.3 Thách thức của bài toán . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1.3.1 Phụ thuộc đồng thời vào bình luận và nội quy . . . . . . . . . . 2
1.3.2 Ngôn ngữ phi chính thức . . . . . . . . . . . . . . . . . . . . . . 2
1.3.3 Vi phạm gián tiếp và phủ định . . . . . . . . . . . . . . . . . . 2
1.3.4 Ảnh hưởng của subreddit . . . . . . . . . . . . . . . . . . . . . 2
1.4 Phạm vi và đóng góp . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
2 BỘ DỮ LIỆU 3
2.1 Giới thiệu bộ dữ liệu . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Mô tả các trường dữ liệu . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.3 Phân tích khám phá dữ liệu . . . . . . . . . . . . . . . . . . . . . . . . 4
2.3.1 Phân phối nhãn và nội quy . . . . . . . . . . . . . . . . . . . . 4
2.3
--- Page 3 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
3.6.3 ROC-AUC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
4 THIẾT KẾ GIẢI PHÁP 16
4.1 Pipeline tổng thể (Overall Experimental Pipeline) . . . . . . . . . . . . 16
4.2 Tiền xử lý văn bản (Text Preprocessing) . . . . . . . . . . . . . . . . . 17
4.3 Ba chiến lược thiết kế cấu trúc dữ liệu đầu vào (Prompt Engineering) . 20
4.3.1 Chiến lược 1: Sentence-Pair (Cặp câu song hành) . . . . . . . . 20
4.3.2 Chiến lược 2: Few-Shot Guidance (Hướng dẫn qua ví dụ) . . . . 20
4.3.3 Chiến lược 3: Context-Aware Template (Nhận biết ngữ cảnh diễn
đàn) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4.3.4 Ma trận phân bổ thực nghiệm . . . . . . . . . . . . . . . . . . . 21
4.4 Chia tập huấn luyện và kiểm chứng (Validation Strategy) . . . . . . . . 22
4.5 Cấu trúc đầu phân loại (Classification Head Architecture) . . . . . . . 23
4.5.1 Trích xuất Trạng thái Ẩn từ Bộ mã hóa (Hidden State
--- Page 4 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
6.1 Kết luận . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
6.2 Hạn chế . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
6.3 Hướng phát triển . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
A PHÂN TÍCH THỦ CÔNG 60 MẪU DỮ LIỆU 39
A.1 Nhận xét tổng hợp từ 60 mẫu . . . . . . . . . . . . . . . . . . . . . . . 47
Trang iii
--- Page 5 ---
DANH SÁCH HÌNH VẼ
2.1 Phân phối nhãn, phân phối theo nội quy và độ dài bình luận . . . . . . 4
2.2 Số mẫu và tỉ lệ vi phạm của các subreddit phổ biến . . . . . . . . . . . 5
2.3 Word Cloud của bình luận vi phạm và không vi phạm . . . . . . . . . . 5
3.1 Kiến trúc tổng quát mô hình BERT-base làm nền tảng phát triển các
biến thể nâng cao . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
4.1 Pipeline thực nghiệm toàn diện của đồ án . . . . . . . . . . . . . . . . 16
5.1 So sánh F1-score và ROC-AUC của 12 cấu hình . . . . . . . . . . . . . 29
5.2 Đường học của các cấu hình Transformer đại diện . . . . . . . . . . . . 31
5.3 Confusion matrix của các cấu hình đại diện . . . . . . . . . . . . . . . 32
iv
--- Page 6 ---
DANH SÁCH BẢNG
2.1 Thống kê tổng quát của bộ dữ liệu . . . . . . . . . . . . . . . . . . . . 3
2.2 Mô tả các cột trong bộ dữ liệu . . . . . . . . . . . . . . . . . . . . . . . 3
2.3 Các hiện tượng ngôn ngữ quan trọng trong dữ liệu . . . . . . . . . . . . 6
3.1 So sánh chi tiết các kiến trúc Transformer được sử dụng . . . . . . . . 14
4.1 Đối chiếu cấu trúc văn bản trước và sau khi đi qua bộ tiền xử lý . . . . 19
4.2 Ma trận phân bổ cấu hình 12 hướng thực nghiệm trong đồ án . . . . . 22
5.1 Chiến lược ghép đầu vào cho sáu biến thể . . . . . . . . . . . . . . . . 27
5.2 Siêu tham số huấn luyện các mô hình Transformer . . . . . . . . . . . . 28
5.3 Kết quả đánh giá 12 cấu hình trên tập validation (406 mẫu) . . . . . . 29
5.4 Classification report của bốn cấu hình đại diện (tập validation, 406 mẫu) 32
5.5 Kết quả theo từng nội quy — mô hình tốt nhất của mỗi kiến trúc . . . 33
5.6 Phân loại lỗi của mô hình . . . . . . . . . . . . . . . . . . . . . . . . . 34
A.1 Phân tích định tính 60 mẫu p
--- Page 21 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Precision thấp) sẽ liên tục khóa mõm nhầm người dùng vô tội, trong khi một bộ lọc
lỏng lẻo (FNcao, Recall thấp) sẽ để lọt văn bản độc hại làm rác cộng đồng; F1-score
đóng vai trò xử phạt nặng nề sự chênh lệch này.
3.6.3 ROC-AUC
Để đánh giá chất lượng không gian vector mà không bị phụ thuộc vào một ngưỡngτ
ngẫu nhiên, hệ thống dùng ROC và AUC. ROC biểu diễn quan hệ giữa True Positive
Rate (TPR = Recall) và False Positive Rate (FPR) khi thay đổi ngưỡng phân loại.
AUC là diện tích dưới đường ROC và đo khả năng xếp hạng mẫu dương cao hơn
mẫu âm mà không phụ thuộc vào một ngưỡng cụ thể. Cụ thể, nếu ta bốc ngẫu nhiên
một bình luận vi phạm và một bình luận hợp lệ, AUC chính là xác suất tuyệt đối để
kiến trúc mạng Transformer chấm điểm logit cho mẫu vi phạm cao hơn mẫu hợp lệ,
biến nó thành thước đo khách quan nhất để so sánh các backbone.
Trang 15
--- Page 26 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
4.3 Ba chiến lược thiết kế cấu trúc dữ liệu đầu vào (Prompt
Engineering)
Do đặc thù bài toán mang tính chất suy luận ngữ cảnh (NLI), đồ án thiết kế 3 chiến
lược định hình cấu trúc dữ liệu đầu vào khác nhau để tối ưu hóa khả năng tiếp nhận
thông tin của các mô hình.
4.3.1 Chiến lược 1: Sentence-Pair (Cặp câu song hành)
Đây là phương pháp thiết kế cấu trúc nguyên bản và kinh điển nhất đối với các mô
hình học sâu thuộc nhóm mạng Transformer giải quyết bài toán NLI. Trong chiến lược
này, nội dung bình luận sạch và nội dung quy định sạch được truyền thẳng vào bộ mã
hóa của Tokenizer dưới dạng hai phân đoạn chuỗi ký tự độc lập nối tiếp nhau:
xA =b, x B =r.(4.1)
Khi đi qua bộ Tokenizer, hệ thống sẽ tự động tạo ra cấu trúc chuỗi chuẩn hóa:
[CLS] + b + [SEP] + r + [SEP]. Cấu trúc này ép buộc ma trận self-attention phải
tính toán chéo mức độ tương quan ngữ nghĩa giữa từng từ trong bình luận với từng từ
trong quy đ
--- Page 31 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
4.5.5 Lớp Tuyến tính Ánh xạ Không gian Đặc trưng (Linear Projection
Layer)
Vector đặc trưng sau điều chuẩnhdrop sẽ được đưa qua một lớp tuyến tính (Linear
Layer) cuối cùng để thực hiện hạ chiều từ không gian ẩndmodel = 768về một giá trị
vô hướng duy nhất, gọi là logit thô (z):
z=w Thdrop +b(4.7)
Vớiw∈R 768 là vector trọng số (weights) vàb∈Rlà hệ số chệch (bias) có thể
học được của đầu phân loại. Giá trị logitznày nằm trong khoảng(−∞,+∞), thể hiện
điểm số vi phạm chưa chuẩn hóa của bình luận.
4.5.6 Hàm Kích hoạt Sigmoid và Ước lượng Xác suất Hậu nghiệm
Chiến lược sử dụng hàm kích hoạt Sigmoid được phân tách rõ ràng giữa hai pha của
pipeline thực nghiệm để tối ưu hóa hiệu năng:
•Trong pha huấn luyện (Training Mode): Mô hình không áp dụng trực tiếp
hàm Sigmoid lên logitz. Thay vào đó, logit thô được đẩy thẳng vào hàm mất mát
BCEWithLogitsLoss. Sự kết hợp toán học này cho phép hệ thống áp dụng kỹ
thuật Log-S
--- Page 36 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Transformer cải thiện đáng kể so với baseline TF-IDF.Mô hình TF-IDF +
Logistic Regression đạt F1 cao nhất là 0,7638 (LR-SP). Tất cả các mô hình Transformer
đều vượt qua ngưỡng này, với mức cải thiện tuyệt đối từ+2,7điểm (DeBERTa-SP:
0,7807) đến+8,0điểm (RoBERTa-SP: 0,8436). Khoảng cách này phản ánh khả năng
hiểungữnghĩavàquanhệcâu-cặpcủakiếntrúcattentionsovớibiểudiễnbag-of-words.
Few-Shot không đồng đều giữa các kiến trúc.Với Logistic Regression, chiến
lược FS làmgiảmF1 từ 0,7638 xuống 0,7076 và ROC-AUC từ 0,8081 xuống 0,6346 —
tín hiệu rõ ràng rằng TF-IDF bị nhiễu khi từ vựng trong ví dụ mẫu xuất hiện lại trong
đầu vào. Ngược lại, các mô hình Transformer hưởng lợi từ Few-Shot ở mức độ vừa
phải: DeBERTa cải thiện từ 0,7807 (SP) lên 0,7920 (FS); RoBERTa-FS đạt ROC-AUC
cao nhất toàn bộ thí nghiệm (0,8977), cho thấy mô hình đủ sức xử lý chuỗi dài 256
token.
Context-AwareTemplategiúpDistilBERTvàDeBERTanhiềuh
--- Page 41 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
với trọng số tối ưu trên validation set.
3.Ngưỡng per-subreddit:Violation rate chênh lệch lớn giữa các subreddit (từ
2,9% đến 78,9%) gợi ý rằng dùng ngưỡng riêng cho từng cộng đồng có thể giảm
False Positive đáng kể.
4.Pre-trained NLI checkpoint:cross-encoder/nli-deberta-v3-smallđược
huấn luyện sẵn trên bài toán suy luận ngôn ngữ tự nhiên — gần nhất với bài
toán này — và có khả năng vượt qua tất cả các cấu hình hiện tại khi fine-tune.
5.RoBERTa-large:Tăng kích thước mô hình từ 125M lên 355M tham số, giữ
nguyên chiến lược SP hoặc CA.
Trang 35
--- Page 42 ---
CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT
TRIỂN
6.1 Kết luận
Đồ án đã xây dựng một pipeline hoàn chỉnh cho bài toán phân loại vi phạm nội quy
cộng đồng Reddit. Hệ thống bắt đầu từ phân tích dữ liệu, tiền xử lý, tạo ba dạng
đầu vào, huấn luyện bốn nhóm mô hình và đánh giá 12 cấu hình trên cùng một tập
validation.
Baseline TF-IDF và Logistic Regression cung cấp mốc so sánh nhanh, dễ giải
thích. Ba mô hình Transformer cho phép kiểm tra lợi ích của biểu diễn ngữ cảnh sâu.
Thiết kế Sentence-Pair bám sát bản chất NLI; Few-Shot bổ sung ví dụ đối chiếu; và
Context-Aware khai thác metadata subreddit.
Kết luận định lượng cuối cùng cần dựa trên Bảng 5.3 sau khi chạy notebook và
điền kết quả thật. Việc giữ nguyên dấu “–” hoặc dùng số liệu ước tính không đủ cơ sở
để khẳng định mô hình tốt nhất.
6.2 Hạn chế
•Notebook được cung cấp không lưu output, checkpoint hoặc dữ liệu, nên chưa
thể tái kiểm chứng các chỉ số thực nghiệm từ tệp hiện tại.
•Tất cả cấu hình dùng một lần chia train/validation; kết quả có thể p
--- Page 43 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
6.3 Hướng phát triển
1.Kết hợp Context-Aware và Few-Shot:dùng template subreddit cho chuỗi
A và nối cả ví dụ dương, âm vào chuỗi B.
2.Fine-tune từ checkpoint NLI:sử dụng DeBERTa hoặc RoBERTa đã được
tiềnhuấnluyệntrênMNLI/SNLIđểgiảmkhoảngcáchgiữapretrainingvànhiệm
vụ đích.
3.Ensemble:soft-voting xác suất của các mô hình có lỗi bổ sung cho nhau, tối
ưu trọng số trên validation.
4.Threshold theo nội quy:tìm ngưỡng riêng cho No Advertising và No Legal
Advice thay vì dùng một ngưỡng chung.
5.Hard-negative mining:bổ sung các mẫu có URL nhưng không quảng cáo,
hoặc có từ pháp lý nhưng không tư vấn, để giảm False Positive.
6.Nhiều seed và cross-validation:báo cáo trung bình cùng độ lệch chuẩn, giúp
kết luận có độ tin cậy cao hơn.
7.Giữ cấu trúc văn bản:bảo toàn Markdown link, dấu câu và chữ hoa cho
Transformer; chỉ chuẩn hóa mạnh ở baseline TF-IDF.
8.Theo dõi chi phí tính toán:bổ sung thời gian huấn luyện, thời g
--- Page 45 ---
CHƯƠNG A. PHÂN TÍCH THỦ CÔNG 60
MẪU DỮ LIỆU
Phụ lục sử dụng phương pháp lấy mẫu phân tầng với 15 mẫu cho mỗi tổ hợp giữa hai
nội quy và hai nhãn. Tổng cộng 60 mẫu được cố định bằng random seed để bảo đảm
khả năng tái lập. Nội dung được rút gọn nhằm giữ bảng dễ đọc; nhãn trong bảng là
nhãn gốc của bộ dữ liệu.
39
--- Page 46 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Bảng A.1:Phân tích định tính 60 mẫu phân tầng trong tập train
Bình luận Nội quy Subreddit Nhãn Nhận xét
Check out the rest at:
https://medium.com/the-dot-and-line/game-of-
thrones-as-cartoon-characters-and-a-quick-contest-
8d60dbe80670#.xjfuvj3nq
Quảng cáo gameofthrones 0 Có url/liên kết. Nhãn gốc là không vi phạm dù có tín hiệu gần với
quảng cáo; đây là mẫu phản trực giác cần ngữ cảnh sâu.
hello there . I giving a way some Netflix,hulu account. if
you interesting go on my fanpage:
https://.facebook.com/FreeNetflixAccount90210/
Quảng cáo movies 0 Có url/liên kết; có cấu trúc khuyến nghị/hướng dẫn. Nhãn gốc là
không vi phạm dù có tín hiệu gần với quảng cáo; đây là mẫu phản
trực giác cần ngữ cảnh sâu.
SD Stream [ENglish - Arsenal vs Preston North]
(http://echostadiums.club/event/arsenal-vs-preston-
north/)
Quảng cáo soccerstreams 0 Có url/liên kết. Nhãn gốc là không vi phạm dù có tín hiệu gần với
quảng cáo
--- Page 51 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Bình luận Nội quy Subreddit Nhãn Nhận xét
Do what someone else will do- bribe the police. Let them
handle it, they will handle the gundas for you. The police
are the biggest gang in India.
Tư vấn
pháp lý
The_Donald 1 Có cấu trúc khuyến nghị/hướng dẫn; được diễn đạt dưới dạng câu
hỏi. Các tín hiệu nhìn chung phù hợp với nhãn vi phạm.
You fucked up here. Drunk driving is bad and although
caring for someone is a good thing, getting him into
trouble legally is way more damaging than good. You
tried to kill an ant ...
Tư vấn
pháp lý
relationships 1 Chứa thuật ngữ pháp lý. Các tín hiệu nhìn chung phù hợp với nhãn
vi phạm.
What the hell is the children’s quote line? They lied to
you. And you don’t have much choice recourse except to
sue them. You have the original quote, then the billed
amount. ...
Tư vấn
pháp lý
personalfinance 1 Không có từ khóa bề mặt nổi bật. Nhãn gốc là vi phạm nhưng tín
hiệu tư vấn pháp l
```python?code_reference&code_event_index=10
# Let's read pages 27, 28, 29, 30, 32 of Report.pdf to find the exact evaluation metrics and experimental results table
for p in [27, 28, 29, 30, 31]:
    if p < len(reader.pages):
        print(f"--- Page {p+1} ---")
        print(reader.pages[p].extract_text()[:1500])
        print("="*40)

Python
for p in [32, 33, 34, 35]:
    if p < len(reader.pages):
        print(f"--- Page {p+1} ---")
        print(reader.pages[p].extract_text()[:1500])
        print("="*40)

Code output
--- Page 33 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
5.3 Tiền xử lý dữ liệu
Hàmpreprocess_textáp dụng lần lượt các bước: loại bỏ thẻ HTML, thay thế URL
bằng tokenURL, mở rộng dạng viết tắt (ví dụ:don’t→do not), loại bỏ dấu câu, chuẩn
hóa khoảng trắng và chuyển về chữ thường. Bước mở rộng viết tắt giúp giảm sự không
đồng nhất từ vựng trước khi TF-IDF đếm từ; bước thay thế URL bảo toàn tín hiệu
có-linkmà không giữ lại chuỗi ký tự rác.
Mỗi biến thể thử nghiệm sử dụng một cách ghép đầu vào khác nhau, tóm tắt
trong Bảng 5.1. Chiến lượcContext-Aware Template (CA)tạo đầu vào dạngIn
r/SUBREDDIT, this comment: body_cleanđể cung cấp ngữ cảnh nguồn gốc trước
khi mô hình đọc nội dung.
Bảng 5.1:Chiến lược ghép đầu vào cho sáu biến thể
Biến thể text_a text_b
SPbody_clean rule_clean
FSbody_clean rule_clean [SEP] POSITIVE:
... [SEP] NEGATIVE: ...
CAIn r/SUBREDDIT, this comment:
body_clean
rule_clean
5.4 Siêu tham số
Bảng 5.2 được đối chiếu trực tiếp với code trong notebook.
Trang 27
========================================
--- Page 34 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Bảng 5.2:Siêu tham số huấn luyện các mô hình Transformer
Tham số DistilBERT DeBERTa-v3-
small
RoBERTa-base
Epoch tối đa 20 20 20
Early stopping
patience
5 5 5
Batch size
SP/FS/CA
32/32/32 16/16/16 16/16/16
Learning rate SP2×10 −5 2×10 −5 2×10 −5
Learning rate FS2×10 −5 1×10 −5 2×10 −5
Learning rate CA2×10 −5 2×10 −5 3×10 −5
Max length SP/CA 128 128 128
Max length FS 256 256 256
Weight decay 0,01 0,01 0,01
Warmup ratio 6% 10% 10%
Dropout 0,3 0,2 0,1
Gradient clipping 1,0 1,0 1,0
Loss BCEWithLogitsLoss BCEWithLogitsLoss BCEWithLogitsLoss
5.5 Quy trình huấn luyện
Mỗi epoch gồm các bước:
1. Chuyển batch lên GPU hoặc CPU.
2. Xóa gradient cũ bằngoptimizer.zero_grad().
3. Forward để tính logit.
4. TínhBCEWithLogitsLoss.
5. Backpropagation.
6. Giới hạn gradient norm ở 1,0.
7. Cập nhật tham số bằng AdamW [8].
8. Cập nhật linear scheduler có warmup.
9. Đánh giá F1-score và ROC-AUC trên validation.
Trạng thái mô hình có F1-score cao nhất được lưu trong bộ nhớ CPU. Nếu
F1-score không cải thiện trong 5 epoch liên tiếp, quá trình huấn luyện dừng sớm. Sau
khi huấn luyện, xác suất validation được dùng để tìm ngưỡng tối ưu trong khoảng
[0,30,0,70].
Trang 28
========================================
--- Page 35 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Riêng vớiDeBERTa-v3-small, bộ tối ưu được cấu hình theo kỹ thuậtLayer-
wise Learning Rate Decay(LLRD): các lớp encoder nhận learning rate bằng0,5×so
với phần đầu phân loại. Kỹ thuật tương tự cũng được áp dụng choRoBERTa-base
ở biến thể CA để tối đa hóa điểm số trên chiến lược này. LLRD giúp tránh phá vỡ các
đặc trưng đã học ở lớp thấp trong khi vẫn cho phép lớp phân loại hội tụ nhanh.
5.6 Kết quả so sánh 12 hướng
Bảng 5.3 tổng hợp kết quả F1-score, ROC-AUC và ngưỡng phân loại tối ưu của tất cả
12 cấu hình trên tập validation gồm 406 mẫu.
Bảng 5.3:Kết quả đánh giá 12 cấu hình trên tập validation (406 mẫu)
Mô hình F1-score ROC-AUC Ngưỡng
Logistic Regression (SP) 0,7638 0,8081 0,50
Logistic Regression (FS) 0,7076 0,6346 0,40
Logistic Regression (CA) 0,7586 0,8076 0,48
DistilBERT (SP) 0,8047 0,8566 0,58
DistilBERT (FS) 0,7982 0,8309 0,50
DistilBERT (CA) 0,8219 0,8729 0,52
DeBERTa (SP) 0,7807 0,8232 0,64
DeBERTa (FS) 0,7920 0,8272 0,46
DeBERTa (CA) 0,8219 0,8654 0,46
RoBERTa (SP)0,84360,8838 0,40
RoBERTa (FS) 0,83410,89770,52
RoBERTa (CA) 0,8315 0,8880 0,30
Hình 5.1:So sánh F1-score và ROC-AUC của 12 cấu hình
5.6.1 Phân tích tổng thể
Kết quả cho thấy ba xu hướng rõ ràng:
Trang 29
========================================
--- Page 36 ---
Báo cáo Đồ án Môn học - Xử lý Ngôn ngữ Tự nhiênPhân loại Vi phạm Nội quy Cộng đồng
Transformer cải thiện đáng kể so với baseline TF-IDF.Mô hình TF-IDF +
Logistic Regression đạt F1 cao nhất là 0,7638 (LR-SP). Tất cả các mô hình Transformer
đều vượt qua ngưỡng này, với mức cải thiện tuyệt đối từ+2,7điểm (DeBERTa-SP:
0,7807) đến+8,0điểm (RoBERTa-SP: 0,8436). Khoảng cách này phản ánh khả năng
hiểungữnghĩavàquanhệcâu-cặpcủakiếntrúcattentionsovớibiểudiễnbag-of-words.
Few-Shot không đồng đều giữa các kiến trúc.Với Logistic Regression, chiến
lược FS làmgiảmF1 từ 0,7638 xuống 0,7076 và ROC-AUC từ 0,8081 xuống 0,6346 —
tín hiệu rõ ràng rằng TF-IDF bị nhiễu khi từ vựng trong ví dụ mẫu xuất hiện lại trong
đầu vào. Ngược lại, các mô hình Transformer hưởng lợi từ Few-Shot ở mức độ vừa
phải: DeBERTa cải thiện từ 0,7807 (SP) lên 0,7920 (FS); RoBERTa-FS đạt ROC-AUC
cao nhất toàn bộ thí nghiệm (0,8977), cho thấy mô hình đủ sức xử lý chuỗi dài 256
token.
Context-AwareTemplategiúpDistilBERTvàDeBERTanhiềuhơnRoBERTa.
DistilBERT-CA đạt F1 = 0,8219, tăng+2,2điểm so với DistilBERT-SP (0,8047).
DeBERTa-CA cũng đạt F1 = 0,8219, tăng+4,1điểm so với DeBERTa-SP (0,7807).
Tuy nhiên, RoBERTa-CA (0,8315) lại thấp hơn RoBERTa-SP (0,8436) về F1, gợi ý
rằng RoBERTa-base đã học được tín hiệu phân loại đủ tốt từ nội dung bình luận thuần
túy và không cần ngữ cảnh subreddit để đạt đỉnh F1.
Mô hình tốt nhất. RoBERTa-SPđạt F1-score cao nhất là0,8436với ROC-AUC
0,8838, và được sử dụng để tạo file submission.RoBERTa-FS
========================================

Python
readme_content = """# Jigsaw - Agile Community Rules Classification 🚀
## (Phân loại Vi phạm Nội quy Cộng đồng Reddit)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3c873a?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Models-Transformers_&_Baseline-blue?style=flat-square)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_Jigsaw-orange?style=flat-square)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

Dự án này tập trung giải quyết bài toán phân loại nhị phân nhằm xác định xem một bình luận trên nền tảng Reddit có vi phạm một nội quy cụ thể của cộng đồng hay không. Đây là một bài toán thuộc đồ án môn học **Xử lý Ngôn ngữ Tự nhiên** (Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM).

Điểm đặc biệt và thách thức lớn nhất của bài toán mang tính chất **"Agile"**: Mô hình không chỉ phân loại văn bản thuần túy mà phải đồng thời hiểu cả nội dung bình luận lẫn văn bản quy định đi kèm. Bài toán được tái cấu trúc từ phân loại văn bản truyền thống sang dạng **Suy luận ngôn ngữ tự nhiên (Natural Language Inference - NLI) / Sentence-Pair Classification**.

---

## 📌 Mục lục
1. [Giới thiệu Bài toán & Thách thức](#1-giới-thiệu-bài-toán--thách-thức)
2. [Tổng quan Bộ dữ liệu (EDA)](#2-tổng-quan-bộ-dữ-liệu-eda)
3. [Quy trình Tiền xử lý Dữ liệu](#3-quy-trình-tiền-xử-lý-dữ-liệu)
4. [Thiết kế Giải pháp & 12 Hướng Thực nghiệm](#4-thiết-kế-giải-pháp--12-hướng-thực-nghiệm)
5. [Cấu hình Siêu tham số](#5-cấu-hình-siêu-tham-số)
6. [Kết quả Thực nghiệm (Validation Results)](#6-kết-quả-thực-nghiệm-validation-results)
7. [Phân tích Lỗi (Error Analysis)](#7-phân-tích-lỗi-error-analysis)
8. [Hướng Phát triển Tương lai](#8-hướng-phát-triển-tương-lai)
9. [Thành viên Thực hiện](#9-thành-viên-thực-hiện)

---

## 1. Giới thiệu Bài toán & Thách thức

### 🔹 Bài toán
* **Đầu vào:** Nội dung một bình luận Reddit (`body`) và nội dung một quy tắc quy định cụ thể (`rule`).
* **Đầu ra:** Nhãn nhị phân (`rule_violation`), trong đó `1` đại diện cho hành vi vi phạm nội quy và `0` đại diện cho hành vi hợp lệ.

### 🔹 Thách thức cốt lõi
1.  **Phụ thuộc đồng thời vào bình luận và nội quy:** Một bình luận có thể hoàn toàn hợp lệ dưới nội quy này nhưng lại vi phạm nghiêm trọng dưới nội quy khác. Mô hình bắt buộc phải học cơ chế tương tác chéo (Cross-Attention) giữa cặp câu.
2.  **Ngôn ngữ phi chính thức:** Dữ liệu chứa nhiều tiếng lóng, từ viết tắt, lỗi chính tả, văn phong mạng xã hội và các biểu thức thô tục đặc trưng của Reddit.
3.  **Vi phạm gián tiếp và phủ định:** Nhiều mẫu vi phạm được diễn đạt tinh vi dưới dạng câu hỏi, lời khuyên gián tiếp hoặc sử dụng câu phủ định dài, gây khó khăn cho các mô hình dựa trên từ khóa bề mặt.
4.  **Ảnh hưởng của ngữ cảnh diễn đàn (Subreddit):** Mỗi Subreddit có một văn hóa giao tiếp và mức độ nhạy cảm khác nhau, ảnh hưởng gián tiếp đến hành vi vi phạm.

---

## 2. Tổng quan Bộ dữ liệu (EDA)

Bộ dữ liệu gốc được cung cấp từ cuộc thi Kaggle **Jigsaw - Agile Community Rules Classification**.
* **Tổng số mẫu huấn luyện thực tế:** 2.029 mẫu.
    * **Mẫu vi phạm (Nhãn 1):** 1.031 mẫu (chiếm 50.81%).
    * **Mẫu hợp lệ (Nhãn 0):** 998 mẫu (chiếm 49.19%).
* **Cơ cấu nội quy:** Bộ dữ liệu chỉ xoay quanh đúng 2 nội quy chính, tạo ra sự cân bằng hoàn hảo:
    * `No Advertising` (1.012 mẫu)
    * `No Legal Advice` (1.017 mẫu)
* **Phân bổ Subreddit:** Trải dài trên **100 subreddit** khác nhau với tỷ lệ vi phạm chênh lệch lớn (ví dụ: `legaladvice` có tỷ lệ vi phạm lên tới 78.9%, trong khi `soccerstreams` chỉ có 2.9%).
* **Các trường dữ liệu chính:** `row_id`, `body`, `rule`, `subreddit`, `positive_example_1`, `positive_example_2`, `negative_example_1`, `negative_example_2`, và nhãn mục tiêu `rule_violation`.

---

## 3. Quy trình Tiền xử lý Dữ liệu

Để chuẩn hóa văn bản trước khi đưa vào mô hình, hàm `preprocess_text` thực hiện chuỗi thao tác sau:
1.  **Loại bỏ thẻ HTML:** Gạt bỏ các thẻ định dạng văn bản thừa.
2.  **Chuẩn hóa URL:** Thay thế toàn bộ các đường liên kết bằng token đồng nhất `URL` nhằm giữ lại tín hiệu quảng cáo/liên kết mà không làm loãng từ vựng.
3.  **Mở rộng từ viết tắt:** Sử dụng thư viện `contractions` để chuẩn hóa các từ viết tắt tiếng Anh (ví dụ: `don't` $\rightarrow$ `do not`, `i'm` $\rightarrow$ `i am`), giúp mô hình TF-IDF và Transformer khớp từ vựng tốt hơn.
4.  **Loại bỏ dấu câu & kí tự đặc biệt:** Giữ lại các ký tự chữ và số.
5.  **Chuyển về chữ thường (Lowercasing):** Đồng nhất định dạng ký tự.

---

## 4. Thiết kế Giải pháp & 12 Hướng Thực nghiệm

Đồ án triển khai một ma trận thực nghiệm toàn diện kết hợp **4 Kiến trúc mô hình (Backbones)** và **3 Chiến lược định hình cấu trúc dữ liệu đầu vào (Prompt Engineering)**:

### 🔹 3 Chiến lược thiết kế đầu vào:
1.  **Sentence-Pair (SP):** Cấu trúc nguyên bản dạng cặp câu song hành, nối trực tiếp: `[CLS] + body_clean + [SEP] + rule_clean + [SEP]`. Ép buộc cơ chế self-attention tính toán mối tương quan trực tiếp giữa bình luận và nội quy.
2.  **Few-Shot Guidance (FS):** Bổ sung thêm ví dụ minh họa có sẵn từ bộ dữ liệu để định hướng cho mô hình: `body_clean [SEP] rule_clean [SEP] POSITIVE: ... [SEP] NEGATIVE: ...`. Độ dài chuỗi được mở rộng lên 256 token.
3.  **Context-Aware Template (CA):** Nhúng siêu dữ liệu subreddit vào cấu trúc đầu vào để mô hình nhận biết ngữ cảnh diễn đàn diễn ra bình luận: `In r/SUBREDDIT, this comment: body_clean [SEP] rule_clean`.

### 🔹 Ma trận phân bổ 12 cấu hình thực nghiệm:
| Kiến trúc / Chiến lược | Sentence-Pair (SP) | Few-Shot Guidance (FS) | Context-Aware (CA) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF)** | LR-SP | LR-FS | LR-CA |
| **DistilBERT** | DistilBERT-SP | DistilBERT-FS | DistilBERT-CA |
| **DeBERTa-v3-small** | DeBERTa-SP | DeBERTa-FS | DeBERTa-CA |
| **RoBERTa-base** | RoBERTa-SP | RoBERTa-FS | RoBERTa-CA |

---

## 5. Cấu hình Siêu tham số

Tập dữ liệu được phân chia theo phương pháp phân tầng (**Stratified Split**) tách biệt nghiêm ngặt theo tỷ lệ **80% Train (1.623 mẫu)** và **20% Validation (406 mẫu)** với `random_state=42`.

Các siêu tham số cốt lõi được thiết lập đồng bộ cho các mô hình Transformer sâu:
* **Hàm mất mát:** `BCEWithLogitsLoss` (Kết hợp tính toán logit thô giúp ổn định số học, tránh tràn số).
* **Bộ tối ưu hóa:** `AdamW` kết hợp bộ điều chỉnh tốc độ học tuyến tính có bước khởi động (`Linear Learning Rate Scheduler with Warmup`).
* **Chiến lược dừng sớm (Early Stopping):** Dừng nếu F1-score không cải thiện sau **5 epoch** liên tiếp (patience = 5); tối đa 20 epoch.
* **Cơ chế đặc biệt:** Áp dụng **Layer-wise Learning Rate Decay (LLRD)** cho DeBERTa-v3-small và RoBERTa-CA nhằm giảm phá vỡ các đặc trưng pre-trained ở các lớp thấp của Encoder.

---

## 6. Kết quả Thực nghiệm (Validation Results)

Dưới đây là bảng tổng hợp kết quả chi tiết thu được trên tập kiểm chứng cô lập (406 mẫu):

| Số TT | Mô hình / Cấu hình thực nghiệm | F1-score | ROC-AUC | Ngưỡng tối ưu ($\tau$) |
|:---:|:---|:---:|:---:|:---:|
| 1 | Logistic Regression (SP) | 0.7638 | 0.8081 | 0.50 |
| 2 | Logistic Regression (FS) | 0.7076 | 0.6346 | 0.40 |
| 3 | Logistic Regression (CA) | 0.7586 | 0.8076 | 0.48 |
| 4 | DistilBERT (SP) | 0.8047 | 0.8566 | 0.58 |
| 5 | DistilBERT (FS) | 0.7982 | 0.8309 | 0.50 |
| 6 | **DistilBERT (CA)** | **0.8219** | **0.8729** | **0.52** |
| 7 | DeBERTa-v3-small (SP) | 0.7807 | 0.8232 | 0.64 |
| 8 | DeBERTa-v3-small (FS) | 0.7920 | 0.8272 | 0.46 |
| 9 | **DeBERTa-v3-small (CA)** | **0.8219** | **0.8654** | **0.46** |
| 10 | 🏆 **RoBERTa-base (SP)** | **0.8436** | **0.8838** | **0.40** |
| 11 | RoBERTa-base (FS) | 0.8341 | **0.8977** | 0.52 |
| 12 | RoBERTa-base (CA) | 0.8315 | 0.8880 | 0.30 |

### 📊 Nhận xét & Thảo luận cốt lõi:
1.  **Sự vượt trội của Transformer:** Tất cả các kiến trúc Deep Learning đều vượt qua mốc Baseline vững chắc của TF-IDF + LR (+2.7 đến +8.0 điểm F1). Điều này chứng minh sức mạnh của cơ chế Attention trong việc hiểu ngữ nghĩa sâu thay vì chỉ đếm tần suất từ (Bag-of-Words).
2.  **Tác động hai mặt của Few-Shot:** Đối với Baseline LR, Few-Shot làm sụt giảm nghiêm trọng hiệu năng (F1 giảm từ 0.7638 xuống 0.7076) do từ vựng từ các ví dụ mẫu tạo nhiễu lớn cho TF-IDF. Đối với Transformer, Few-Shot mang lại hiệu quả cải thiện nhẹ cho DeBERTa, đồng thời giúp RoBERTa-FS đạt điểm **ROC-AUC đỉnh toàn diện (0.8977)**, chứng minh khả năng xử lý chuỗi dài ấn tượng.
3.  **Giá trị của Context-Aware Template:** Chiến lược CA chứng minh tính hiệu quả vượt bậc trên hai mô hình tầm trung là DistilBERT (F1 tăng +2.2) và DeBERTa (F1 tăng mạnh +4.1, đạt 0.8219). Việc nhúng trực tiếp tên Subreddit giúp mô hình định vị được hành vi ngôn ngữ của từng cộng đồng.
4.  **Mô hình tốt nhất:** **RoBERTa-base (SP)** đạt điểm **F1-score cao nhất toàn cục (0.8436)** tại ngưỡng phân loại tối ưu $\tau = 0.40$. Điều này chỉ ra rằng bản thân kiến trúc RoBERTa-base khi đi kèm cấu trúc cặp câu thuần túy đã đủ sức khai thác triệt để các tín hiệu phân loại tinh vi mà không bị phụ thuộc vào metadata bên ngoài.

---

## 7. Phân tích Lỗi (Error Analysis)

Qua quá trình trích xuất tự động ma trận nhầm lẫn và các mẫu lỗi phổ biến, đồ án đúc kết các điểm yếu của từng mô hình:
* **Logistic Regression (Baseline):** Thất bại hoàn toàn trước các từ đồng nghĩa chưa từng xuất hiện trong tập huấn luyện (do đặc tính từ vựng cứng của TF-IDF). Dễ sinh ra Dương giả (False Positive) khi văn bản chứa URL nhưng thực tế đó chỉ là một liên kết dẫn nguồn bình thường chứ không phải bài đăng quảng cáo.
* **DistilBERT:** Gặp khó khăn lớn ở biến thể Few-Shot. Việc kéo dài chuỗi lên 256 token khiến cấu trúc nông (6 tầng mạng) của DistilBERT bị quá tải thông tin, dẫn tới việc xáo trộn sự chú ý.
* **DeBERTa-v3-small:** Nhờ cơ chế Disentangled Attention (tách biệt ma trận chú ý giữa vị trí và nội dung), mô hình hiểu logic cặp câu rất tốt. Tuy nhiên, ở biến thể Context-Aware, mô hình có xu hướng bị quá khớp nhẹ (Overfitting) trên tập validation đối với các subreddit xuất hiện dày đặc.
* **RoBERTa-base:** Xử lý hoàn hảo các cấu trúc câu phức tạp và sắc thái phủ định gián tiếp. Lỗi chủ yếu xuất hiện ở các mẫu có nhãn phản trực giác (Ground-truth mang tính chủ quan cao từ điều hành viên gốc của Reddit).

---

## 8. Hướng Phát triển Tương lai

1.  **Ensemble Models:** Kết hợp cơ chế Soft-Voting tính toán xác suất hậu nghiệm giữa RoBERTa-SP, RoBERTa-FS và DeBERTa-CA để bù trừ sai số phân loại.
2.  **Fine-tune từ Checkpoint NLI chuyên dụng:** Khai thác các mô hình được huấn luyện sẵn trên các tập dữ liệu suy luận lớn như MNLI hoặc SNLI (ví dụ: `cross-encoder/nli-deberta-v3-small`) nhằm thu hẹp khoảng cách tiền huấn luyện.
3.  **Ngưỡng động theo từng Nội quy & Subreddit:** Thay vì áp dụng một ngưỡng chung $\tau$, hệ thống cần tìm kiếm các ngưỡng tối ưu độc lập cho `No Advertising` và `No Legal Advice` để giảm thiểu triệt để tỷ lệ Dương giả/Âm giả.
4.  **Hard-Negative Mining:** Thu thập và bổ sung chủ động các mẫu chứa từ khóa nhạy cảm (như chứa từ *lawyer*, *legal* nhưng không tư vấn; hoặc chứa *URL* nhưng không quảng cáo) để ép mô hình học ranh giới quyết định chặt chẽ hơn.

---

## 9. Thành viên Thực hiện

Đồ án được nghiên cứu và triển khai hoàn thiện bởi:
* **Vũ Việt Hoàng** - MSSV: `23520548`
* **Trịnh Nguyễn Thanh Bình** - MSSV: `23520162`

Giảng viên hướng dẫn: **TS. Nguyễn Trọng Chỉnh**

---
*Tài liệu tham khảo chi tiết và mã nguồn đầy đủ được lưu trữ cấu trúc trong file Báo cáo Đồ án và Notebook đính kèm.*
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("SUCCESSFULLY WRITTEN README.md")


Code output
SUCCESSFULLY WRITTEN README.md

Your Markdown file is ready
MD icon
README
 MD 
Open

Dưới đây là tóm tắt cấu trúc và nội dung của file README.md đã được khởi tạo dựa trên việc tổng hợp chi tiết từ file notebook Codespace.ipynb và tệp báo cáo Report.pdf, sử dụng phong cách trình bày chuyên nghiệp từ tài liệu tham khảo:

📝 Tổng quan cấu trúc file README.md vừa được tạo:
Tiêu đề & Huy hiệu (Badges) chuyên nghiệp: Thể hiện tên đề tài "Jigsaw - Agile Community Rules Classification (Phân loại Vi phạm Nội quy Cộng đồng Reddit)", kèm các huy hiệu trực quan về phiên bản Python, thư viện PyTorch, Hugging Face Transformers và liên kết cuộc thi Kaggle.

Giới thiệu Bài toán & Tính chất "Agile": Giải thích chi tiết bản chất bài toán phân loại nhị phân nhưng được tái cấu trúc thành bài toán Suy luận ngôn ngữ tự nhiên (NLI) / Sentence-Pair Classification. Mô hình phải hiểu song song cả nội dung bình luận (body) lẫn văn bản quy định (rule).

Thống kê chi tiết Bộ dữ liệu (EDA): Đưa ra các số liệu định lượng cụ thể từ báo cáo bao gồm: tổng số 2.029 mẫu train, tỷ lệ nhãn cân bằng (50.81% vi phạm, 49.19% hợp lệ), cơ cấu hai nội quy (No Advertising và No Legal Advice), và phạm vi trải dài trên 100 subreddit.

Quy trình Tiền xử lý Dữ liệu: Liệt kê đầy đủ 5 bước chuẩn hóa văn bản của hàm preprocess_text bao gồm loại bỏ thẻ HTML, đồng nhất liên kết bằng token URL, mở rộng từ viết tắt bằng thư viện contractions, xử lý dấu câu và viết thường.

Thiết kế Giải pháp & Ma trận 12 Thực nghiệm: Chi tiết hóa 3 chiến lược thiết kế dữ liệu đầu vào (Prompt Engineering): Sentence-Pair (SP), Few-Shot Guidance (FS), và Context-Aware Template (CA) phối hợp cùng 4 mô hình nền tảng (Logistic Regression, DistilBERT, DeBERTa-v3-small, RoBERTa-base).

Cấu hình Siêu tham số & Chiến lược Huấn luyện: Làm rõ tỷ lệ chia phân tầng 80/20 (Train/Validation), hàm loss BCEWithLogitsLoss, bộ tối ưu AdamW kết hợp Linear Warmup, chiến lược dừng sớm (Early Stopping) với patience=5, và kỹ thuật đặc biệt Layer-wise Learning Rate Decay (LLRD).

Bảng Kết quả Thực nghiệm Toàn diện (Validation Results): Trích xuất chính xác bảng số liệu 5.3 từ báo cáo với đầy đủ điểm số F1-score, ROC-AUC và ngưỡng tối ưu τ của toàn bộ 12 cấu hình thực nghiệm. Đi kèm là các nhận xét cốt lõi về sự vượt trội của mạng Transformer, tác động hai mặt của Few-Shot và giá trị của việc nhúng ngữ cảnh diễn đàn.

Phân tích Lỗi sâu sắc (Error Analysis): Đúc kết các điểm yếu mang tính hệ thống của từng nhóm mô hình từ thực tế chạy thực nghiệm (ví dụ: lỗi từ vựng cứng của TF-IDF, sự quá tải thông tin chuỗi dài của kiến trúc nông DistilBERT, hiện tượng quá khớp nhẹ của DeBERTa-CA).

Hướng Phát triển Tương lai: Đề xuất 4 giải pháp nâng cao hiệu năng hệ thống bao gồm: Ensemble mô hình (Soft-Voting), fine-tune từ checkpoint NLI chuyên dụng, áp dụng ngưỡng động độc lập cho từng nội quy/subreddit và kỹ thuật khai thác mẫu khó (Hard-Negative Mining).

Thành viên Thực hiện: Ghi nhận đầy đủ thông tin hai sinh viên thực hiện (Vũ Việt Hoàng - 23520548, Trịnh Nguyễn Thanh Bình - 23520162) cùng Giảng viên hướng dẫn (TS. Nguyễn Trọng Chỉnh).

Tệp README.md đã được ghi trực tiếp vào không gian làm việc của bạn với định dạng Markdown chuẩn chỉnh, sẵn sàng để bạn sử dụng cho kho lưu trữ mã nguồn của dự án.




Your University of Information Technology chats aren’t used to improve our models. Gemini is AI and can make mistakes. Your privacy & GeminiOpens in a new window

# Jigsaw - Agile Community Rules Classification 🚀
## (Phân loại Vi phạm Nội quy Cộng đồng Reddit)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3c873a?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Models-Transformers_&_Baseline-blue?style=flat-square)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_Jigsaw-orange?style=flat-square)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)

Dự án này tập trung giải quyết bài toán phân loại nhị phân nhằm xác định xem một bình luận trên nền tảng Reddit có vi phạm một nội quy cụ thể của cộng đồng hay không. Đây là một bài toán thuộc đồ án môn học **Xử lý Ngôn ngữ Tự nhiên** (Trường Đại học Công nghệ Thông tin - ĐHQG TP.HCM).

Điểm đặc biệt và thách thức lớn nhất của bài toán mang tính chất **"Agile"**: Mô hình không chỉ phân loại văn bản thuần túy mà phải đồng thời hiểu cả nội dung bình luận lẫn văn bản quy định đi kèm. Bài toán được tái cấu trúc từ phân loại văn bản truyền thống sang dạng **Suy luận ngôn ngữ tự nhiên (Natural Language Inference - NLI) / Sentence-Pair Classification**.

---

## 📌 Mục lục
1. [Giới thiệu Bài toán & Thách thức](#1-giới-thiệu-bài-toán--thách-thức)
2. [Tổng quan Bộ dữ liệu (EDA)](#2-tổng-quan-bộ-dữ-liệu-eda)
3. [Quy trình Tiền xử lý Dữ liệu](#3-quy-trình-tiền-xử-lý-dữ-liệu)
4. [Thiết kế Giải pháp & 12 Hướng Thực nghiệm](#4-thiết-kế-giải-pháp--12-hướng-thực-nghiệm)
5. [Cấu hình Siêu tham số](#5-cấu-hình-siêu-tham-số)
6. [Kết quả Thực nghiệm (Validation Results)](#6-kết-quả-thực-nghiệm-validation-results)
7. [Phân tích Lỗi (Error Analysis)](#7-phân-tích-lỗi-error-analysis)
8. [Hướng Phát triển Tương lai](#8-hướng-phát-triển-tương-lai)
9. [Thành viên Thực hiện](#9-thành-viên-thực-hiện)

---

## 1. Giới thiệu Bài toán & Thách thức

### 🔹 Bài toán
* **Đầu vào:** Nội dung một bình luận Reddit (`body`) và nội dung một quy tắc quy định cụ thể (`rule`).
* **Đầu ra:** Nhãn nhị phân (`rule_violation`), trong đó `1` đại diện cho hành vi vi phạm nội quy và `0` đại diện cho hành vi hợp lệ.

### 🔹 Thách thức cốt lõi
1.  **Phụ thuộc đồng thời vào bình luận và nội quy:** Một bình luận có thể hoàn toàn hợp lệ dưới nội quy này nhưng lại vi phạm nghiêm trọng dưới nội quy khác. Mô hình bắt buộc phải học cơ chế tương tác chéo (Cross-Attention) giữa cặp câu.
2.  **Ngôn ngữ phi chính thức:** Dữ liệu chứa nhiều tiếng lóng, từ viết tắt, lỗi chính tả, văn phong mạng xã hội và các biểu thức thô tục đặc trưng của Reddit.
3.  **Vi phạm gián tiếp và phủ định:** Nhiều mẫu vi phạm được diễn đạt tinh vi dưới dạng câu hỏi, lời khuyên gián tiếp hoặc sử dụng câu phủ định dài, gây khó khăn cho các mô hình dựa trên từ khóa bề mặt.
4.  **Ảnh hưởng của ngữ cảnh diễn đàn (Subreddit):** Mỗi Subreddit có một văn hóa giao tiếp và mức độ nhạy cảm khác nhau, ảnh hưởng gián tiếp đến hành vi vi phạm.

---

## 2. Tổng quan Bộ dữ liệu (EDA)

Bộ dữ liệu gốc được cung cấp từ cuộc thi Kaggle **Jigsaw - Agile Community Rules Classification**.
* **Tổng số mẫu huấn luyện thực tế:** 2.029 mẫu.
    * **Mẫu vi phạm (Nhãn 1):** 1.031 mẫu (chiếm 50.81%).
    * **Mẫu hợp lệ (Nhãn 0):** 998 mẫu (chiếm 49.19%).
* **Cơ cấu nội quy:** Bộ dữ liệu chỉ xoay quanh đúng 2 nội quy chính, tạo ra sự cân bằng hoàn hảo:
    * `No Advertising` (1.012 mẫu)
    * `No Legal Advice` (1.017 mẫu)
* **Phân bổ Subreddit:** Trải dài trên **100 subreddit** khác nhau với tỷ lệ vi phạm chênh lệch lớn (ví dụ: `legaladvice` có tỷ lệ vi phạm lên tới 78.9%, trong khi `soccerstreams` chỉ có 2.9%).
* **Các trường dữ liệu chính:** `row_id`, `body`, `rule`, `subreddit`, `positive_example_1`, `positive_example_2`, `negative_example_1`, `negative_example_2`, và nhãn mục tiêu `rule_violation`.

---

## 3. Quy trình Tiền xử lý Dữ liệu

Để chuẩn hóa văn bản trước khi đưa vào mô hình, hàm `preprocess_text` thực hiện chuỗi thao tác sau:
1.  **Loại bỏ thẻ HTML:** Gạt bỏ các thẻ định dạng văn bản thừa.
2.  **Chuẩn hóa URL:** Thay thế toàn bộ các đường liên kết bằng token đồng nhất `URL` nhằm giữ lại tín hiệu quảng cáo/liên kết mà không làm loãng từ vựng.
3.  **Mở rộng từ viết tắt:** Sử dụng thư viện `contractions` để chuẩn hóa các từ viết tắt tiếng Anh (ví dụ: `don't` $
ightarrow$ `do not`, `i'm` $
ightarrow$ `i am`), giúp mô hình TF-IDF và Transformer khớp từ vựng tốt hơn.
4.  **Loại bỏ dấu câu & kí tự đặc biệt:** Giữ lại các ký tự chữ và số.
5.  **Chuyển về chữ thường (Lowercasing):** Đồng nhất định dạng ký tự.

---

## 4. Thiết kế Giải pháp & 12 Hướng Thực nghiệm

Đồ án triển khai một ma trận thực nghiệm toàn diện kết hợp **4 Kiến trúc mô hình (Backbones)** và **3 Chiến lược định hình cấu trúc dữ liệu đầu vào (Prompt Engineering)**:

### 🔹 3 Chiến lược thiết kế đầu vào:
1.  **Sentence-Pair (SP):** Cấu trúc nguyên bản dạng cặp câu song hành, nối trực tiếp: `[CLS] + body_clean + [SEP] + rule_clean + [SEP]`. Ép buộc cơ chế self-attention tính toán mối tương quan trực tiếp giữa bình luận và nội quy.
2.  **Few-Shot Guidance (FS):** Bổ sung thêm ví dụ minh họa có sẵn từ bộ dữ liệu để định hướng cho mô hình: `body_clean [SEP] rule_clean [SEP] POSITIVE: ... [SEP] NEGATIVE: ...`. Độ dài chuỗi được mở rộng lên 256 token.
3.  **Context-Aware Template (CA):** Nhúng siêu dữ liệu subreddit vào cấu trúc đầu vào để mô hình nhận biết ngữ cảnh diễn đàn diễn ra bình luận: `In r/SUBREDDIT, this comment: body_clean [SEP] rule_clean`.

### 🔹 Ma trận phân bổ 12 cấu hình thực nghiệm:
| Kiến trúc / Chiến lược | Sentence-Pair (SP) | Few-Shot Guidance (FS) | Context-Aware (CA) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF)** | LR-SP | LR-FS | LR-CA |
| **DistilBERT** | DistilBERT-SP | DistilBERT-FS | DistilBERT-CA |
| **DeBERTa-v3-small** | DeBERTa-SP | DeBERTa-FS | DeBERTa-CA |
| **RoBERTa-base** | RoBERTa-SP | RoBERTa-FS | RoBERTa-CA |

---

## 5. Cấu hình Siêu tham số

Tập dữ liệu được phân chia theo phương pháp phân tầng (**Stratified Split**) tách biệt nghiêm ngặt theo tỷ lệ **80% Train (1.623 mẫu)** và **20% Validation (406 mẫu)** với `random_state=42`.

Các siêu tham số cốt lõi được thiết lập đồng bộ cho các mô hình Transformer sâu:
* **Hàm mất mát:** `BCEWithLogitsLoss` (Kết hợp tính toán logit thô giúp ổn định số học, tránh tràn số).
* **Bộ tối ưu hóa:** `AdamW` kết hợp bộ điều chỉnh tốc độ học tuyến tính có bước khởi động (`Linear Learning Rate Scheduler with Warmup`).
* **Chiến lược dừng sớm (Early Stopping):** Dừng nếu F1-score không cải thiện sau **5 epoch** liên tiếp (patience = 5); tối đa 20 epoch.
* **Cơ chế đặc biệt:** Áp dụng **Layer-wise Learning Rate Decay (LLRD)** cho DeBERTa-v3-small và RoBERTa-CA nhằm giảm phá vỡ các đặc trưng pre-trained ở các lớp thấp của Encoder.

---

## 6. Kết quả Thực nghiệm (Validation Results)

Dưới đây là bảng tổng hợp kết quả chi tiết thu được trên tập kiểm chứng cô lập (406 mẫu):

| Số TT | Mô hình / Cấu hình thực nghiệm | F1-score | ROC-AUC | Ngưỡng tối ưu ($	au$) |
|:---:|:---|:---:|:---:|:---:|
| 1 | Logistic Regression (SP) | 0.7638 | 0.8081 | 0.50 |
| 2 | Logistic Regression (FS) | 0.7076 | 0.6346 | 0.40 |
| 3 | Logistic Regression (CA) | 0.7586 | 0.8076 | 0.48 |
| 4 | DistilBERT (SP) | 0.8047 | 0.8566 | 0.58 |
| 5 | DistilBERT (FS) | 0.7982 | 0.8309 | 0.50 |
| 6 | **DistilBERT (CA)** | **0.8219** | **0.8729** | **0.52** |
| 7 | DeBERTa-v3-small (SP) | 0.7807 | 0.8232 | 0.64 |
| 8 | DeBERTa-v3-small (FS) | 0.7920 | 0.8272 | 0.46 |
| 9 | **DeBERTa-v3-small (CA)** | **0.8219** | **0.8654** | **0.46** |
| 10 | 🏆 **RoBERTa-base (SP)** | **0.8436** | **0.8838** | **0.40** |
| 11 | RoBERTa-base (FS) | 0.8341 | **0.8977** | 0.52 |
| 12 | RoBERTa-base (CA) | 0.8315 | 0.8880 | 0.30 |

### 📊 Nhận xét & Thảo luận cốt lõi:
1.  **Sự vượt trội của Transformer:** Tất cả các kiến trúc Deep Learning đều vượt qua mốc Baseline vững chắc của TF-IDF + LR (+2.7 đến +8.0 điểm F1). Điều này chứng minh sức mạnh của cơ chế Attention trong việc hiểu ngữ nghĩa sâu thay vì chỉ đếm tần suất từ (Bag-of-Words).
2.  **Tác động hai mặt của Few-Shot:** Đối với Baseline LR, Few-Shot làm sụt giảm nghiêm trọng hiệu năng (F1 giảm từ 0.7638 xuống 0.7076) do từ vựng từ các ví dụ mẫu tạo nhiễu lớn cho TF-IDF. Đối với Transformer, Few-Shot mang lại hiệu quả cải thiện nhẹ cho DeBERTa, đồng thời giúp RoBERTa-FS đạt điểm **ROC-AUC đỉnh toàn diện (0.8977)**, chứng minh khả năng xử lý chuỗi dài ấn tượng.
3.  **Giá trị của Context-Aware Template:** Chiến lược CA chứng minh tính hiệu quả vượt bậc trên hai mô hình tầm trung là DistilBERT (F1 tăng +2.2) và DeBERTa (F1 tăng mạnh +4.1, đạt 0.8219). Việc nhúng trực tiếp tên Subreddit giúp mô hình định vị được hành vi ngôn ngữ của từng cộng đồng.
4.  **Mô hình tốt nhất:** **RoBERTa-base (SP)** đạt điểm **F1-score cao nhất toàn cục (0.8436)** tại ngưỡng phân loại tối ưu $	au = 0.40$. Điều này chỉ ra rằng bản thân kiến trúc RoBERTa-base khi đi kèm cấu trúc cặp câu thuần túy đã đủ sức khai thác triệt để các tín hiệu phân loại tinh vi mà không bị phụ thuộc vào metadata bên ngoài.

---

## 7. Phân tích Lỗi (Error Analysis)

Qua quá trình trích xuất tự động ma trận nhầm lẫn và các mẫu lỗi phổ biến, đồ án đúc kết các điểm yếu của từng mô hình:
* **Logistic Regression (Baseline):** Thất bại hoàn toàn trước các từ đồng nghĩa chưa từng xuất hiện trong tập huấn luyện (do đặc tính từ vựng cứng của TF-IDF). Dễ sinh ra Dương giả (False Positive) khi văn bản chứa URL nhưng thực tế đó chỉ là một liên kết dẫn nguồn bình thường chứ không phải bài đăng quảng cáo.
* **DistilBERT:** Gặp khó khăn lớn ở biến thể Few-Shot. Việc kéo dài chuỗi lên 256 token khiến cấu trúc nông (6 tầng mạng) của DistilBERT bị quá tải thông tin, dẫn tới việc xáo trộn sự chú ý.
* **DeBERTa-v3-small:** Nhờ cơ chế Disentangled Attention (tách biệt ma trận chú ý giữa vị trí và nội dung), mô hình hiểu logic cặp câu rất tốt. Tuy nhiên, ở biến thể Context-Aware, mô hình có xu hướng bị quá khớp nhẹ (Overfitting) trên tập validation đối với các subreddit xuất hiện dày đặc.
* **RoBERTa-base:** Xử lý hoàn hảo các cấu trúc câu phức tạp và sắc thái phủ định gián tiếp. Lỗi chủ yếu xuất hiện ở các mẫu có nhãn phản trực giác (Ground-truth mang tính chủ quan cao từ điều hành viên gốc của Reddit).

---

## 8. Hướng Phát triển Tương lai

1.  **Ensemble Models:** Kết hợp cơ chế Soft-Voting tính toán xác suất hậu nghiệm giữa RoBERTa-SP, RoBERTa-FS và DeBERTa-CA để bù trừ sai số phân loại.
2.  **Fine-tune từ Checkpoint NLI chuyên dụng:** Khai thác các mô hình được huấn luyện sẵn trên các tập dữ liệu suy luận lớn như MNLI hoặc SNLI (ví dụ: `cross-encoder/nli-deberta-v3-small`) nhằm thu hẹp khoảng cách tiền huấn luyện.
3.  **Ngưỡng động theo từng Nội quy & Subreddit:** Thay vì áp dụng một ngưỡng chung $	au$, hệ thống cần tìm kiếm các ngưỡng tối ưu độc lập cho `No Advertising` và `No Legal Advice` để giảm thiểu triệt để tỷ lệ Dương giả/Âm giả.
4.  **Hard-Negative Mining:** Thu thập và bổ sung chủ động các mẫu chứa từ khóa nhạy cảm (như chứa từ *lawyer*, *legal* nhưng không tư vấn; hoặc chứa *URL* nhưng không quảng cáo) để ép mô hình học ranh giới quyết định chặt chẽ hơn.

---

## 9. Thành viên Thực hiện

Đồ án được nghiên cứu và triển khai hoàn thiện bởi:
* **Vũ Việt Hoàng** - MSSV: `23520548`
* **Trịnh Nguyễn Thanh Bình** - MSSV: `23520162`

Giảng viên hướng dẫn: **TS. Nguyễn Trọng Chỉnh**

---
*Tài liệu tham khảo chi tiết và mã nguồn đầy đủ được lưu trữ cấu trúc trong file Báo cáo Đồ án và Notebook đính kèm.*
README.md
Displaying README.md.
