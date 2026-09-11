# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU (NVIDIA T4)

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Cập nhật biến `KHOA = "K4"` tại ô lưu bài nộp lên Google Drive.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `class_id`: 468
  - `class_name`: "cab"
  - `rank`: 1
  - `score`: 0.510915
  - `taxonomy_name`: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
  - Mô hình phân loại gán một nhãn duy nhất đại diện cho toàn bộ bức ảnh thay vì định vị từng đối tượng riêng lẻ. Với score cao nhất là 0.510915 (chiếm ~51% confidence), mô hình nhận định bối cảnh tổng thể của ảnh `traffic` là xe taxi ("cab").
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Danh sách lớp do tập dữ liệu ImageNet-1K quy định gồm 1,000 danh mục đã được chuẩn hóa trước khi huấn luyện mô hình `yolo11n-cls.pt`. Mô hình chỉ có thể dự đoán trong phạm vi 1,000 lớp này, không thể tự sinh ra lớp mới ngoài taxonomy.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - Vì mỗi taxonomy (ImageNet-1K, COCO-80, Pascal VOC) có bảng ánh xạ `class_id` khác nhau (ví dụ `class_id=0` trong COCO là "person", còn trong ImageNet là "tench"). `class_name` giúp con người đọc hiểu trực quan, còn `class_id` phục vụ lập trình và tính toán số học. Giữ cả 3 trường giúp tránh xung đột mã lớp, bảo đảm tính nhất quán và dễ tái lập dữ liệu.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline cần quy định rõ nguyên tắc chọn nhãn: (1) Chọn đối tượng chiếm diện tích lớn nhất ở tiền cảnh (dominant object), hoặc (2) Chuyển đổi sang bài toán multi-label classification để gắn nhiều nhãn đồng thời (như cab, minibus, person), hoặc (3) Thiết lập thứ tự ưu tiên giữa các lớp chủ thể, và (4) Quy định gắn cờ escalation khi ảnh có độ mơ hồ cao.
- Vì sao model score không phải ground truth?
  - Model score chỉ là giá trị xác suất toán học (confidence score) do mạng nơ-ron tính toán dựa trên trọng số đã học, phản ánh mức độ tự tin của mô hình chứ không phải sự thật khách quan. Mô hình có thể overconfident ở dự đoán sai hoặc underconfident ở ảnh phức tạp. Ground truth phải là nhãn chuẩn do con người xác minh và thống nhất theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - `class_name`: "person"
  - `score`: 0.912625
  - `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92]
  - `bbox_width`: 113.58
  - `bbox_height`: 279.68
- Diễn giải vị trí box bằng lời:
  - Bounding box định vị một người ("person") trong không gian bếp. Tọa độ góc trên-trái của hộp là (x=385.33, y=69.24) và góc dưới-phải là (x=498.92, y=348.92) tính bằng pixel từ gốc tọa độ (0, 0) ở góc trên cùng bên trái ảnh. Hộp có kích thước rộng 113.58 px, cao 279.68 px, bao trọn phần cơ thể nhìn thấy của người đang đứng nấu ăn.
- So sánh số prediction ở hai threshold:
  - Ở threshold = 0.20: mô hình phát hiện 17 vật thể (gồm 'person', 'bowl', 'oven', 'cup', 'spoon', 'potted plant', 'dining table', 'bottle').
  - Ở threshold = 0.60: mô hình chỉ giữ lại 6 vật thể có độ tự tin cao nhất (gồm 'person', 'bowl', 'oven').
  - (Ở threshold mặc định 0.35: phát hiện 11 vật thể).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi hạ threshold (từ 0.60 xuống 0.20), độ bao phủ (coverage / recall) tăng lên, phát hiện được thêm các vật thể nhỏ và mờ (như thìa, chai, chậu cây), nhưng xuất hiện nhiều dự đoán nhiễu (false positives) và trùng lặp (như 2 box oven đè nhau), làm tăng khối lượng kiểm duyệt cho reviewer. Khi tăng threshold lên 0.60, số lượng hộp giảm giúp reviewer duyệt nhanh hơn nhưng lại bỏ sót nhiều vật thể thực tế trong ảnh (false negatives).
- Đề xuất một quy tắc box chặt:
  - Hộp giới hạn phải bao khít toàn bộ phần nhìn thấy được (visible boundary) của vật thể, 4 cạnh hộp phải chạm vào các điểm ngoài cùng của vật thể với sai số không quá 3 pixel; không chừa khoảng trống pixel nền dư thừa và không được cắt lẹm vào đối tượng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần xác định: (1) Đóng khung theo phần nhìn thấy được (modal box) hay ước lượng toàn bộ hình thể thực tế (amodal box); (2) Ngưỡng diện tích che khuất tối đa cho phép gắn nhãn (ví dụ: nếu bị che khuất > 80% không thể nhận dạng chắc chắn thì bỏ qua); (3) Quy trình escalation: khi gặp đối tượng bị cắt mép hoặc che khuất phức tạp, annotator đánh dấu yêu cầu Reviewer/Lead đưa ra quyết định chuẩn hóa.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: "kitchen-001"
  - `class_name`: "person"
  - `score`: 0.899318
  - `polygon_point_count`: 348
  - `polygon_xy` (5 điểm đầu): [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
- Polygon bổ sung chi tiết gì so với box?
  - Bounding box chỉ là một hình chữ nhật bao quanh toàn bộ khu vực ngoại tiếp, bao gồm cả pixel phông nền và các vật thể phụ nằm ở các góc hộp. Polygon cung cấp đường viền đa giác chi tiết theo từng điểm ảnh, ôm khít biên dạng hình học thực tế của vật thể và loại bỏ hoàn toàn các pixel nền không thuộc đối tượng.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` (ví dụ `kitchen-001`) dùng để phân biệt riêng biệt từng cá thể đối tượng trong cùng một bức ảnh (kể cả khi chúng cùng lớp như nhiều chiếc bát `bowl`). `instance_id` KHÔNG phải là `class_id` (mã danh mục loại vật thể) và KHÔNG phải là `track_id` (mã theo dõi định danh xuyên suốt chuỗi video).
- Đề xuất một quy tắc biên mask:
  - Đường biên đa giác phải men sát ranh giới pixel nhìn thấy được của vật thể với độ lệch không quá 2 pixel; không được lẹm vào thân vật thể và không được chứa pixel nền hoặc pixel của vật thể khác che phủ.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần làm rõ: (1) Cách phân chia ranh giới khi các instance cùng lớp tiếp xúc hoặc chồng lên nhau (như các bát xếp lồng vào nhau) để không chồng chéo mask; (2) Cách xử lý vùng biên mờ do chuyển động hoặc bóng đổ; (3) Escalation: với các ranh giới tranh chấp hoặc che khuất phức tạp, annotator phải chuyển tiếp trường hợp đó lên Reviewer để thống nhất ranh giới chuẩn.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp (`class_id`, `class_name`) cho toàn bộ ảnh | Ảnh `traffic` có nhiều chủ thể (taxi, xe buýt, minibus, người) nhưng mô hình chỉ gán 1 lớp `cab` (score 0.51) | Xác định và gán nhãn theo chủ thể chính (dominant object) hoặc gắn nhãn đa lớp theo quy định guideline | Kiểm tra tính nhất quán giữa nhãn phân loại với ngữ cảnh bức ảnh; xác minh các trường hợp ảnh đa chủ thể |
| Phát hiện vật thể | Bounding box `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật thể | Ở threshold 0.35 bỏ sót các vật nhỏ như `spoon`, `bottle`; xuất hiện 2 hộp `oven` chồng lấn nhau | Vẽ box ôm khít toàn bộ phần nhìn thấy của từng vật thể thuộc danh mục COCO, không bỏ sót đồ vật nhỏ | Kiểm tra độ chặt của box (IoU), phát hiện box thừa, box thiếu, box cắt mép hoặc gán sai class |
| Instance segmentation | Đa giác polygon `[[x1, y1], [x2, y2], ...]` và mask nhị phân kèm `class_id`, `instance_id` | Ranh giới giữa các bát (`bowl`) xếp chồng khó phân biệt; tay người và dụng cụ nấu ăn bị dính mask | Chấm các điểm polygon bám sát đường bao thực tế của từng instance riêng biệt, tách rời các vật thể chạm nhau | Kiểm tra độ mịn và độ chính xác của đường biên mask (không dính nền, không mất chi tiết), kiểm tra tính duy nhất của `instance_id` |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Tuyệt đối không tải lên, lưu trữ hoặc chia sẻ dữ liệu nhạy cảm, thông tin định danh cá nhân (PII như họ tên, CCCD, khuôn mặt, biển số xe riêng tư) hay dữ liệu nội bộ/khách hàng lên các nền tảng đám mây hoặc kho lưu trữ công cộng.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Giảng viên hướng dẫn (GV), Lab Coach hoặc Đội ngũ vận hành đào tạo của chương trình.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
