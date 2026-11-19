import os
import requests
import json

BASE = "http://localhost:8000"

print("=" * 70)
print("  KIỂM THỬ GIAI ĐOẠN 4 (PHASE 4: END-TO-END VERIFICATION)")
print("=" * 70)

# 1. Health check
print("\n[TEST 1] Kiểm tra sức khỏe hệ thống (/health):")
r0 = requests.get(f"{BASE}/health")
assert r0.status_code == 200
data_health = r0.json()
print(f"  -> Trạng thái       : {data_health['status']} (Ready: {data_health['ready']})")
print(f"  -> Ngôn ngữ hỗ trợ  : {data_health['languages_supported']}")
print(f"  -> Các mô hình nạp  : {list(data_health['models_loaded'].keys())}")
print(f"  -> PhoBERT sẵn sàng : {data_health['models_loaded']['phobert']}")

# 2. Vietnamese AI Text
print("\n[TEST 2] Phân tích văn bản Tiếng Việt do AI sinh (PhoBERT Router):")
ai_vi = (
    "Trí tuệ nhân tạo (AI) ngày nay đang đóng vai trò then chốt trong sự phát triển của nền công nghệ toàn cầu. "
    "Các mô hình xử lý ngôn ngữ tự nhiên hiện đại có khả năng tiếp nhận và xử lý lượng thông tin khổng lồ một cách ấn tượng. "
    "Tuy nhiên, sự xuất hiện ngày càng phổ biến của các nội dung do máy sinh đòi hỏi các giải pháp kiểm định nghiêm ngặt và đáng tin cậy. "
    "Chính vì vậy, việc ứng dụng các mô hình học sâu như PhoBERT vào bài toán phát hiện văn bản AI là một hướng đi vô cùng cấp thiết."
)
r1 = requests.post(f"{BASE}/api/v1/detect/sentences", json={"text": ai_vi, "threshold": 0.5})
assert r1.status_code == 200
d1 = r1.json()
print(f"  -> Scan ID          : {d1['scan_id']}")
print(f"  -> Ngôn ngữ nhận diện: {d1['overall']['language_label']} ({d1['overall']['detected_model']})")
print(f"  -> Xác suất AI      : {d1['overall']['ai_percentage']}")
print(f"  -> Kết luận         : {d1['overall']['label_vi']}")
print(f"  -> Độ tin cậy       : {d1['overall']['confidence']}")
print(f"  -> Thời gian xử lý  : {d1['total_processing_time_ms']} ms")
print(f"  -> Bóc tách câu ({len(d1['sentences'])} câu):")
for s in d1['sentences']:
    print(f"     * [{s['highlight_color'].upper()}] ({s['ai_percentage']} AI) - \"{s['sentence'][:65]}...\"")

# 3. Vietnamese Human Text
print("\n[TEST 3] Phân tích văn bản Tiếng Việt do Người viết (PhoBERT Router):")
human_vi = (
    "Chè cam tào phớ giải nhiệt mùa hè là một món giải khát rất dễ làm, vừa giúp thanh lọc cơ thể lại không bị béo ngậy. "
    "Cam tươi gọt sạch vỏ, bóc lấy phần tép và giữ lại một chút vỏ thái chỉ để tạo mùi thơm dịu nhẹ. "
    "Nước cam hòa cùng đường phèn đun sôi liu riu trên bếp, thêm chút bột năng khuấy đều đến khi sánh mịn thì tắt bếp. "
    "Múc từng lát tào phớ mềm mượt ra bát, chan nước cam lên trên và thưởng thức cùng vài viên đá mát lạnh."
)
r2 = requests.post(f"{BASE}/api/v1/detect/sentences", json={"text": human_vi, "threshold": 0.5})
assert r2.status_code == 200
d2 = r2.json()
print(f"  -> Scan ID          : {d2['scan_id']}")
print(f"  -> Ngôn ngữ nhận diện: {d2['overall']['language_label']}")
print(f"  -> Xác suất AI      : {d2['overall']['ai_percentage']} (Xác suất Người: {d2['overall']['human_probability']*100:.1f}%)")
print(f"  -> Kết luận         : {d2['overall']['label_vi']}")
print(f"  -> Thời gian xử lý  : {d2['total_processing_time_ms']} ms")
for s in d2['sentences']:
    print(f"     * [{s['highlight_color'].upper()}] ({s['ai_percentage']} AI) - \"{s['sentence'][:65]}...\"")

# 4. English Text (Ensemble PLMs)
print("\n[TEST 4] Phân tích văn bản Tiếng Anh (Ensemble 3 PLMs Router):")
en_sample = (
    "Artificial intelligence models have demonstrated exceptional capabilities across various natural language processing domains. "
    "These sophisticated architectures leverage deep contextual self-attention layers to produce coherent, syntactically fluent discourse. "
    "Consequently, developing resilient multi-model ensemble detection frameworks is of paramount importance to safeguard digital information fidelity."
)
r3 = requests.post(f"{BASE}/api/v1/detect/sentences", json={"text": en_sample, "threshold": 0.5})
assert r3.status_code == 200
d3 = r3.json()
print(f"  -> Scan ID          : {d3['scan_id']}")
print(f"  -> Ngôn ngữ nhận diện: {d3['overall']['language_label']} ({d3['overall']['detected_model']})")
print(f"  -> Xác suất AI      : {d3['overall']['ai_percentage']}")
print(f"  -> Phân rã mô hình  : Longformer={d3['overall']['model_breakdown']['individual_models']['longformer']['probability']*100:.1f}%, BERT={d3['overall']['model_breakdown']['individual_models']['bert']['probability']*100:.1f}%, RoBERTa={d3['overall']['model_breakdown']['individual_models']['roberta']['probability']*100:.1f}%")

# 5. PDF Report Generation
print("\n[TEST 5] Kiểm tra xuất báo cáo kiểm định PDF chuẩn hóa tiếng Việt:")
pdf_url = f"{BASE}/api/v1/report/pdf/{d1['scan_id']}"
r4 = requests.get(pdf_url)
assert r4.status_code == 200
assert len(r4.content) > 10000
print(f"  -> Tải PDF thành công: {len(r4.content):,} bytes từ {pdf_url}")

# 6. File upload inspection
print("\n[TEST 6] Tiếp nhận và phân tích tệp tài liệu tiếng Việt (.txt):")
dummy_content = "Học sinh, sinh viên cần tích cực rèn luyện tư duy phản biện khi ứng dụng trí tuệ nhân tạo trong học tập nghiên cứu. Việc phụ thuộc hoàn toàn vào các câu trả lời do máy sinh sẽ làm giảm sút năng lực tự học và tính sáng tạo cá nhân."
r5 = requests.post(
    f"{BASE}/api/v1/detect/upload",
    files={"file": ("bai_luan_tieng_viet.txt", dummy_content.encode("utf-8"), "text/plain")},
    data={"threshold": 0.5}
)
assert r5.status_code == 200
d5 = r5.json()
print(f"  -> Scan ID: {d5['scan_id']} | Tên tệp: {d5['filename']}")
print(f"  -> Ngôn ngữ: {d5['overall']['language_label']} | Xác suất AI: {d5['overall']['ai_percentage']}")

print("\n" + "=" * 70)
print("  [SUCCESS] TẤT CẢ 6/6 KỊCH BẢN GIAI ĐOẠN 4 ĐÃ ĐẠT CHUẨN HOÀN TOÀN!")
print("=" * 70)
