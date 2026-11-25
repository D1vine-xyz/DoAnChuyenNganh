"""
Tiện ích xem nhanh toàn bộ dữ liệu trong Microsoft SQL Server 2022 (Database: VeriTextAI)
Đồ án Chuyên ngành Kỹ thuật Phần mềm (Chủ đề 6)
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database

def print_separator(char="=", length=80):
    print(char * length)

def view_database():
    print_separator("=")
    print("        TRUY VẤN TRỰC TIẾP MICROSOFT SQL SERVER 2022 (VERITEXTAI)")
    print_separator("=")

    with database.get_connection() as conn:
        cursor = conn.cursor()

        # 1. BẢNG NGƯỜI DÙNG (USERS)
        print("\n[+] 1. BẢNG TÀI KHOẢN NGƯỜI DÙNG (TABLE: users)")
        print_separator("-", 80)
        cursor.execute("SELECT id, email, full_name, role, is_active, created_at FROM users ORDER BY id ASC;")
        users = cursor.fetchall()
        
        print(f"{'ID':<4} | {'EMAIL':<26} | {'HỌ VÀ TÊN':<30} | {'VAI TRÒ':<10} | {'NGÀY TẠO':<19}")
        print_separator("-", 80)
        for u in users:
            uid, email, name, role, active, created = u
            created_str = created.strftime("%Y-%m-%d %H:%M:%S") if created else ""
            role_badge = "Giảng viên" if role == "teacher" else "Sinh viên"
            print(f"{uid:<4} | {email:<26} | {name:<30} | {role_badge:<10} | {created_str:<19}")
        print(f"Tổng số tài khoản: {len(users)}\n")

        # 2. BẢNG LỊCH SỬ THẨM ĐỊNH (DETECTION_HISTORY)
        print("[+] 2. BẢNG LỊCH SỬ THẨM ĐỊNH (TABLE: detection_history)")
        print_separator("-", 95)
        cursor.execute("""
            SELECT h.id, 
                   COALESCE(u.full_name, N'Khách vãng lai (Ẩn danh)') as user_name,
                   h.title, 
                   h.label, 
                   h.ai_probability, 
                   h.language, 
                   h.word_count,
                   h.inference_time_ms,
                   h.created_at
            FROM detection_history h
            LEFT JOIN users u ON h.user_id = u.id
            ORDER BY h.id DESC;
        """)
        scans = cursor.fetchall()

        print(f"{'ID':<4} | {'NGƯỜI QUÉT':<24} | {'TIÊU ĐỀ BÀI QUÉT':<28} | {'KẾT QUẢ':<14} | {'AI %':<8} | {'NGÔN NGỮ':<8} | {'THỜI GIAN':<19}")
        print_separator("-", 95)
        for s in scans:
            sid, uname, title, label, ai_prob, lang, words, time_ms, created = s
            created_str = created.strftime("%Y-%m-%d %H:%M:%S") if created else ""
            title_short = (title[:25] + "...") if title and len(title) > 25 else (title or "")
            prob_pct = f"{ai_prob * 100:.1f}%" if ai_prob <= 1.0 else f"{ai_prob:.1f}%"
            print(f"{sid:<4} | {uname:<24} | {title_short:<28} | {label:<14} | {prob_pct:<8} | {lang:<8} | {created_str:<19}")
        print(f"Tổng số lượt quét đã ghi nhận: {len(scans)}\n")

        # 3. BẢNG LÔ BÀI TẬP LỚP HỌC (CLASSROOM_BATCHES)
        print("[+] 3. BẢNG LÔ CHẤM BÀI TẬP LỚP HỌC (TABLE: classroom_batches)")
        print_separator("-", 95)
        cursor.execute("""
            SELECT b.id, COALESCE(u.full_name, N'Giảng viên'), b.batch_name,
                   b.total_files, b.high_risk_count, b.moderate_risk_count,
                   b.safe_count, b.bypass_count, b.avg_ai_probability, b.created_at
            FROM classroom_batches b
            LEFT JOIN users u ON b.teacher_id = u.id
            ORDER BY b.id DESC;
        """)
        batches = cursor.fetchall()
        if not batches:
            print("  (Chưa có lô bài tập nào - sẵn sàng cho chức năng chấm bài tập hàng loạt)")
        else:
            print(f"{'ID':<4} | {'TÊN ĐỢT CHẤM':<28} | {'BÀI':<4} | {'CAO/NV/AT':<11} | {'BYPASS':<6} | {'AI TB':<7} | {'THỜI GIAN':<19}")
            print_separator("-", 95)
            for b in batches:
                bid, tname, bname, total, hi, mod, safe, byp, avg_p, created = b
                bname_short = (bname[:25] + "...") if bname and len(bname) > 25 else (bname or "")
                created_str = created.strftime("%Y-%m-%d %H:%M:%S") if created else ""
                print(f"{bid:<4} | {bname_short:<28} | {total:<4} | {hi}/{mod}/{safe:<5} | {byp:<6} | {avg_p*100:>5.1f}% | {created_str:<19}")
        print_separator("=")

if __name__ == "__main__":
    try:
        view_database()
    except Exception as e:
        print(f"[!] Lỗi truy vấn database: {e}")
