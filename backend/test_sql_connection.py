import sys
import pyodbc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=VeriTextAI;Trusted_Connection=yes;"

print("=" * 65)
print("  KIỂM TRA KẾT NỐI TOÀN DIỆN VỚI SQL SERVER (VERITEXTAI)")
print("=" * 65)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    print("[+] 1. Kết nối vật lý tới SQL Server : THÀNH CÔNG (OK)")

    # 1. Phiên bản
    cursor.execute("SELECT @@VERSION;")
    version_str = cursor.fetchone()[0].split("\n")[0].strip()
    print(f"[+] 2. Phiên bản Microsoft SQL Server : {version_str}")

    # 2. Database
    cursor.execute("SELECT DB_NAME();")
    db_name = cursor.fetchone()[0]
    print(f"[+] 3. Cơ sở dữ liệu đang hoạt động   : {db_name}")

    # 3. Danh sách bảng
    cursor.execute("""
        SELECT TABLE_NAME,
               (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = t.TABLE_NAME) as col_count
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE TABLE_TYPE = 'BASE TABLE';
    """)
    tables = cursor.fetchall()
    print("[+] 4. Danh sách các bảng đã sẵn sàng:")
    for t in tables:
        print(f"       - Bảng: {t[0]:<22} | Số lượng cột: {t[1]}")

    # 4. Thử nghiệm ghi & đọc dữ liệu (CRUD test)
    test_email = "test_connection@veritext.ai"
    cursor.execute("""
        INSERT INTO users (email, password_hash, full_name, role)
        VALUES (?, ?, ?, ?);
    """, (test_email, "hash_pbkdf2_secret", "Tài khoản Kiểm tra Kết nối", "admin"))
    conn.commit()

    cursor.execute("SELECT id, email, full_name, role, created_at FROM users WHERE email = ?;", (test_email,))
    row = cursor.fetchone()
    print(f"[+] 5. Thử nghiệm GHI & ĐỌC dữ liệu   : THÀNH CÔNG RỰC RỠ!")
    print(f"       -> ID={row[0]} | Email={row[1]} | Quyền={row[3]} | Tạo lúc={row[4]}")

    # Dọn dẹp bản ghi kiểm tra
    cursor.execute("DELETE FROM users WHERE email = ?;", (test_email,))
    conn.commit()
    print("[+] 6. Dọn dẹp dữ liệu thử nghiệm     : SẠCH SẼ HOÀN TOÀN.")

    conn.close()
    print("=" * 65)
    print("🎉 KẾT LUẬN: KẾT NỐI SQL SERVER 100% HOÀN HẢO & SẴN SÀNG!")
    print("=" * 65)

except Exception as e:
    print("[-] LỖI KẾT NỐI:", e)
    sys.exit(1)
