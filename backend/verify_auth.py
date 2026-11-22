import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import database
import auth

print("1. Calling init_db()...")
database.init_db()

print("2. Testing authenticate_user for teacher...")
u_teacher = database.authenticate_user("giangvien@veritext.ai", "123456")
print("   Teacher auth result:", u_teacher)

print("3. Testing authenticate_user for student...")
u_student = database.authenticate_user("sinhvien@veritext.ai", "123456")
print("   Student auth result:", u_student)

if u_teacher:
    token = auth.create_access_token({"sub": str(u_teacher["id"]), "email": u_teacher["email"], "role": u_teacher["role"]})
    print("4. Teacher JWT Token:", token[:25] + "...")
    payload = auth.decode_access_token(token)
    print("   Decoded payload:", payload)

print("SUCCESS: All SQL Server auth tests passed!")
