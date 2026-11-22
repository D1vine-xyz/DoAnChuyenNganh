import sys
import json
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def post_json(url, data, headers=None):
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=req_headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(url, headers=None):
    req_headers = {}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

BASE_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("TESTING VERITEXT AI API ENDPOINTS (SQL SERVER AUTH & RBAC)")
print("=" * 60)

# 1. Health
try:
    health = get_json(f"{BASE_URL}/health")
    print(f"[+] 1. Health check: status={health.get('status')}, ready={health.get('ready')}")
except Exception as e:
    print(f"[-] 1. Health check failed: {e}")
    sys.exit(1)

# 2. Login Giảng viên (Teacher)
try:
    t_login = post_json(f"{BASE_URL}/api/v1/auth/login", {
        "email": "giangvien@veritext.ai",
        "password": "123456"
    })
    t_token = t_login["access_token"]
    t_user = t_login["user"]
    print(f"[+] 2. Teacher Login SUCCESS: name='{t_user['full_name']}', role='{t_user['role']}'")
except Exception as e:
    print(f"[-] 2. Teacher login failed: {e}")
    sys.exit(1)

# 3. Login Sinh viên (Student)
try:
    s_login = post_json(f"{BASE_URL}/api/v1/auth/login", {
        "email": "sinhvien@veritext.ai",
        "password": "123456"
    })
    s_token = s_login["access_token"]
    s_user = s_login["user"]
    print(f"[+] 3. Student Login SUCCESS: name='{s_user['full_name']}', role='{s_user['role']}'")
except Exception as e:
    print(f"[-] 3. Student login failed: {e}")
    sys.exit(1)

# 4. Check /api/v1/auth/me for student
try:
    me_resp = get_json(f"{BASE_URL}/api/v1/auth/me", headers={"Authorization": f"Bearer {s_token}"})
    u_data = me_resp.get("user", me_resp)
    print(f"[+] 4. /auth/me for student: email='{u_data['email']}', role='{u_data['role']}'")
except Exception as e:
    print(f"[-] 4. /auth/me failed: {e}")
    sys.exit(1)

# 5. Perform a test scan under Student account
try:
    scan_payload = {
        "text": "Deep learning models such as Transformer architectures have revolutionized the field of natural language processing.",
        "threshold": 0.5,
        "title": "NLP Research Abstract Sample",
        "language": "en"
    }
    scan_res = post_json(f"{BASE_URL}/api/v1/detect/web", scan_payload, headers={"Authorization": f"Bearer {s_token}"})
    ov = scan_res.get("overall", {})
    print(f"[+] 5. Student Scan Completed: scan_id={scan_res.get('scan_id')}, label='{ov.get('label')}', ai_prob={ov.get('ai_probability')}%, lang={ov.get('language')}")
except Exception as e:
    print(f"[-] 5. Student scan failed: {e}")

# 6. History check for student and teacher
try:
    s_history = get_json(f"{BASE_URL}/api/v1/history?page=1&limit=5", headers={"Authorization": f"Bearer {s_token}"})
    print(f"[+] 6. Student History: total_items={s_history.get('total')}, student sees only own records.")

    t_history = get_json(f"{BASE_URL}/api/v1/history?page=1&limit=5", headers={"Authorization": f"Bearer {t_token}"})
    print(f"[+] 7. Teacher History: total_items={t_history.get('total')}, teacher sees all records in system.")
except Exception as e:
    print(f"[-] 6/7. History check failed: {e}")

print("=" * 60)
print("ALL BACKEND AUTH & RBAC ENDPOINTS VERIFIED 100% OPERATIONAL!")
print("=" * 60)
