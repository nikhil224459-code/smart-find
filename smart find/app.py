from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import sqlite3
import os
import uuid
import hashlib
import re
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.secret_key = 'smartfind_secret_key_2025'
CORS(app)

# ─── Config ───

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH       = os.path.join(BASE_DIR, 'smartfind.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def row_to_dict(row):
    return dict(row)

# ════════════════════════════════════════════
# TEXT MATCHING ENGINE
# ════════════════════════════════════════════

STOPWORDS = {'a','an','the','of','in','at','on','and','or','is','was','for',
             'with','to','from','by','near','last','seen','found','male','female'}

def tokenize(text):
    """Lowercase, remove punctuation, split into meaningful tokens."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def token_overlap_score(a, b):
    """Jaccard-like token overlap between two strings. Returns 0.0-1.0."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)

def location_score(loc_a, loc_b):
    """Check if locations share any keywords."""
    return token_overlap_score(loc_a, loc_b)

def category_match(cat_a, cat_b):
    """Exact category match gives bonus."""
    return 1.0 if cat_a.lower() == cat_b.lower() else 0.0

def compute_match_score(case_a, case_b):
    """
    Compute weighted text similarity between two cases.
    Returns score 0-100 and breakdown dict.
    Only compares cases of the same report_type.
    """
    if case_a['report_type'] != case_b['report_type']:
        return 0, {}

    desc_s  = token_overlap_score(case_a['description'], case_b['description'])
    loc_s   = location_score(case_a['location'],    case_b['location'])
    cat_s   = category_match(case_a['category'] or '', case_b['category'] or '')

    # Weights: description 50%, location 30%, category 20%
    weighted = (desc_s * 0.50) + (loc_s * 0.30) + (cat_s * 0.20)
    score    = round(weighted * 100, 1)

    breakdown = {
        'description_similarity': round(desc_s * 100, 1),
        'location_similarity':    round(loc_s  * 100, 1),
        'category_match':         round(cat_s  * 100, 1),
        'total_score':            score,
    }
    return score, breakdown

def find_matches_for_case(target, all_cases, threshold=30):
    results = []
    for c in all_cases:
        if c.get('case_id') == target.get('case_id'):
            continue
        if c.get('status', '') == 'Closed':
            continue
        score, breakdown = compute_match_score(target, c)
        if score >= threshold:
            results.append({
                'matched_case_id':   c.get('case_id', ''),
                'matched_type':      c.get('report_type', ''),
                'matched_desc':      c.get('description', ''),
                'matched_location':  c.get('location', ''),
                'matched_date':      c.get('date', ''),
                'matched_status':    c.get('status', ''),
                'score':             score,
                'breakdown':         breakdown,
            })
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def detect_duplicate(new_case, all_cases, threshold=75):
    for c in all_cases:
        if c.get('status', '') == 'Closed':
            continue
        score, _ = compute_match_score(new_case, c)
        if score >= threshold:
            return True, c.get('case_id', '')
    return False, None

# ════════════════════════════════════════════
# IMAGE SIMILARITY ENGINE (OpenCV)
# ════════════════════════════════════════════

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def compute_image_similarity(img_path_a, img_path_b):
    """
    Compare two images using OpenCV histogram + ORB feature matching.
    Returns similarity score 0.0-100.0
    """
    if not CV2_AVAILABLE:
        return None, 'opencv-not-installed'
    try:
        full_a = os.path.join(BASE_DIR, 'static', img_path_a)
        full_b = os.path.join(BASE_DIR, 'static', img_path_b)
        if not os.path.exists(full_a) or not os.path.exists(full_b):
            return None, 'file-not-found'
        img_a = cv2.imread(full_a)
        img_b = cv2.imread(full_b)
        if img_a is None or img_b is None:
            return None, 'read-error'
        size  = (256, 256)
        img_a = cv2.resize(img_a, size)
        img_b = cv2.resize(img_b, size)
        # HSV Histogram
        hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        hist_score = max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
        # ORB Feature matching
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=500)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        feature_score = 0.0
        if des_a is not None and des_b is not None and len(des_a) > 0 and len(des_b) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_a, des_b)
            good = [m for m in matches if m.distance < 50]
            total = min(len(kp_a), len(kp_b))
            feature_score = min(len(good) / total, 1.0) if total > 0 else 0.0
        final = (hist_score * 0.60) + (feature_score * 0.40)
        return round(final * 100, 1), 'ok'
    except Exception as e:
        return None, str(e)

def find_image_matches(target_case, all_cases, threshold=40):
    if not CV2_AVAILABLE:
        return [], 'opencv-not-installed'
    if not target_case.get('photo_path'):
        return [], 'no-photo'
    results = []
    for c in all_cases:
        if c.get('case_id') == target_case.get('case_id'):
            continue
        if c.get('report_type') != 'Lost Item':
            continue
        if not c.get('photo_path') or c.get('status', '') == 'Closed':
            continue
        score, img_status = compute_image_similarity(target_case['photo_path'], c['photo_path'])
        if score is not None and score >= threshold:
            results.append({
                'matched_case_id':  c.get('case_id', ''),
                'matched_desc':     c.get('description', ''),
                'matched_location': c.get('location', ''),
                'image_score':      score,
            })
    results.sort(key=lambda x: x['image_score'], reverse=True)
    return results, 'ok'

# ════════════════════════════════════════════
# FACE COMPARISON ENGINE (face_recognition)
# ════════════════════════════════════════════

try:
    import face_recognition
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False

def encode_face(img_path):
    """Load image and return face encoding."""
    if not FACE_AVAILABLE:
        return None, 'face-recognition-not-installed'
    try:
        full_path = os.path.join(BASE_DIR, 'static', img_path)
        if not os.path.exists(full_path):
            return None, 'file-not-found'
        image = face_recognition.load_image_file(full_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            return None, 'no-face-detected'
        return encodings[0], 'ok'
    except Exception as e:
        return None, str(e)

def compare_faces(img_path_a, img_path_b):
    """
    Compare two face photos.
    Distance < 0.6 = same person. Converts to 0-100 score.
    """
    if not FACE_AVAILABLE:
        return None, False, 'face-recognition-not-installed'
    enc_a, status_a = encode_face(img_path_a)
    enc_b, status_b = encode_face(img_path_b)
    if enc_a is None:
        return None, False, f'Case A: {status_a}'
    if enc_b is None:
        return None, False, f'Case B: {status_b}'
    try:
        distance = face_recognition.face_distance([enc_a], enc_b)[0]
        score    = round(max(0.0, (1.0 - distance / 0.6)) * 100, 1)
        is_match = bool(distance < 0.6)
        return score, is_match, 'ok'
    except Exception as e:
        return None, False, str(e)

def find_face_matches(target_case, all_cases, threshold=50):
    if not FACE_AVAILABLE:
        return [], 'face-recognition-not-installed'
    if not target_case.get('photo_path'):
        return [], 'no-photo'
    results = []
    for c in all_cases:
        if c.get('case_id') == target_case.get('case_id'):
            continue
        if c.get('report_type') != 'Missing Person':
            continue
        if not c.get('photo_path') or c.get('status', '') == 'Closed':
            continue
        score, is_match, face_status = compare_faces(target_case['photo_path'], c['photo_path'])
        if score is not None and score >= threshold:
            results.append({
                'matched_case_id':  c.get('case_id', ''),
                'matched_desc':     c.get('description', ''),
                'matched_location': c.get('location', ''),
                'face_score':       score,
                'is_face_match':    is_match,
            })
    results.sort(key=lambda x: x['face_score'], reverse=True)
    return results, 'ok'

# ════════════════════════════════════════════
# AUTH DECORATORS
# ════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return jsonify({'success': False, 'error': 'Unauthorized. Please login.'}), 401
        return f(*args, **kwargs)
    return decorated

def police_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return jsonify({'success': False, 'error': 'Unauthorized. Please login.'}), 401
        if session.get('role') != 'police':
            return jsonify({'success': False, 'error': 'Access denied. Police Admin only.'}), 403
        return f(*args, **kwargs)
    return decorated

# ════════════════════════════════════════════
# DATABASE INIT
# ════════════════════════════════════════════

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id       TEXT UNIQUE NOT NULL,
                report_type   TEXT NOT NULL,
                description   TEXT NOT NULL,
                category      TEXT,
                date          TEXT NOT NULL,
                location      TEXT NOT NULL,
                reporter_name TEXT NOT NULL,
                contact       TEXT NOT NULL,
                photo_path    TEXT,
                status        TEXT DEFAULT 'Pending',
                archived      INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                closed_at     TEXT
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS match_suggestions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id         TEXT NOT NULL,
                matched_case_id TEXT NOT NULL,
                score           REAL NOT NULL,
                breakdown       TEXT,
                confirmed       INTEGER DEFAULT 0,
                confirmed_by    TEXT,
                created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(case_id, matched_case_id)
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE NOT NULL,
                password   TEXT NOT NULL,
                role       TEXT NOT NULL,
                full_name  TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id  INTEGER,
                username  TEXT,
                action    TEXT,
                case_id   TEXT,
                details   TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # Seed admins
        if conn.execute('SELECT COUNT(*) FROM admins').fetchone()[0] == 0:
            conn.executemany(
                'INSERT OR IGNORE INTO admins (username,password,role,full_name) VALUES (?,?,?,?)',
                [
                    ('police_admin', hash_password('police123'), 'police', 'Inspector Ramesh Kumar'),
                    ('ngo_admin',    hash_password('ngo123'),    'ngo',    'NGO Coordinator Priya'),
                ]
            )
            conn.commit()

        # Seed sample cases
        if conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0] == 0:
            samples = [
                ('SF-001','Lost Item',      'Black Samsung S23 Ultra phone',      'Mobile Phone',        '2025-01-10','Honnavar Bus Stand',    'Arjun Naik','9876543210','Pending'),
                ('SF-002','Missing Person', 'Ramesh K 45 male blue shirt grey trouser','Missing Person', '2025-01-08','Kumta Market',          'Seema K',   '9123456780','Verified'),
                ('SF-003','Lost Item',      'Brown leather wallet cash aadhar',    'Wallet / Purse',      '2025-01-12','SDM College Honnavar',  'Pooja M',   '9988776655','Matched'),
                ('SF-004','Missing Person', 'Priya S 19 female blue jeans white top','Missing Person',   '2025-01-15','Sirsi Road petrol bunk','Anil S',    '9012345678','Pending'),
                ('SF-005','Lost Item',      'Dell Inspiron laptop black bag',      'Laptop / Electronics','2025-01-06','Karwar Railway Station','Ravi P',   '9871234567','Closed'),
                ('SF-006','Lost Item',      'Gold chain 22 karat 8 grams yellow',  'Jewellery',           '2025-01-14','Ankola Temple',         'Meera R',   '9900112233','Verified'),
                ('SF-007','Missing Person', 'Suresh B 62 male diabetes forgets route','Missing Person',  '2025-01-11','Bhatkal Hospital',      'Kavita B',  '9765432109','Matched'),
                ('SF-008','Lost Item',      'Aadhar card PAN card blue folder',    'Document / ID',       '2025-01-09','Gokarna Bus Depot',     'Sunil H',   '9812345678','Closed'),
                ('SF-009','Lost Item',      'Red Honda Activa scooter KA15XX1234', 'Vehicle',             '2025-01-13','Honnavar Town',         'Ganesh T',  '9081726354','Pending'),
                ('SF-010','Missing Person', 'Anjali M 8 female blue school uniform','Missing Person',     '2025-01-16','Dandeli Forest Area',   'Rekha M',   '9900223344','Verified'),
                ('SF-011','Lost Item',      'Canon DSLR camera black lens kit bag','Laptop / Electronics','2025-01-07','Jog Falls Viewpoint',  'Vikram S',  '9876001234','Matched'),
                ('SF-012','Missing Person', 'Mohammed R 33 male truck driver cap', 'Missing Person',      '2025-01-05','Yellapur NH Road',      'Fatima R',  '9654321098','Verified'),
                ('SF-013','Lost Item',      'Samsung Galaxy S23 black smartphone', 'Mobile Phone',        '2025-01-11','Honnavar Bus Stand area','Rohan K',  '9112233445','Pending'),
                ('SF-014','Missing Person', 'Ramesh Kumar 45 years blue shirt',    'Missing Person',      '2025-01-09','Near Kumta Market road', 'Deepa K',  '9223344556','Pending'),
                ('SF-015','Lost Item',      'Gold necklace yellow 22k jewellery',  'Jewellery',           '2025-01-15','Ankola Temple road',    'Suman R',   '9334455667','Pending'),
            ]
            for s in samples:
                conn.execute(
                    'INSERT OR IGNORE INTO cases (case_id,report_type,description,category,date,location,reporter_name,contact,status) VALUES (?,?,?,?,?,?,?,?,?)',
                    s
                )
            conn.commit()

        # Auto-generate match suggestions for existing cases
        _generate_all_matches(conn)

def _generate_all_matches(conn):
    """Run matching engine on all non-closed cases and store suggestions."""
    rows = conn.execute(
        "SELECT case_id,report_type,description,category,location,date,status FROM cases WHERE status != 'Closed'"
    ).fetchall()
    cases = [row_to_dict(r) for r in rows]

    for case in cases:
        matches = find_matches_for_case(case, cases, threshold=30)
        for m in matches:
            import json
            try:
                conn.execute(
                    '''INSERT OR IGNORE INTO match_suggestions (case_id,matched_case_id,score,breakdown)
                       VALUES (?,?,?,?)''',
                    (case['case_id'], m['matched_case_id'], m['score'], json.dumps(m['breakdown']))
                )
            except Exception:
                pass
    conn.commit()

# ════════════════════════════════════════════
# PAGE ROUTES
# ════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    if 'admin_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def register_admin():
    try:
        data      = request.get_json()
        username  = data.get('username', '').strip()
        password  = data.get('password', '').strip()
        full_name = data.get('full_name', '').strip()
        role      = data.get('role', '').strip()

        if not all([username, password, full_name, role]):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400

        if role not in ['police', 'ngo']:
            return jsonify({'success': False, 'error': 'Role must be police or ngo'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400

        with get_db() as conn:
            existing = conn.execute('SELECT id FROM admins WHERE username=?', (username,)).fetchone()
            if existing:
                return jsonify({'success': False, 'error': 'Username already exists'}), 409

            conn.execute(
                'INSERT INTO admins (username, password, role, full_name) VALUES (?,?,?,?)',
                (username, hash_password(password), role, full_name)
            )
            conn.commit()

        return jsonify({'success': True, 'message': 'Registration successful! You can now login.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ════════════════════════════════════════════
# AUTH API
# ════════════════════════════════════════════

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400

    with get_db() as conn:
        admin = conn.execute(
            'SELECT * FROM admins WHERE username=? AND password=?',
            (username, hash_password(password))
        ).fetchone()

    if not admin:
        return jsonify({'success': False, 'error': 'Invalid username or password'}), 401

    session['admin_id']  = admin['id']
    session['username']  = admin['username']
    session['role']      = admin['role']
    session['full_name'] = admin['full_name']

    return jsonify({'success': True, 'username': admin['username'],
                    'role': admin['role'], 'full_name': admin['full_name']})

@app.route('/api/session', methods=['GET'])
def check_session():
    if 'admin_id' in session:
        return jsonify({'logged_in': True, 'username': session['username'],
                        'role': session['role'], 'full_name': session['full_name']})
    return jsonify({'logged_in': False})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})

# ════════════════════════════════════════════
# CASES API
# ════════════════════════════════════════════

@app.route('/api/cases', methods=['GET'])
def get_cases():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT case_id,report_type,description,category,date,location,reporter_name,contact,status,photo_path,archived FROM cases ORDER BY created_at DESC'
        ).fetchall()
    return jsonify([row_to_dict(r) for r in rows])

@app.route('/api/report', methods=['POST'])
def submit_report():
    import json
    try:
        report_type   = request.form.get('report_type', '').strip()
        description   = request.form.get('description', '').strip()
        category      = request.form.get('category', '').strip()
        date          = request.form.get('date', '').strip()
        location      = request.form.get('location', '').strip()
        reporter_name = request.form.get('reporter_name', '').strip()
        contact       = request.form.get('contact', '').strip()

        if not all([report_type, description, date, location, reporter_name, contact]):
            return jsonify({'success': False, 'error': 'All required fields must be filled'}), 400

        # ── Duplicate Detection ──
        with get_db() as conn:
            existing = conn.execute(
                "SELECT case_id,report_type,description,category,location FROM cases WHERE status != 'Closed'"
            ).fetchall()
        existing_list = [row_to_dict(r) for r in existing]

        new_case_check = {'case_id': 'NEW', 'report_type': report_type,
                          'description': description, 'category': category, 'location': location}
        is_dup, dup_id = detect_duplicate(new_case_check, existing_list, threshold=75)
        if is_dup:
            return jsonify({
                'success': False,
                'error': f'A very similar case already exists: {dup_id}. Please check existing cases before submitting.',
                'duplicate_case_id': dup_id
            }), 409

        # ── Generate Case ID ──
        with get_db() as conn:
            count   = conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
            case_id = f'SF-{str(count + 1).zfill(3)}'

        # ── Photo Upload ──
        photo_path = None
        if 'photo' in request.files:
            file = request.files['photo']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{case_id}_{uuid.uuid4().hex[:6]}_{file.filename}")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                photo_path = f'uploads/{filename}'

        # ── Save Case ──
        with get_db() as conn:
            conn.execute(
                '''INSERT INTO cases (case_id,report_type,description,category,date,location,reporter_name,contact,photo_path,status)
                   VALUES (?,?,?,?,?,?,?,?,?,'Pending')''',
                (case_id, report_type, description, category, date, location, reporter_name, contact, photo_path)
            )
            conn.commit()

        # ── Run Text Matching Engine ──
        with get_db() as conn:
            all_rows = conn.execute(
                "SELECT case_id,report_type,description,category,location,date,status,photo_path FROM cases WHERE status != 'Closed'"
            ).fetchall()
        all_cases = [row_to_dict(r) for r in all_rows]
        target    = next(c for c in all_cases if c['case_id'] == case_id)
        matches   = find_matches_for_case(target, all_cases, threshold=30)

        # ── Run Image / Face Matching Engine ──
        img_matches  = []
        face_matches = []

        if report_type == 'Lost Item' and photo_path:
            target_with_photo = {**target, 'photo_path': photo_path}
            img_matches, _ = find_image_matches(target_with_photo, all_cases, threshold=40)
            for im in img_matches:
                existing = next((m for m in matches if m['matched_case_id'] == im['matched_case_id']), None)
                if existing:
                    existing['breakdown']['image_similarity'] = im['image_score']
                    existing['score'] = round(min(100, existing['score'] * 0.80 + im['image_score'] * 0.20), 1)
                else:
                    matches.append({
                        'matched_case_id':  im['matched_case_id'],
                        'matched_type':     'Lost Item',
                        'matched_desc':     im['matched_desc'],
                        'matched_location': im['matched_location'],
                        'matched_date':     '',
                        'matched_status':   '',
                        'score':            im['image_score'],
                        'breakdown': {
                            'description_similarity': 0,
                            'location_similarity':    0,
                            'category_match':         0,
                            'image_similarity':       im['image_score'],
                            'total_score':            im['image_score'],
                        }
                    })

        elif report_type == 'Missing Person' and photo_path:
            target_with_photo = {**target, 'photo_path': photo_path}
            face_matches, _ = find_face_matches(target_with_photo, all_cases, threshold=50)
            for fm in face_matches:
                existing = next((m for m in matches if m['matched_case_id'] == fm['matched_case_id']), None)
                if existing:
                    existing['breakdown']['face_similarity'] = fm['face_score']
                    existing['score'] = round(min(100, existing['score'] * 0.70 + fm['face_score'] * 0.30), 1)
                else:
                    matches.append({
                        'matched_case_id':  fm['matched_case_id'],
                        'matched_type':     'Missing Person',
                        'matched_desc':     fm['matched_desc'],
                        'matched_location': fm['matched_location'],
                        'matched_date':     '',
                        'matched_status':   '',
                        'score':            fm['face_score'],
                        'breakdown': {
                            'description_similarity': 0,
                            'location_similarity':    0,
                            'category_match':         0,
                            'face_similarity':        fm['face_score'],
                            'total_score':            fm['face_score'],
                        }
                    })

        matches.sort(key=lambda x: x['score'], reverse=True)

        with get_db() as conn:
            for m in matches:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO match_suggestions (case_id,matched_case_id,score,breakdown) VALUES (?,?,?,?)',
                        (case_id, m['matched_case_id'], m['score'], json.dumps(m['breakdown']))
                    )
                    conn.execute(
                        'INSERT OR IGNORE INTO match_suggestions (case_id,matched_case_id,score,breakdown) VALUES (?,?,?,?)',
                        (m['matched_case_id'], case_id, m['score'], json.dumps(m['breakdown']))
                    )
                except Exception:
                    pass
            conn.commit()

        return jsonify({
            'success':        True,
            'case_id':        case_id,
            'matches_found':  len(matches),
            'image_matches':  len(img_matches),
            'face_matches':   len(face_matches),
            'cv2_available':  CV2_AVAILABLE,
            'face_available': FACE_AVAILABLE,
            'top_match':      matches[0] if matches else None
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cases/<case_id>/status', methods=['PUT'])
@login_required
def update_status(case_id):
    try:
        data       = request.get_json()
        new_status = data.get('status', '').strip()
        role       = session.get('role')

        allowed_statuses = ['Pending', 'Verified', 'Under Investigation', 'Matched', 'Closed']
        if new_status not in allowed_statuses:
            return jsonify({'success': False, 'error': 'Invalid status value'}), 400

        # Both Police and NGO have full rights to update status

        from datetime import datetime
        closed_at = datetime.now().isoformat() if new_status == 'Closed' else None
        archived  = 1 if new_status == 'Closed' else 0

        with get_db() as conn:
            result = conn.execute(
                'UPDATE cases SET status=?, archived=?, closed_at=? WHERE case_id=?',
                (new_status, archived, closed_at, case_id)
            )
            conn.execute(
                'INSERT INTO audit_log (admin_id,username,action,case_id,details) VALUES (?,?,?,?,?)',
                (session['admin_id'], session['username'], 'STATUS_UPDATE', case_id, f'Changed to {new_status}')
            )
            conn.commit()
            if result.rowcount == 0:
                return jsonify({'success': False, 'error': 'Case not found'}), 404

        return jsonify({'success': True, 'case_id': case_id, 'status': new_status})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ════════════════════════════════════════════
# MATCHING API
# ════════════════════════════════════════════

@app.route('/api/matches', methods=['GET'])
def get_all_matches():
    """Return all pending match suggestions with both case details."""
    with get_db() as conn:
        rows = conn.execute('''
            SELECT ms.id, ms.case_id, ms.matched_case_id, ms.score, ms.breakdown,
                   ms.confirmed, ms.confirmed_by, ms.created_at,
                   c1.report_type as type1, c1.description as desc1,
                   c1.location as loc1, c1.date as date1, c1.status as status1,
                   c2.report_type as type2, c2.description as desc2,
                   c2.location as loc2, c2.date as date2, c2.status as status2
            FROM match_suggestions ms
            JOIN cases c1 ON ms.case_id         = c1.case_id
            JOIN cases c2 ON ms.matched_case_id = c2.case_id
            WHERE ms.confirmed = 0
              AND c1.status != "Closed"
              AND c2.status != "Closed"
            ORDER BY ms.score DESC
        ''').fetchall()
    return jsonify([row_to_dict(r) for r in rows])

@app.route('/api/cases/<case_id>/matches', methods=['GET'])
def get_case_matches(case_id):
    """Return match suggestions for a specific case."""
    import json
    with get_db() as conn:
        target_row = conn.execute(
            'SELECT case_id,report_type,description,category,location,date,status FROM cases WHERE case_id=?',
            (case_id,)
        ).fetchone()
        if not target_row:
            return jsonify({'error': 'Case not found'}), 404

        all_rows = conn.execute(
            "SELECT case_id,report_type,description,category,location,date,status FROM cases WHERE status != 'Closed'"
        ).fetchall()

    target    = row_to_dict(target_row)
    all_cases = [row_to_dict(r) for r in all_rows]
    matches   = find_matches_for_case(target, all_cases, threshold=25)

    return jsonify({'case_id': case_id, 'matches': matches})

@app.route('/api/matches/<int:match_id>/confirm', methods=['PUT'])
@login_required
def confirm_match(match_id):
    """Confirms a match suggestion -> sets both cases to Matched."""
    try:
        with get_db() as conn:
            row = conn.execute('SELECT * FROM match_suggestions WHERE id=?', (match_id,)).fetchone()
            if not row:
                return jsonify({'success': False, 'error': 'Match not found'}), 404

            conn.execute(
                'UPDATE match_suggestions SET confirmed=1, confirmed_by=? WHERE id=?',
                (session['username'], match_id)
            )
            conn.execute(
                "UPDATE cases SET status='Matched' WHERE case_id IN (?,?)",
                (row['case_id'], row['matched_case_id'])
            )
            conn.execute(
                'INSERT INTO audit_log (admin_id,username,action,case_id,details) VALUES (?,?,?,?,?)',
                (session['admin_id'], session['username'], 'MATCH_CONFIRMED',
                 row['case_id'], f"Confirmed match with {row['matched_case_id']} (score {row['score']})")
            )
            conn.commit()

        return jsonify({'success': True, 'message': 'Match confirmed. Both cases marked as Matched.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/matches/<int:match_id>/reject', methods=['PUT'])
@login_required
def reject_match(match_id):
    """Dismiss a match suggestion."""
    try:
        with get_db() as conn:
            conn.execute(
                'UPDATE match_suggestions SET confirmed=-1, confirmed_by=? WHERE id=?',
                (session['username'], match_id)
            )
            conn.execute(
                'INSERT INTO audit_log (admin_id,username,action,case_id,details) VALUES (?,?,?,?,?)',
                (session['admin_id'], session['username'], 'MATCH_REJECTED', str(match_id), 'Match suggestion rejected')
            )
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/match/run', methods=['POST'])
def run_matching():
    """Manually trigger matching engine on all active cases."""
    import json
    with get_db() as conn:
        rows = conn.execute(
            "SELECT case_id,report_type,description,category,location,date,status FROM cases WHERE status != 'Closed'"
        ).fetchall()
    cases     = [row_to_dict(r) for r in rows]
    total_new = 0

    with get_db() as conn:
        for case in cases:
            matches = find_matches_for_case(case, cases, threshold=30)
            for m in matches:
                try:
                    conn.execute(
                        'INSERT OR IGNORE INTO match_suggestions (case_id,matched_case_id,score,breakdown) VALUES (?,?,?,?)',
                        (case['case_id'], m['matched_case_id'], m['score'], json.dumps(m['breakdown']))
                    )
                    total_new += 1
                except Exception:
                    pass
        conn.commit()

    return jsonify({'success': True, 'new_suggestions': total_new, 'cases_scanned': len(cases)})

# ════════════════════════════════════════════
# IMAGE & FACE COMPARISON API ROUTES
# ════════════════════════════════════════════

@app.route('/api/cases/<case_id>/image-matches', methods=['GET'])
def get_image_matches(case_id):
    """Compare uploaded item photo against all other Lost Item photos using OpenCV."""
    with get_db() as conn:
        target_row = conn.execute(
            'SELECT case_id,report_type,description,category,location,date,status,photo_path FROM cases WHERE case_id=?',
            (case_id,)
        ).fetchone()
        if not target_row:
            return jsonify({'error': 'Case not found'}), 404
        all_rows = conn.execute(
            "SELECT case_id,report_type,description,category,location,date,status,photo_path FROM cases WHERE status != 'Closed'"
        ).fetchall()

    target    = row_to_dict(target_row)
    all_cases = [row_to_dict(r) for r in all_rows]

    if not CV2_AVAILABLE:
        return jsonify({
            'case_id':   case_id,
            'available': False,
            'message':   'OpenCV not installed. Run: pip install opencv-python',
            'matches':   []
        })

    if not target.get('photo_path'):
        return jsonify({'case_id': case_id, 'available': True, 'message': 'No photo uploaded for this case', 'matches': []})

    results, status = find_image_matches(target, all_cases, threshold=30)
    return jsonify({
        'case_id':   case_id,
        'available': True,
        'matches':   results,
        'count':     len(results)
    })

@app.route('/api/cases/<case_id>/face-matches', methods=['GET'])
def get_face_matches(case_id):
    """Compare uploaded person photo against all other Missing Person photos using face_recognition."""
    with get_db() as conn:
        target_row = conn.execute(
            'SELECT case_id,report_type,description,category,location,date,status,photo_path FROM cases WHERE case_id=?',
            (case_id,)
        ).fetchone()
        if not target_row:
            return jsonify({'error': 'Case not found'}), 404
        all_rows = conn.execute(
            "SELECT case_id,report_type,description,category,location,date,status,photo_path FROM cases WHERE status != 'Closed'"
        ).fetchall()

    target    = row_to_dict(target_row)
    all_cases = [row_to_dict(r) for r in all_rows]

    if not FACE_AVAILABLE:
        return jsonify({
            'case_id':   case_id,
            'available': False,
            'message':   'face_recognition not installed. Run: pip install face_recognition',
            'matches':   []
        })

    if not target.get('photo_path'):
        return jsonify({'case_id': case_id, 'available': True, 'message': 'No photo uploaded for this case', 'matches': []})

    results, status = find_face_matches(target, all_cases, threshold=40)
    return jsonify({
        'case_id':   case_id,
        'available': True,
        'matches':   results,
        'count':     len(results)
    })

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Return status of all comparison engines."""
    return jsonify({
        'text_matching':    True,
        'image_comparison': CV2_AVAILABLE,
        'face_recognition': FACE_AVAILABLE,
        'opencv_version':   cv2.__version__ if CV2_AVAILABLE else None,
    })

# ════════════════════════════════════════════
# STATS & AUDIT
# ════════════════════════════════════════════

@app.route('/api/stats', methods=['GET'])
def get_stats():
    with get_db() as conn:
        total       = conn.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
        pending     = conn.execute("SELECT COUNT(*) FROM cases WHERE status='Pending'").fetchone()[0]
        verified    = conn.execute("SELECT COUNT(*) FROM cases WHERE status='Verified'").fetchone()[0]
        matched     = conn.execute("SELECT COUNT(*) FROM cases WHERE status='Matched'").fetchone()[0]
        closed      = conn.execute("SELECT COUNT(*) FROM cases WHERE status='Closed'").fetchone()[0]
        archived    = conn.execute("SELECT COUNT(*) FROM cases WHERE archived=1").fetchone()[0]
        suggestions = conn.execute("SELECT COUNT(*) FROM match_suggestions WHERE confirmed=0").fetchone()[0]
    return jsonify({'total': total, 'pending': pending, 'verified': verified,
                    'matched': matched, 'closed': closed, 'archived': archived,
                    'pending_suggestions': suggestions})

@app.route('/api/audit', methods=['GET'])
@login_required
def get_audit():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT username,action,case_id,details,timestamp FROM audit_log ORDER BY timestamp DESC LIMIT 50'
        ).fetchall()
    return jsonify([row_to_dict(r) for r in rows])

# ─── Run ───

import os
with app.app_context():
    init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)