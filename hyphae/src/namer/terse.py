import json
import random
import string

def terse_name(value_str):
    # Record IDs: table abbreviation
    if ':' in value_str and not value_str.startswith('{') and not value_str.startswith('['):
        table = value_str.split(':')[0]
        abbr = {
            'user': 'usr', 'session': 'sess', 'order': 'ord', 'product': 'prod',
            'comment': 'cmt', 'post': 'pst', 'file': 'file', 'image': 'img',
            'video': 'vid', 'config': 'cfg', 'cache': 'cache', 'log': 'log',
            'event': 'evt', 'notification': 'notif', 'message': 'msg',
            'transaction': 'txn', 'payment': 'pmt', 'account': 'acct',
            'setting': 'set', 'profile': 'prof', 'record': 'rec'
        }.get(table, table[:3])
        return abbr

    # Boolean
    if value_str in ('true', 'false'):
        return 'active' if value_str == 'true' else 'inactive'  # or just 'ok'? examples show 'active'

    # Number
    if value_str.lstrip('-').replace('.','',1).isdigit():
        num = float(value_str)
        if num == 0: return 'zero'
        if num == 1: return 'one'
        if 0 < num < 100: return 'n'
        return 'cnt'  # count

    # String name (alice, bob, etc.)
    if value_str.isalpha() and value_str[0].isupper() or value_str in ('alice','bob','carol','dave','eve'):
        return value_str.lower()

    # URL / endpoint
    if value_str.startswith(('ws://','http://','https://','postgres://')):
        return 'ep'

    # Date or time
    if '-' in value_str and len(value_str) == 10:  # YYYY-MM-DD
        return 'date'
    if ':' in value_str and len(value_str) <= 8:   # HH:MM:SS
        return 'time'

    # Array
    if value_str.startswith('[') and value_str.endswith(']'):
        return 'list'

    # Object (JSON)
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            obj = json.loads(value_str)
            # Priority fields
            for field in ['id','name','user','role','type','status','action','event','record']:
                if field in obj:
                    val = obj[field]
                    if isinstance(val, str):
                        if ':' in val:  # record id inside object
                            return val.split(':')[0][:3]
                        if val.lower() in ('alice','bob','carol','dave','eve'):
                            return val.lower()
                        if len(val) < 10 and val.isalpha():
                            return val.lower()
                    return field[:4]  # e.g., 'name' -> 'name', 'role' -> 'role'
            return 'obj'
        except:
            return 'obj'

    # Fallback
    return 'val'

def generate_terse_example():
    templates = [
        # Record IDs
        ("user:abc123", "usr"),
        ("session:xyz", "sess"),
        ("order:42", "ord"),
        ("product:99", "prod"),
        ("comment:xyz", "cmt"),
        ("event:login", "evt"),
        ("txn:abc", "txn"),
        ("file:report.pdf", "file"),
        # Names
        ("Alice", "alice"),
        ("Bob", "bob"),
        ("carol", "carol"),
        ("Dave", "dave"),
        ("Eve", "eve"),
        # Numbers
        ("42", "n"),
        ("100", "cnt"),
        ("0", "zero"),
        ("1", "one"),
        ("3.14", "n"),
        # Booleans
        ("true", "active"),
        ("false", "inactive"),
        # URLs / endpoints
        ("ws://localhost:8000", "ep"),
        ("https://api.example.com", "ep"),
        # Dates / times
        ("2024-01-01", "date"),
        ("23:59:59", "time"),
        # Arrays
        ("[\"a\",\"b\",\"c\"]", "list"),
        ("[1,2,3]", "list"),
        # Objects
        ('{"id":"user:1","name":"Bob","role":"admin"}', "bob"),
        ('{"user":"alice","action":"login"}', "alice"),
        ('{"type":"error","code":404}', "type"),
        ('{"status":"pending"}', "status"),
        ('{"record":"session:xyz"}', "sess"),
        ('{"event":"click"}', "event"),
        ('{"name":"carol"}', "carol"),
        ('{"role":"admin"}', "role"),
        # Fallback
        ("null", "val"),
        ("some-other-string", "val")
    ]
    # Weighted: choose from templates but also generate random variations
    if random.random() < 0.7:
        return random.choice(templates)
    else:
        # Random record ID
        table = random.choice(['user','session','order','product','comment','event','txn','file'])
        rid = f"{table}:{''.join(random.choices(string.ascii_lowercase+string.digits, k=4))}"
        abbr = {'user':'usr','session':'sess','order':'ord','product':'prod','comment':'cmt','event':'evt','txn':'txn','file':'file'}[table]
        return (rid, abbr)

# Generate 500 examples
examples = []
for _ in range(500):
    val, name = generate_terse_example()
    examples.append({"value": val, "name": name})

print(json.dumps(examples, indent=2))