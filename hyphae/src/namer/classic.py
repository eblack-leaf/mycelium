import json
import random
import string

def generate_record_id():
    tables = ["user", "session", "order", "product", "comment", "post", "file", "image", "video", "audio", "document", "config", "cache", "log", "metric", "event", "notification", "message", "thread", "reply", "like", "follow", "tag", "category", "role", "permission", "setting", "profile", "account", "transaction", "payment", "invoice", "subscription", "plan", "feature", "flag", "variant", "experiment", "rollout", "deployment", "container", "pod", "node", "cluster", "region", "zone", "bucket", "object", "blob", "record", "entry", "item", "entity", "resource", "asset", "media", "content", "metadata", "schema", "index", "query", "result", "output", "input", "param", "argument", "field", "column", "row", "cell", "value", "key", "id", "name", "title", "label", "description", "summary", "detail", "note"]
    role_words = ["target", "active", "source", "selected", "primary", "secondary", "current", "previous", "next", "first", "last", "original", "copy", "draft", "published", "archived", "deleted", "temp", "permanent", "linked", "embedded", "nested", "root", "leaf", "parent", "child", "sibling", "ancestor", "descendant", "default", "custom", "system", "user", "admin", "guest", "owner", "creator", "updater", "viewer", "editor", "moderator", "subscriber", "member", "lead", "follower"]
    table = random.choice(tables)
    role = random.choice(role_words)
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(3,6)))
    return f"{table}:{role}-{suffix}" if random.random() > 0.3 else f"{table}:{suffix}"

def generate_name_from_record_id(value):
    table = value.split(':')[0]
    # Extract role from the part after colon, or invent based on table
    remainder = value.split(':',1)[1]
    if '-' in remainder and remainder.split('-')[0] in ['target','active','source','selected','primary','secondary','current','previous','next','first','last','original','copy','draft','published','archived','deleted','temp','permanent','linked','embedded','nested','root','leaf','parent','child','sibling','ancestor','descendant','default','custom','system','user','admin','guest','owner','creator','updater','viewer','editor','moderator','subscriber','member','lead','follower']:
        role = remainder.split('-')[0]
    else:
        role_candidates = ["target", "active", "source", "selected", "current"]
        role = random.choice(role_candidates)
    return f"{role}-{table}"

def generate_string_name():
    names = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi", "ivan", "judy", "mallory", "nina", "oscar", "peggy", "rupert", "sybil", "trent", "victor", "walter", "xena", "yves", "zara"]
    return random.choice(names)

def generate_number_name(value):
    num = int(value) if value.isdigit() else float(value)
    contexts = [
        ("result-count", "count"), ("page-limit", "limit"), ("offset", "offset"),
        ("max-items", "max"), ("min-value", "min"), ("total-score", "total"),
        ("average-rating", "avg"), ("percentile-rank", "percentile"),
        ("timeout-seconds", "timeout"), ("retry-attempts", "retries"),
        ("batch-size", "batch"), ("buffer-length", "length"),
        ("threshold-value", "threshold"), ("weight-factor", "weight"),
        ("priority-level", "priority"), ("version-number", "version"),
        ("error-code", "code"), ("port-number", "port"), ("thread-count", "threads"),
        ("chunk-size", "chunk"), ("limit-value", "limit")
    ]
    name = random.choice(contexts)[0]
    # sometimes make it descriptive of the number's magnitude
    if num == 0:
        name = "zero-offset" if random.random() > 0.5 else name
    elif num == 1:
        name = "single-item" if random.random() > 0.5 else name
    elif num == 100:
        name = "default-limit" if random.random() > 0.5 else name
    return name

def generate_object_name(obj_str):
    try:
        obj = json.loads(obj_str)
    except:
        return "json-object"
    # Look for prominent field
    priority_fields = ["id", "name", "username", "email", "role", "type", "kind", "status", "state", "action", "event", "record", "user", "item", "product", "order"]
    for field in priority_fields:
        if field in obj:
            val = obj[field]
            if isinstance(val, str):
                if ":" in val and len(val.split(":"))==2:  # looks like record id
                    table = val.split(":")[0]
                    return f"target-{table}"
                elif val.lower() in ["alice","bob","carol","dave","eve"]:
                    return val.lower()
                elif len(val) < 20 and val.isalpha():
                    return val.lower()
            elif isinstance(val, (int, float)):
                return f"{field}-value"
    # fallback to object type
    if "id" in obj:
        return "identified-object"
    if "type" in obj:
        return f"{obj['type']}-object"
    return "data-object"

def generate_value():
    choice = random.random()
    # 25% record id, 25% string name, 20% number, 20% object, 10% other (bool, url, date, array)
    if choice < 0.25:
        rid = generate_record_id()
        name = generate_name_from_record_id(rid)
        return {"value": rid, "name": name}
    elif choice < 0.5:
        name = generate_string_name()
        return {"value": name, "name": name}
    elif choice < 0.7:
        num = random.choice([str(random.randint(0,1000)), str(random.randint(1,100)), str(random.randint(0,10)), str(random.randint(0,1)), str(round(random.uniform(0,100),2)), "42", "100", "0", "1", "999", "3.14159", "2.71828"])
        name = generate_number_name(num)
        return {"value": num, "name": name}
    elif choice < 0.9:
        # object
        obj_templates = [
            '{"id":"user:abc123","name":"Alice","role":"admin"}',
            '{"id":"order:x9f","status":"pending","total":42.99}',
            '{"user_id":"bob","action":"login","timestamp":1640995200}',
            '{"record":"session:xyz","expires_in":3600}',
            '{"type":"error","code":404,"message":"Not found"}',
            '{"current_item":"product:99","quantity":2}',
            '{"name":"carol","email":"carol@example.com"}',
            '{"id":"doc:123","title":"Report","author":"dave"}',
            '{"event":"click","element":"button","page":"home"}',
            '{"config":{"theme":"dark","language":"en"}}',
            '{"result":{"score":98,"rank":1}}'
        ]
        obj_str = random.choice(obj_templates)
        name = generate_object_name(obj_str)
        return {"value": obj_str, "name": name}
    else:
        # other: bool, url, date, array, etc.
        other_vals = [
            ("true", "is-active"), ("false", "is-disabled"),
            ("ws://localhost:8000", "db-endpoint"),
            ("https://api.example.com/v1/users", "api-url"),
            ("postgres://localhost:5432/db", "database-dsn"),
            ("2001-01-01", "start-date"),
            ("14:30:00", "expiration-time"),
            ("[\"a\",\"b\",\"c\"]", "tag-list"),
            ("[1,2,3,4,5]", "number-array"),
            ("{\"x\":1,\"y\":2}", "coordinate-pair"),
            ("null", "empty-value")
        ]
        val, name = random.choice(other_vals)
        return {"value": val, "name": name}

# Generate 500 examples
examples = []
for _ in range(500):
    examples.append(generate_value())

# Output as JSON array
print(json.dumps(examples, indent=2))