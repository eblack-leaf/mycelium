import json
import random
import string

def hacker_name(value_str):
    # Record IDs
    if ':' in value_str and not value_str.startswith(('{','[')):
        table = value_str.split(':')[0]
        attitudes = [
            "that-user", "our-guy", "suspect", "culprit", "the-guy",
            "the-thing", "whatever", "this-guy", "some-record", "the-entry"
        ]
        # table-specific nicknames
        if table == "user":
            return random.choice(["that-user", "our-guy", "suspect", "the-dude"])
        if table == "session":
            return random.choice(["the-sesh", "this-session", "active-guy"])
        if table == "order":
            return random.choice(["the-order", "that-order", "order-thing"])
        if table == "product":
            return random.choice(["the-item", "that-product", "thingy"])
        return random.choice(attitudes)

    # Booleans
    if value_str == "true":
        return random.choice(["yep", "oh-yeah", "sure", "true-dat", "indeed"])
    if value_str == "false":
        return random.choice(["nope", "nah", "falsey", "uh-uh", "not-now"])

    # Numbers
    if value_str.lstrip('-').replace('.','',1).isdigit():
        num = float(value_str)
        if num == 0:
            return random.choice(["zero", "zilch", "nada", "the-big-zero"])
        if num == 1:
            return random.choice(["one", "single", "lone-guy"])
        if 1 < num < 100:
            return random.choice(["magic-num", "the-limit", "offset", "some-num", "count-this"])
        return random.choice(["big-number", "whatever", "the-count", "total-guy"])

    # Simple name strings
    if value_str.isalpha() and len(value_str) <= 10:
        if value_str[0].isupper() or value_str.lower() in ["alice","bob","carol","dave","eve"]:
            base = value_str.lower()
            if random.random() > 0.7:
                return base + random.choice(["bo", "ski", "ster", "inator", "bot"])
            return base

    # URLs / endpoints
    if value_str.startswith(('http://','https://','ws://','postgres://','redis://')):
        return random.choice(["the-db", "home-base", "local", "that-endpoint", "where-to", "the-gateway"])

    # Arrays
    if value_str.startswith('[') and value_str.endswith(']'):
        return random.choice(["the-list", "that-array", "stuff", "items", "batch"])

    # Objects (JSON)
    if value_str.startswith('{') and value_str.endswith('}'):
        try:
            obj = json.loads(value_str)
            # Look for role-like fields
            if "role" in obj:
                role = obj["role"]
                if role == "admin":
                    return "the-boss"
                if role == "user":
                    return "some-user"
                return f"{role}-guy"
            if "name" in obj:
                name = obj["name"]
                if isinstance(name, str) and name.lower() in ["alice","bob","carol"]:
                    return name.lower()
                return "named-guy"
            if "id" in obj:
                return "the-record"
            if "type" in obj:
                return f"{obj['type']}-thing"
            return random.choice(["payload", "the-blob", "our-victim", "culprit", "whatever-obj"])
        except:
            return random.choice(["bad-json", "garbage", "the-blob"])

    # Fallback
    return random.choice(["thingy", "stuff", "that-value", "the-data"])

def generate_hacker_example():
    # Lots of templates to ensure variety
    templates = [
        # Record IDs
        ("user:abc123", "that-user"),
        ("user:xyz", "our-guy"),
        ("session:deadbeef", "the-sesh"),
        ("order:x9f", "the-order"),
        ("product:42", "the-item"),
        ("comment:foo", "that-comment"),
        ("post:wall", "post-thing"),
        ("file:temp", "temp-file"),
        ("config:secrets", "secrets-guy"),
        ("log:error", "error-buddy"),
        ("event:click", "clicky"),
        ("notification:alert", "alert-guy"),
        # Names
        ("Alice", "alice"),
        ("Bob", "bob"),
        ("carol", "carol"),
        ("Dave", "dave"),
        ("Eve", "eve"),
        ("Mallory", "mallory"),
        ("alice", "alice"),
        ("bobbo", "bobbo"),
        # Numbers
        ("42", "magic-num"),
        ("100", "the-limit"),
        ("0", "zero"),
        ("1", "single"),
        ("7", "lucky"),
        ("256", "byte-limit"),
        ("1024", "kinda-big"),
        ("3.14", "pi-ish"),
        ("-1", "negative-guy"),
        # Booleans
        ("true", "yep"),
        ("false", "nope"),
        ("true", "oh-yeah"),
        ("false", "nah"),
        # URLs
        ("ws://localhost:8000", "the-db"),
        ("https://api.example.com", "home-base"),
        ("postgres://localhost:5432/db", "local-db"),
        ("redis://cache:6379", "cache-guy"),
        # Arrays
        ("[\"a\",\"b\",\"c\"]", "the-list"),
        ("[1,2,3]", "number-stuff"),
        ("[]", "empty-list"),
        # Objects
        ('{"id":"user:1","name":"Bob","role":"admin"}', "the-boss"),
        ('{"user":"alice","action":"login"}', "alice"),
        ('{"type":"error","code":404}', "error-thing"),
        ('{"status":"pending"}', "pending-guy"),
        ('{"record":"session:xyz"}', "the-record"),
        ('{"payload":"data"}', "payload"),
        ('{"foo":"bar"}', "whatever-obj"),
        ('{"name":"carol"}', "carol"),
        ('{}', "empty-blob"),
        # Other
        ("null", "nothing"),
        ("undefined", "uh-oh"),
    ]
    # Randomly pick from templates or generate fresh
    if random.random() < 0.8:
        return random.choice(templates)
    else:
        # Custom record ID
        table = random.choice(['user','session','order','product','comment','log','event'])
        rid = f"{table}:{''.join(random.choices(string.ascii_lowercase+string.digits, k=4))}"
        name = hacker_name(rid)
        return (rid, name)

# Generate 500 examples
examples = []
for _ in range(500):
    val, name = generate_hacker_example()
    examples.append({"value": val, "name": name})

print(json.dumps(examples, indent=2))