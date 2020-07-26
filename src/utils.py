import json
from hashlib import sha1


def get_params_hash(params):
    hash_str = json.dumps(params,
                          sort_keys=True,
                          ensure_ascii=False,
                          separators=None)
    hash_str = hash_str.encode('utf-8')
    return sha1(hash_str).hexdigest()[:7]
