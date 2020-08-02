import json
from hashlib import sha1


def get_params_hash(params):
    hash_str = json.dumps(params,
                          sort_keys=True,
                          ensure_ascii=False,
                          separators=None)
    hash_str = hash_str.encode('utf-8')
    return sha1(hash_str).hexdigest()[:7]


def initialize_amp(model,
                   opt_level='O1',
                   keep_batchnorm_fp32=None,
                   loss_scale='dynamic'):
    from apex import amp
    model.nn_module, model.optimizer = amp.initialize(
        model.nn_module, model.optimizer,
        opt_level=opt_level,
        keep_batchnorm_fp32=keep_batchnorm_fp32,
        loss_scale=loss_scale
    )
    model.amp = amp
