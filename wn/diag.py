def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def total_attention(attention_weights: list):

    total_weight = None

    for layer in attention_weights:

        if total_weight is None:
            total_weight = layer
        else:
            total_weight = total_weight @ layer

    return total_weight