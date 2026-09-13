class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s3)", arg1_1: "bf16[1, s3]", arg2_1: "i32[1]"):
        # File: /home/liutaishuang/projects/shennong-tokenspeed/tokenspeed/python/tokenspeed/runtime/sampling/utils.py:72 in gather_token_logprobs_torch, code: raw_logprobs = torch.log_softmax(logits.float(), dim=-1)
        convert_element_type: "f32[1, s3]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        _log_softmax: "f32[1, s3]" = torch.ops.aten._log_softmax.default(convert_element_type, -1, False);  convert_element_type = None
        
        # File: /home/liutaishuang/projects/shennong-tokenspeed/tokenspeed/python/tokenspeed/runtime/sampling/utils.py:73 in gather_token_logprobs_torch, code: return raw_logprobs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        unsqueeze: "i32[1, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        gather: "f32[1, 1]" = torch.ops.aten.gather.default(_log_softmax, -1, unsqueeze);  _log_softmax = unsqueeze = None
        squeeze: "f32[1]" = torch.ops.aten.squeeze.dim(gather, -1);  gather = None
        return (squeeze,)
        