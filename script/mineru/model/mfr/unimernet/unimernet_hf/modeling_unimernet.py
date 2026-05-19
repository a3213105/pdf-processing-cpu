import os
import warnings
from typing import Optional
from tqdm import tqdm

import torch
from ftfy import fix_text
from loguru import logger

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import logger as base_model_logger

from .unimer_swin import UnimerSwinConfig, UnimerSwinModel, UnimerSwinImageProcessor
from .unimer_mbart import UnimerMBartConfig, UnimerMBartForCausalLM
from ...utils import latex_rm_whitespace

AutoConfig.register(UnimerSwinConfig.model_type, UnimerSwinConfig)
AutoConfig.register(UnimerMBartConfig.model_type, UnimerMBartConfig)
AutoModel.register(UnimerSwinConfig, UnimerSwinModel)
AutoModelForCausalLM.register(UnimerMBartConfig, UnimerMBartForCausalLM)


# TODO: rewrite tokenizer
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text, **kwargs):
        return self.tokenizer(
            text,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            **kwargs,
        )

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated_text = [fix_text(text) for text in generated_text]
        return generated_text

    def detokenize(self, tokens):
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]):
                    del toks[b][i]
        return toks

class UnimernetModel(VisionEncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        tokenizer: Optional = None,
        transform: Optional = None,
    ):
        # VisionEncoderDecoderModel's checking log has bug, disable for temp.
        base_model_logger.disabled = True
        try:
            super().__init__(config, encoder, decoder)
        finally:
            base_model_logger.disabled = False

        if not config or not hasattr(config, "_name_or_path"):
            raise RuntimeError("config._name_or_path is required by UnimernetModel.")

        model_path = config._name_or_path
        if transform is None :
            self.transform = UnimerSwinImageProcessor()
        else :
            self.transform = transform
        if tokenizer is None:
            self.tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(model_path))
        else :
            self.tokenizer = tokenizer
        self._post_check()
    
    def _post_check(self):
        tokenizer = self.tokenizer

        if tokenizer.tokenizer.model_max_length != self.config.decoder.max_position_embeddings:
            warnings.warn(
                f"decoder.max_position_embeddings={self.config.decoder.max_position_embeddings}," +
                f" but tokenizer.model_max_length={tokenizer.tokenizer.model_max_length}, will set" +
                f" tokenizer.model_max_length to {self.config.decoder.max_position_embeddings}.")
            tokenizer.tokenizer.model_max_length = self.config.decoder.max_position_embeddings

        assert self.config.decoder.vocab_size == len(tokenizer)
        assert self.config.decoder_start_token_id == tokenizer.bos_token_id
        assert self.config.pad_token_id == tokenizer.pad_token_id

    @classmethod
    def from_checkpoint(cls, model_path: str, model_filename: str = "pytorch_model.pth", state_dict_strip_prefix="model.model."):
        config = VisionEncoderDecoderConfig.from_pretrained(model_path)
        config._name_or_path = model_path
        config.encoder = UnimerSwinConfig(**vars(config.encoder))
        config.decoder = UnimerMBartConfig(**vars(config.decoder))

        encoder = UnimerSwinModel(config.encoder)
        decoder = UnimerMBartForCausalLM(config.decoder)
        model = cls(config, encoder, decoder)

        # load model weights
        model_file_path = os.path.join(model_path, model_filename)
        checkpoint = torch.load(model_file_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        if not state_dict:
            raise RuntimeError("state_dict is empty.")
        if state_dict_strip_prefix:
            state_dict = {
                k[len(state_dict_strip_prefix):] if k.startswith(state_dict_strip_prefix) else k: v
                for k, v in state_dict.items()
            }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(unexpected_keys) > 0:
            warnings.warn("Unexpected key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            raise RuntimeError("Missing key(s) in state_dict: {}.".format(", ".join(f'"{k}"' for k in missing_keys)))
        return model

    def forward_bak(self, samples):
        pixel_values, text = samples["image"], samples["text_input"]

        text_inputs = self.tokenizer.tokenize(text).to(pixel_values.device)
        decoder_input_ids, decoder_attention_mask = text_inputs["input_ids"], text_inputs["attention_mask"]

        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        labels = decoder_input_ids * 1
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        loss = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1],
            labels=labels[:, 1:],
        ).loss
        return {"loss": loss}

    def generate(self, samples, do_sample: bool = False, temperature: float = 0.2, top_p: float = 0.95, batch_size=64):
        pixel_values = samples["image"]
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        
        kwargs = {}
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        if self.tokenizer.tokenizer.model_max_length > 1152:
            if batch_size <= 32:
                self.tokenizer.tokenizer.model_max_length = 1152  # 6g
            else:
                self.tokenizer.tokenizer.model_max_length = 1344  # 8g

        outputs = super().generate(
            pixel_values=pixel_values,
            max_new_tokens=self.tokenizer.tokenizer.model_max_length, # required
            decoder_start_token_id=self.tokenizer.tokenizer.bos_token_id,
            do_sample=do_sample,
            **kwargs,
        )

        outputs = outputs[:, 1:].cpu().numpy()
        pred_tokens = self.tokenizer.detokenize(outputs)
        pred_str = self.tokenizer.token2str(outputs)
        fixed_str = [latex_rm_whitespace(s) for s in pred_str]
        return {"pred_ids": outputs, "pred_tokens": pred_tokens, "pred_str": pred_str, "fixed_str": fixed_str}

    def parser_result(self, outputs) :
        pred_str = self.tokenizer.token2str(outputs)
        fixed_str = [latex_rm_whitespace(s) for s in pred_str]
        return fixed_str

    def generate_enc(self, pixel_values, return_dict=False):
        pixel_values = pixel_values.repeat(1, 3, 1, 1)
        encoder_outputs, _ = self.encoder(pixel_values=pixel_values, return_dict=return_dict)
        enc_kv_cache = []
        for i, layer in enumerate(self.decoder.model.decoder.layers):
            key_values = layer.enc_key_values(encoder_outputs)
            enc_kv_cache.append(key_values)
        return encoder_outputs, enc_kv_cache

    def generate_dec(self, encoder_outputs, enc_kv_cache):           
        outputs = super().generate(
            encoder_outputs=encoder_outputs,
            decoder_enc_past_key_values=enc_kv_cache,
        )
        return outputs

    def generate_data(self, pixel_values):
        encoder_outputs, enc_kv_cache = self.generate_enc(pixel_values)
        outputs1 = self.generate_dec(encoder_outputs, enc_kv_cache)
        outputs1 = outputs1[:, 1:]
        return outputs1
        
        pixel_values = pixel_values.repeat(1, 3, 1, 1)
        outputs = super().generate(
            pixel_values=pixel_values,
            # max_new_tokens=self.tokenizer.tokenizer.model_max_length, # required
            # decoder_start_token_id=self.tokenizer.tokenizer.bos_token_id,
            # do_sample=False,
        )
        outputs = outputs[:, 1:]
        print(f"### OUT generate: outputs={outputs1.shape}")
        print(f"### OUT original: outputs={outputs.shape}")
        print(f"{outputs==outputs1}")
        return outputs


# from .ov_convert_utils import patch_stateful, cleanup_torchscript_cache
# import numpy as np
# from pathlib import Path
# import openvino as ov

# UnimernetModelENC_PATH = "unimernet-enc-openvino.xml"
# UnimernetModelDEC_PATH = "unimernet-dec-openvino.xml"
# UnimernetModelToken_PATH = "unimernet-token-openvino.xml"

# class UnimernetModel_ov :
#     def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size=1):
#         # print(f"### Initializing UnimernetModel_ov with model_path={model_path}, enc_type={enc_type}, dec_type={dec_type}, cache_size={cache_size}")
#         ov_path = Path(model_path)
#         self.transform = UnimerSwinImageProcessor()
#         self.tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(model_path))

#         self.next_beam_idx = None
#         self.infer_mode = 0
#         self.bos_token_id = 0
#         self.pad_token_id = 1
#         self.eos_token_id = 2
#         self.maxlen = 32
#         self._past_length = 0
        
#         self.torch_model = None
#         self.converted_to_ov = False
#         self.using_ov = False

#         self.ov_core = ov_core
#         self.ov_encoder_path = ov_path / "ov_model" / UnimernetModelENC_PATH
#         self.ov_decoder_path = ov_path / "ov_model" / UnimernetModelDEC_PATH
#         if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists() :
#             self.converted_to_ov = True
#         self.enc_type = enc_type
#         self.dec_type = dec_type
#         valid_types = {"f32", "f16", "bf16"}
#         if (self.enc_type is not None and self.enc_type.lower() in valid_types
#             and self.dec_type is not None and self.dec_type.lower() in valid_types):
#             self.load_ov_model(cache_size)
#         self.torch_model = None

#     def load_ov_model(self, cache_size):
#         try :
#             if self.ov_core is None :
#                 self.ov_core = ov.Core()
#             cache_size_str = f"{cache_size}"
#             # self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
#             self.ov_core.set_property("CPU", {"ENABLE_MMAP": False})
#             ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type, 'PERFORMANCE_HINT': "LATENCY"}
#             self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
#             self.ov_encoder_request = self.ov_encoder_model.create_infer_request()
#             ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
#             self.ov_decoder_model = self.ov_core.compile_model(self.ov_decoder_path, 'CPU', ov_config)
#             self.ov_decoder_request = self.ov_decoder_model.create_infer_request()
#             self.using_ov = True
#         except Exception as e:
#             print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")

#     def eval(self):
#         if self.torch_model is not None :
#             self.torch_model.eval()  

#     def cpu(self):
#         if self.torch_model is not None :
#             self.torch_model.cpu()  

#     def generate(self, pixel_values):
#         inputs = {'pixel_values':pixel_values}
#         bs = pixel_values.shape[0]
#         # enc_kv_cache = self.ov_encoder_model(inputs)
#         self.ov_encoder_request.start_async(inputs, share_inputs=True)
#         self.ov_decoder_request.reset_state()
#         self.next_beam_idx = np.arange(bs, dtype=int)
#         input_ids = np.ones((bs,1), dtype=np.int64) * self.bos_token_id
#         unfinished_sequences = np.ones((bs,1), dtype=np.int64)
#         self.ov_encoder_request.wait()
#         next_tokens = input_ids
#         for t in range(self.maxlen):
#             self.ov_decoder_request.start_async({'input_ids': next_tokens,
#                                                  'enc_past_key_values.0.key': self.ov_encoder_request.get_output_tensor(0).data,
#                                                  'enc_past_key_values.0.value': self.ov_encoder_request.get_output_tensor(1).data,
#                                                  'enc_past_key_values.1.key': self.ov_encoder_request.get_output_tensor(2).data,
#                                                  'enc_past_key_values.1.value': self.ov_encoder_request.get_output_tensor(3).data,
#                                                  'enc_past_key_values.2.key': self.ov_encoder_request.get_output_tensor(4).data,
#                                                  'enc_past_key_values.2.value': self.ov_encoder_request.get_output_tensor(5).data,
#                                                  'enc_past_key_values.3.key': self.ov_encoder_request.get_output_tensor(6).data,
#                                                  'enc_past_key_values.3.value': self.ov_encoder_request.get_output_tensor(7).data,
#                                                  'enc_past_key_values.4.key': self.ov_encoder_request.get_output_tensor(8).data,
#                                                  'enc_past_key_values.4.value': self.ov_encoder_request.get_output_tensor(9).data,
#                                                  'enc_past_key_values.5.key': self.ov_encoder_request.get_output_tensor(10).data,
#                                                  'enc_past_key_values.5.value': self.ov_encoder_request.get_output_tensor(11).data,
#                                                  'enc_past_key_values.6.key': self.ov_encoder_request.get_output_tensor(12).data,
#                                                  'enc_past_key_values.6.value': self.ov_encoder_request.get_output_tensor(13).data,
#                                                  'enc_past_key_values.7.key': self.ov_encoder_request.get_output_tensor(14).data,
#                                                  'enc_past_key_values.7.value': self.ov_encoder_request.get_output_tensor(15).data,
#                                                  'beam_idx': self.next_beam_idx
#                                                 }, share_inputs=True)
#             self.ov_decoder_request.wait()
#             next_tokens = self.ov_decoder_request.get_output_tensor(0).data
#             self._past_length += input_ids.shape[1]
#             next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
#             input_ids = np.concatenate([input_ids, next_tokens], axis=-1)
#             unfinished_sequences = unfinished_sequences & ~(next_tokens==self.eos_token_id)
#             this_peer_finished = unfinished_sequences.max() == 0
#             if this_peer_finished :
#                 break
#         return input_ids
   
#     def inference(self, sorted_images, batch_size, tqdm_enable: bool = False) :
#         # Process batches and store results
#         mfr_res = []
#         desc_str = f"MFR Predict with OV_{self.enc_type}_{self.dec_type}"
#         for mf_img in tqdm(sorted_images, desc=desc_str, disable=not tqdm_enable):
#             mf_img = self.transform(mf_img).unsqueeze(0)
#             outputs = self.generate(mf_img)
#             mfr_res.extend(outputs)
#         mfr_res = self.parser_result(mfr_res)
#         return mfr_res

#     def parser_result(self, outputs) :
#         pred_str = self.tokenizer.token2str(outputs)
#         fixed_str = [latex_rm_whitespace(s) for s in pred_str]
#         return fixed_str

#     @torch.inference_mode()
#     def convert_ov_model(self, torch_model, pixel_values):
#         print(f"### Converting PyTorch model to OpenVINO format... pixel_values={pixel_values.shape}")
#         class ModelEncoderWrapper(torch.nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.model = model.eval()

#             def forward(self, pixel_values):
#                 with torch.no_grad():
#                     pixel_values = pixel_values.repeat(1, 3, 1, 1)
#                     encoder_outputs, _ = self.model.encoder(pixel_values=pixel_values, return_dict=False)
#                     enc_kv_cache = ()
#                     for i, layer in enumerate(self.model.decoder.model.decoder.layers):
#                         key_values = layer.enc_key_values(encoder_outputs)
#                         enc_kv_cache += (key_values)
#                     return enc_kv_cache

#         encoder_model = ModelEncoderWrapper(torch_model)
#         encoder_model.eval()
        
#         class ModelDecoderWrapper(torch.nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.model = model.eval()

#             def forward(self, input_ids, enc_past_key_values, past_key_values):
#                 with torch.no_grad():
#                     outputs = self.model.decoder(input_ids=input_ids,
#                                                  enc_past_key_values=enc_past_key_values,
#                                                  past_key_values=past_key_values,
#                                                  use_cache=True,
#                                                  return_dict=False,
#                                                  output_attentions=False,
#                                                  output_hidden_states=False)
#                     next_token_logits = outputs[0]
#                     next_tokens = torch.argmax(next_token_logits, dim=-1)
#                     return next_tokens, outputs[1]
          
#         decoder_model = ModelDecoderWrapper(torch_model)
#         decoder_model.eval()

#         enc_kv_cache = encoder_model(pixel_values)
#         enc_past_key_values = []
#         for i in range(0, len(enc_kv_cache), 2):
#             enc_past_key_values.append((enc_kv_cache[i], enc_kv_cache[i+1]))

#         seq_len = 2
#         num_pkv = len(enc_past_key_values)
#         bs = enc_past_key_values[0][0].shape[0]
#         input_ids = torch.zeros(bs,1).long()
#         tokens = decoder_model(input_ids, enc_past_key_values, None)[0]

#         if not self.ov_encoder_path.exists() :
#             example_inputs = {"pixel_values":pixel_values}
#             ov_model = ov.convert_model(encoder_model, example_input=example_inputs)
#             ov.save_model(ov_model, self.ov_encoder_path, compress_to_fp16=False)
#             print(f"✅ ModelEncoder completed {self.ov_encoder_path}")
#             del ov_model
#             cleanup_torchscript_cache()

#         if not self.ov_decoder_path.exists() :
#             past_key_values = []
#             key_value_input_names = []
#             key_value_output_names = []

#             input_names = ["input_ids"]
#             output_names = ["logits"]
#             for i in range(num_pkv):
#                 input_names.extend([f"enc_past.{i}.key", f"enc_past.{i}.value"])

#             for i in range(num_pkv):
#                 kv0 = torch.randn((bs, 16, seq_len, 24))
#                 kv1 = torch.randn((bs, 16, seq_len, 48))
#                 past_key_values.append((kv0, kv1))
#                 input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
#                 output_names.extend([f"present.{i}.key", f"present.{i}.value"])
#                 key_value_input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
#                 key_value_output_names.extend([f"present.{i}.key", f"present.{i}.value"])

#             example_inputs = {"input_ids" : input_ids,
#                               "enc_past_key_values": enc_past_key_values,
#                               "past_key_values": past_key_values,}
          
#             ov_model = ov.convert_model(decoder_model, example_input=example_inputs)

#             for inp, inp_name in zip(ov_model.inputs, input_names):
#                 inp.set_names({inp_name})

#             for out, out_name in zip(ov_model.outputs, output_names):
#                 out.set_names({out_name})

#             patch_stateful(ov_model, key_value_input_names, key_value_output_names)

#             ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
#             print(f"✅ ModelDecoder completed {self.ov_decoder_path}")
#             del ov_model
#             cleanup_torchscript_cache()
