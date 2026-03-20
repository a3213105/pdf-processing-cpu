import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import os
import warnings
import openvino as ov
from .ov_convert_utils import patch_stateful, cleanup_torchscript_cache
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from .unimernet_hf import UnimerSwinImageProcessor, TokenizerWrapper, latex_rm_whitespace

class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image

UnimernetModelENC_PATH = "unimernet-enc-openvino.xml"
UnimernetModelDEC_PATH = "unimernet-dec-openvino.xml"
UnimernetModelToken_PATH = "unimernet-token-openvino.xml"

class UnimernetModel_ov :
    def __init__(self, ov_core, model_path, enc_type, dec_type, cache_size=1):
        ov_path = Path(model_path)
        self.transform = UnimerSwinImageProcessor()
        self.tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(model_path))

        self.next_beam_idx = None
        self.infer_mode = 0
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.maxlen = 32
        self._past_length = 0
        
        self.torch_model = None
        self.converted_to_ov = False
        self.using_ov = False

        self.ov_core = ov_core
        self.ov_encoder_path = ov_path / "ov_model" / UnimernetModelENC_PATH
        self.ov_decoder_path = ov_path / "ov_model" / UnimernetModelDEC_PATH
        if not self.ov_encoder_path.exists() or not self.ov_decoder_path.exists() :
            self.converted_to_ov = True
        self.enc_type = enc_type
        self.dec_type = dec_type
        valid_types = {"f32", "f16", "bf16"}
        if (self.enc_type is not None and self.enc_type.lower() in valid_types
            and self.dec_type is not None and self.dec_type.lower() in valid_types):
            self.load_ov_model(cache_size)
        self.torch_model = None

    def load_ov_model(self, cache_size):
        try :
            if self.ov_core is None :
                self.ov_core = ov.Core()
            cache_size_str = f"{cache_size}"
            # self.ov_core.set_property("CPU", {"CPU_RUNTIME_CACHE_CAPACITY": cache_size_str})
            self.ov_core.set_property("CPU", {"ENABLE_MMAP": False})
            ov_config = {'INFERENCE_PRECISION_HINT': self.enc_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_encoder_model = self.ov_core.compile_model(self.ov_encoder_path, 'CPU', ov_config)
            self.ov_encoder_request = self.ov_encoder_model.create_infer_request()
            ov_config = {'INFERENCE_PRECISION_HINT': self.dec_type, 'PERFORMANCE_HINT': "LATENCY"}
            self.ov_decoder_model = self.ov_core.compile_model(self.ov_decoder_path, 'CPU', ov_config)
            self.ov_decoder_request = self.ov_decoder_model.create_infer_request()
            self.using_ov = True
        except Exception as e:
            print(f"### ov load {self.ov_encoder_path} or {self.ov_decoder_path} failed, {e}")

    def eval(self):
        if self.torch_model is not None :
            self.torch_model.eval()  

    def cpu(self):
        if self.torch_model is not None :
            self.torch_model.cpu()  

    def generate(self, pixel_values):
        inputs = {'pixel_values':pixel_values}
        bs = pixel_values.shape[0]
        # enc_kv_cache = self.ov_encoder_model(inputs)
        self.ov_encoder_request.start_async(inputs, share_inputs=True)
        self.ov_decoder_request.reset_state()
        self.next_beam_idx = np.arange(bs, dtype=int)
        input_ids = np.ones((bs,1), dtype=np.int64) * self.bos_token_id
        unfinished_sequences = np.ones((bs,1), dtype=np.int64)
        self.ov_encoder_request.wait()
        next_tokens = input_ids
        for t in range(self.maxlen):
            self.ov_decoder_request.start_async({'input_ids': next_tokens,
                                                 'enc_past_key_values.0.key': self.ov_encoder_request.get_output_tensor(0).data,
                                                 'enc_past_key_values.0.value': self.ov_encoder_request.get_output_tensor(1).data,
                                                 'enc_past_key_values.1.key': self.ov_encoder_request.get_output_tensor(2).data,
                                                 'enc_past_key_values.1.value': self.ov_encoder_request.get_output_tensor(3).data,
                                                 'enc_past_key_values.2.key': self.ov_encoder_request.get_output_tensor(4).data,
                                                 'enc_past_key_values.2.value': self.ov_encoder_request.get_output_tensor(5).data,
                                                 'enc_past_key_values.3.key': self.ov_encoder_request.get_output_tensor(6).data,
                                                 'enc_past_key_values.3.value': self.ov_encoder_request.get_output_tensor(7).data,
                                                 'enc_past_key_values.4.key': self.ov_encoder_request.get_output_tensor(8).data,
                                                 'enc_past_key_values.4.value': self.ov_encoder_request.get_output_tensor(9).data,
                                                 'enc_past_key_values.5.key': self.ov_encoder_request.get_output_tensor(10).data,
                                                 'enc_past_key_values.5.value': self.ov_encoder_request.get_output_tensor(11).data,
                                                 'enc_past_key_values.6.key': self.ov_encoder_request.get_output_tensor(12).data,
                                                 'enc_past_key_values.6.value': self.ov_encoder_request.get_output_tensor(13).data,
                                                 'enc_past_key_values.7.key': self.ov_encoder_request.get_output_tensor(14).data,
                                                 'enc_past_key_values.7.value': self.ov_encoder_request.get_output_tensor(15).data,
                                                 'beam_idx': self.next_beam_idx
                                                }, share_inputs=True)
            self.ov_decoder_request.wait()
            next_tokens = self.ov_decoder_request.get_output_tensor(0).data
            self._past_length += input_ids.shape[1]
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            input_ids = np.concatenate([input_ids, next_tokens], axis=-1)
            unfinished_sequences = unfinished_sequences & ~(next_tokens==self.eos_token_id)
            this_peer_finished = unfinished_sequences.max() == 0
            if this_peer_finished :
                break
        return input_ids
   
    def inference(self, sorted_images, batch_size, tqdm_enable = False) :
        # Process batches and store results
        mfr_res = []
        desc_str = f"MFR_OV_{self.enc_type}_{self.dec_type} Predict"
        for mf_img in tqdm(sorted_images, desc=desc_str, disable=not tqdm_enable):
            mf_img = self.transform(mf_img).unsqueeze(0)
            outputs = self.generate(mf_img)
            mfr_res.extend(outputs)
        mfr_res = self.parser_result(mfr_res)
        return mfr_res

    def parser_result(self, outputs) :
        pred_str = self.tokenizer.token2str(outputs)
        fixed_str = [latex_rm_whitespace(s) for s in pred_str]
        return fixed_str

    @torch.inference_mode()
    def convert_ov_model(self, torch_model, pixel_values):  
        class ModelEncoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, pixel_values):
                with torch.no_grad():
                    pixel_values = pixel_values.repeat(1, 3, 1, 1)
                    encoder_outputs, _ = self.model.encoder(pixel_values=pixel_values, return_dict=False)
                    enc_kv_cache = ()
                    for i, layer in enumerate(self.model.decoder.model.decoder.layers):
                        key_values = layer.enc_key_values(encoder_outputs)
                        enc_kv_cache += (key_values)
                    return enc_kv_cache

        encoder_model = ModelEncoderWrapper(torch_model)
        encoder_model.eval()
        
        class ModelDecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.eval()

            def forward(self, input_ids, enc_past_key_values, past_key_values):
                with torch.no_grad():
                    outputs = self.model.decoder(input_ids=input_ids,
                                                 enc_past_key_values=enc_past_key_values,
                                                 past_key_values=past_key_values,
                                                 use_cache=True,
                                                 return_dict=False,
                                                 output_attentions=False,
                                                 output_hidden_states=False)
                    next_token_logits = outputs[0]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                    return next_tokens, outputs[1]
          
        decoder_model = ModelDecoderWrapper(torch_model)
        decoder_model.eval()

        enc_kv_cache = encoder_model(pixel_values)
        enc_past_key_values = []
        for i in range(0, len(enc_kv_cache), 2):
            enc_past_key_values.append((enc_kv_cache[i], enc_kv_cache[i+1]))

        seq_len = 2
        num_pkv = len(enc_past_key_values)
        bs = enc_past_key_values[0][0].shape[0]
        input_ids = torch.zeros(bs,1).long()
        tokens = decoder_model(input_ids, enc_past_key_values, None)[0]

        if not self.ov_encoder_path.exists() :
            example_inputs = {"pixel_values":pixel_values}
            ov_model = ov.convert_model(encoder_model, example_input=example_inputs)
            ov.save_model(ov_model, self.ov_encoder_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_encoder_path}")
            del ov_model
            cleanup_torchscript_cache()

        if not self.ov_decoder_path.exists() :
            past_key_values = []
            key_value_input_names = []
            key_value_output_names = []

            input_names = ["input_ids"]
            output_names = ["logits"]
            for i in range(num_pkv):
                input_names.extend([f"enc_past_key_values.{i}.key", f"enc_past_key_values.{i}.value"])

            for i in range(num_pkv):
                kv0 = torch.randn((bs, 16, seq_len, 24))
                kv1 = torch.randn((bs, 16, seq_len, 48))
                past_key_values.append((kv0, kv1))
                input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
                output_names.extend([f"present.{i}.key", f"present.{i}.value"])
                key_value_input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
                key_value_output_names.extend([f"present.{i}.key", f"present.{i}.value"])

            example_inputs = {"input_ids" : input_ids,
                              "enc_past_key_values": enc_past_key_values,
                              "past_key_values": past_key_values,}
          
            ov_model = ov.convert_model(decoder_model, example_input=example_inputs)

            for inp, inp_name in zip(ov_model.inputs, input_names):
                inp.set_names({inp_name})

            for out, out_name in zip(ov_model.outputs, output_names):
                out.set_names({out_name})

            patch_stateful(ov_model, key_value_input_names, key_value_output_names)

            ov.save_model(ov_model, self.ov_decoder_path, compress_to_fp16=False)
            print(f"✅ ModelEncoder completed {self.ov_decoder_path}")
            del ov_model
            cleanup_torchscript_cache()


class UnimernetModel(object):
    def __init__(self, weight_dir, cfg_path, enable_ov, enc_type, dec_type, _device_="cpu"):
        self.ov_model = UnimernetModel_ov(None, weight_dir, enc_type, dec_type)
        if enable_ov and self.ov_model.using_ov :
            self.enable_ov = True
            return 
        else :
            self.enable_ov = False
        self.infer_type = enc_type
        from .unimernet_hf import UnimernetModel
        self.model = UnimernetModel.from_pretrained(weight_dir, attn_implementation="eager")
        self.device = _device_
        self.model.to(_device_)
        if self.infer_type == "bf16":
            self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval()
        self.ov_model.torch_model = self.model
    
    def predict(self, mfd_res, image):
        formula_list = []
        mf_image_list = []
        for xyxy, conf, cla in zip(
            mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": 13 + int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 2),
                "latex": "",
            }
            formula_list.append(new_item)
            bbox_img = image[ymin:ymax, xmin:xmax]
            mf_image_list.append(bbox_img)

        dataset = MathDataset(mf_image_list, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
        mfr_res = []
        for mf_img in dataloader:
            mf_img = mf_img.to(dtype=self.model.dtype)
            mf_img = mf_img.to(self.device)
            with torch.no_grad():
                output = self.model.generate({"image": mf_img})
            mfr_res.extend(output["fixed_str"])
        for res, latex in zip(formula_list, mfr_res):
            res["latex"] = latex
        return formula_list

    @torch.inference_mode()
    def batch_predict(self, images_mfd_res: list, images: list, batch_size: int) -> list:
        sorted_images, index_mapping, backfill_list, images_formula_list = self.preprocess(images_mfd_res, images, batch_size)
        if self.enable_ov :
            mfr_res = self.ov_model.inference(sorted_images, batch_size)
        else :
            mfr_res = self.inference(sorted_images, batch_size)
        self.postprocess(mfr_res, index_mapping, backfill_list)
        return images_formula_list
    
    def preprocess(self, images_mfd_res: list, images: list, batch_size: int):
        images_formula_list = []
        mf_image_list = []
        backfill_list = []
        image_info = []  # Store (area, original_index, image) tuples

        # Collect images with their original indices
        for image_index in range(len(images_mfd_res)):
            mfd_res = images_mfd_res[image_index]
            np_array_image = images[image_index]
            formula_list = []

            for idx, (xyxy, conf, cla) in enumerate(zip(
                    mfd_res.boxes.xyxy, mfd_res.boxes.conf, mfd_res.boxes.cls
            )):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": 13 + int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": round(float(conf.item()), 2),
                    "latex": "",
                }
                formula_list.append(new_item)
                bbox_img = np_array_image[ymin:ymax, xmin:xmax]
                area = (xmax - xmin) * (ymax - ymin)

                curr_idx = len(mf_image_list)
                image_info.append((area, curr_idx, bbox_img))
                mf_image_list.append(bbox_img)

            images_formula_list.append(formula_list)
            backfill_list += formula_list
        # Stable sort by area
        image_info.sort(key=lambda x: x[0])  # sort by area
        sorted_indices = [x[1] for x in image_info]
        sorted_images = [x[2] for x in image_info]

        # Create mapping for results
        index_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted_indices)}

        return sorted_images, index_mapping, backfill_list, images_formula_list

    @torch.inference_mode()
    def inference(self, sorted_images, batch_size, tqdm_enable = False) :
        # Create dataset with sorted images
        dataset = MathDataset(sorted_images, transform=self.model.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        # Process batches and store results
        mfr_res = []
        desc_str = f"MFR_{self.infer_type} Predict"
        if self.infer_type == "bf16" :
            with tqdm(total=len(sorted_images), desc=desc_str, disable=not tqdm_enable) as pbar:
                for index, mf_img in enumerate(dataloader):
                    mf_img = mf_img.to(dtype=self.model.dtype)
                    mf_img = mf_img.to(self.device)
                    with torch.no_grad(), torch.amp.autocast('cpu'):
                        outputs = self.model.generate(mf_img)
                    outputs = outputs.cpu().numpy()
                    output = self.model.parser_result(outputs)
                    mfr_res.extend(output)
                    # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                    current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                    pbar.update(current_batch_size)
        else :
            with tqdm(total=len(sorted_images), desc=desc_str, disable=not tqdm_enable) as pbar:
                for index, mf_img in enumerate(dataloader):
                    mf_img = mf_img.to(dtype=self.model.dtype)
                    mf_img = mf_img.to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(mf_img)
                        if self.ov_model.converted_to_ov:
                            self.ov_model.convert_ov_model(self.model, mf_img)
                        outputs = outputs.cpu().numpy()
                        output = self.model.parser_result(outputs)
                    mfr_res.extend(output)
                    # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                    current_batch_size = min(batch_size, len(sorted_images) - index * batch_size)
                    pbar.update(current_batch_size)
        return mfr_res
    
    def postprocess(self, mfr_res, index_mapping, backfill_list):
        # Restore original order
        unsorted_results = [""] * len(mfr_res)
        for new_idx, latex in enumerate(mfr_res):
            original_idx = index_mapping[new_idx]
            unsorted_results[original_idx] = latex
        # Fill results back
        for res, latex in zip(backfill_list, unsorted_results):
            res["latex"] = latex
