import os
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, TextStreamer
from app.code.utils.ocr_internal.conversation import conv_templates, SeparatorStyle
from app.code.utils.ocr_internal.utils import disable_torch_init, KeywordsStoppingCriteria
from app.code.core.ocr_model import GOTQwenForCausalLM
from app.code.core.plug.blip_process import BlipImageEvalProcessor
from contextlib import nullcontext

class OcrService:
    def __init__(self, modelName):
        disable_torch_init()
        self.modelName = os.path.expanduser(modelName)

        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName, trust_remote_code=True)
        
        # Determine device based on CUDA availability
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.dtype = torch.bfloat16 # Use bfloat16 for GPU, can be float32 for CPU
        else:
            self.device = 'cpu'
            self.dtype = torch.float32 # Use float32 for CPU

        self.model = GOTQwenForCausalLM.from_pretrained(
            self.modelName,
            low_cpu_mem_usage=True,
            device_map=self.device, 
            use_safetensors=True,
            pad_token_id=151643
        ).eval()
        self.model.to(device=self.device, dtype=self.dtype)

        self.imageProcessor = BlipImageEvalProcessor(image_size=1024)
        self.imageProcessorHigh = BlipImageEvalProcessor(image_size=1024)

        # Constants from run_ocr_2.0.py
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        self.DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
        self.DEFAULT_IM_START_TOKEN = '<img>'
        self.DEFAULT_IM_END_TOKEN = '</img>'
        self.IMAGE_TOKEN_LEN = 256 # Hardcoded from run_ocr_2.0.py

    def _loadImage(self, imageInput):
        """
        Loads an image from a file path or returns it if it's already a PIL Image.
        """
        if isinstance(imageInput, Image.Image):
            return imageInput
        elif isinstance(imageInput, str):
            if imageInput.startswith('http') or imageInput.startswith('https'):
                response = requests.get(imageInput)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(imageInput).convert('RGB')
            return image
        else:
            raise ValueError("imageInput must be a file path (str) or a PIL Image object.")

    def performOcr(self, imageInput, ocrType="plain", box=None, color=None):
        image = self._loadImage(imageInput)
        w, h = image.size

        if ocrType == 'format':
            qs = 'OCR with format: '
        else:
            qs = 'OCR: '

        if box:
            bbox = eval(box) # Assuming box is a string representation of list/tuple
            if len(bbox) == 2:
                bbox[0] = int(bbox[0]/w*1000)
                bbox[1] = int(bbox[1]/h*1000)
            if len(bbox) == 4:
                bbox[0] = int(bbox[0]/w*1000)
                bbox[1] = int(bbox[1]/h*1000)
                bbox[2] = int(bbox[2]/w*1000)
                bbox[3] = int(bbox[3]/h*1000)
            qs = str(bbox) + ' ' + qs

        if color:
            qs = '[' + color + ']' + ' ' + qs

        # Use_im_start_end is set to True in ocr_model.py
        qs = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_PATCH_TOKEN * self.IMAGE_TOKEN_LEN + self.DEFAULT_IM_END_TOKEN + '\n' + qs

        conv_mode = "mpt"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = self.tokenizer([prompt])
        image_1 = image.copy() # Seems redundant, but keeping consistent with original
        image_tensor = self.imageProcessor(image)
        image_tensor_1 = self.imageProcessorHigh(image_1) # High res image processor

        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Use torch.autocast only for CUDA, otherwise run without it
        if self.device == 'cuda':
            context_manager = torch.autocast(self.device, dtype=self.dtype)
        else:
            # For CPU, just use a dummy context manager
            context_manager = nullcontext()

        with context_manager:
            output_ids = self.model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).to(self.device), image_tensor_1.unsqueeze(0).to(self.device))],
                do_sample=False,
                num_beams = 1, # Using 1 for simplicity, original was 1
                no_repeat_ngram_size = 20,
                streamer=streamer, # Streamer might output to console, for programmatic use, this needs adjustment
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
            )
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs