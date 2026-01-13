import json
import os
import argparse
import sys

# --- 修复 Windows 控制台乱码问题 ---
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Get the directory of the current script (main.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the 'app' directory (parent of 'code')
app_dir = os.path.dirname(script_dir)
# Get the project root directory (parent of 'app')
project_root_dir = os.path.dirname(app_dir)

# Add the project root to sys.path
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# print("sys.path after modification:", sys.path) # This was originally uncommented, keeping it for user visibility

from app.code.utils.LogTool import LogTool
from app.code.utils.ConfigTool import ConfigTool
from app.code.utils.FileTool import FileTool
from app.code.core.OcrService import OcrService


def _write_output_to_file(output_dir_path, data, input_filename="output", mode='a'): # Added input_filename
    """Writes structured data as json to a file within the specified directory, named after the input file."""
    try:
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            LogTool.info(f"Created output directory: {output_dir_path}")

        # Construct output filename based on input filename
        base_name = os.path.basename(input_filename)
        output_filename = f"{base_name}.json"
        output_file_path = os.path.join(output_dir_path, output_filename)
        
        with open(output_file_path, mode, encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n') # ensure_ascii=False to handle non-ASCII chars
        LogTool.info(f"Output written to {output_file_path}")
    except IOError as e:
        LogTool.error(f"Failed to write to output file {output_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="OCRBrain CLI - Offline Optical Character Recognition")
    
    parser.add_argument("-i", "--input", required=True, 
                        help="Input path: a single image file, a single PDF file, or a directory containing images/PDFs.")
    parser.add_argument("-o", "--output_dir", 
                        default=os.path.join(project_root_dir, 'results'),
                        help="Output directory path for OCR results (default: project_root/out). "
                             "Results will be named as 'input_filename.json' to prevent overwrites.")
    parser.add_argument("--ocrtype", default="plain",
                        help="Specify the OCR processing type (default: 'plain'). Refer to OcrService.py for available types.")
    
    args = parser.parse_args()

    LogTool.info("=== OCRBrain CLI Start ===")
    
    # 1. 加载配置
    ConfigTool.load("appDev.yaml")
    ConfigTool.load("models.yaml")

    # 2. 获取 OCR 模型路径和下载 URL
    modelDirPath = ConfigTool.get("ocr.modelPath") # From models.yaml: "app/code/data/"
    downloadUrl = ConfigTool.get("ocr.downloadUrl") # From models.yaml: "https://huggingface.co/.../model.safetensors?download=true"
    
    if not modelDirPath or not downloadUrl:
        LogTool.error("OCR 模型路径或下载URL未在 config/models.yaml 中配置。")
        sys.exit(1)
    
    # 构造完整的模型文件路径
    # downloadUrl 格式为 https://.../model.safetensors?download=true
    # 需要提取出 model.safetensors
    fileName = os.path.basename(downloadUrl.split('?')[0])
    modelFilePath = os.path.join(modelDirPath, fileName) # e.g., "app/code/data/model.safetensors"

    # 确保模型文件存在，如果不存在则尝试下载
    if not FileTool.downloadFile(downloadUrl, modelFilePath):
        LogTool.error("模型下载或验证失败，无法启动OCR服务。", None)
        sys.exit(1)

    # 3. 初始化 OCR 服务
    LogTool.info(f"Initializing OCR Service with model directory: {modelDirPath}")
    ocrService = OcrService(modelDirPath) # 传递模型目录路径
    LogTool.info("OCR Service initialized successfully.")

    # 4. 准备文件列表
    files_to_process = []
    input_path = args.input
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    pdf_extensions = ('.pdf',)

    if not os.path.exists(input_path):
        LogTool.error(f"Input path not found: {input_path}")
        return
    
    if os.path.isdir(input_path):
        LogTool.info(f"Processing all supported files in directory: {input_path}")
        for root, _, files in os.walk(input_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                if filename.lower().endswith(image_extensions) or filename.lower().endswith(pdf_extensions):
                    files_to_process.append(file_path)
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(image_extensions) or input_path.lower().endswith(pdf_extensions):
            files_to_process.append(input_path)
        else:
            LogTool.error(f"Unsupported file type for single input: {input_path}. Supported types are images ({image_extensions}) and PDFs ({pdf_extensions}).")
            return
    else:
        LogTool.error(f"Invalid input path: {input_path}. Must be a file or a directory.")
        return

    if not files_to_process:
        LogTool.warning(f"No supported image or PDF files found in {input_path} to process.")
        return

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        LogTool.info(f"Created output directory: {args.output_dir}")

    # 5. 执行 OCR 逻辑
    for file_path in files_to_process:
        try:
            if file_path.lower().endswith(image_extensions):
                LogTool.info(f"Performing OCR on image: {file_path} with ocrtype: {args.ocrtype}")
                result = ocrService.performOcr(file_path, ocrType=args.ocrtype)
                output_data = {"input_path": file_path, "type": "image", "ocr_result": result}
                _write_output_to_file(args.output_dir, output_data, input_filename=file_path)
            
            elif file_path.lower().endswith(pdf_extensions):
                LogTool.info(f"Performing OCR on PDF: {file_path}")
                images = FileTool.pdfToImage(file_path)
                if not images:
                    LogTool.error(f"Could not convert PDF {file_path} to images.")
                    continue # Skip to next file
                
                # Prepare output file path for PDF, clear if exists
                output_base_name = os.path.basename(file_path)
                output_filename = f"{output_base_name}.json"
                output_file_path = os.path.join(args.output_dir, output_filename)
                if os.path.exists(output_file_path):
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        pass # Clear the file for new PDF results

                LogTool.info(f"--- OCR Result (PDF: {len(images)} pages) ---")
                all_pages_results = []
                for i, img in enumerate(images):
                    LogTool.info(f"Processing page {i+1} of PDF {output_base_name}... with ocrtype: {args.ocrtype}")
                    page_result = ocrService.performOcr(img, ocrType=args.ocrtype) # Pass PIL Image directly
                    all_pages_results.append({"page": i + 1, "ocr_result": page_result})
                
                output_data = {"input_path": file_path, "type": "pdf", "pages": all_pages_results}
                _write_output_to_file(args.output_dir, output_data, input_filename=file_path)
            
        except Exception as e:
            LogTool.error(f"Error processing {file_path}: {e}")

    LogTool.info("=== OCRBrain CLI Finished ===")

if __name__ == "__main__":
    main()