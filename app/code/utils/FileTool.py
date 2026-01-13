import os
import json
import fitz # PyMuPDF
from PIL import Image
import requests # 导入 requests 库
from utils.LogTool import LogTool

class FileTool:
    @staticmethod
    def exists(filePath):
        return os.path.exists(filePath)

    @staticmethod
    def ensureDir(filePath):
        """
        确保文件所在的目录存在
        """
        directory = os.path.dirname(filePath)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                LogTool.error(f"Failed to create directory: {directory}", e)

    @staticmethod
    def appendJsonLine(filePath, dataDict):
        """
        向文件追加一行 JSON 数据 (JSONL 格式)
        """
        try:
            FileTool.ensureDir(filePath)
            with open(filePath, 'a', encoding='utf-8') as f:
                jsonLine = json.dumps(dataDict, ensure_ascii=False)
                f.write(jsonLine + "\n")
            return True
        except Exception as e:
            LogTool.error(f"Failed to append to file: {filePath}", e)
            return False

    @staticmethod
    def pdfToImage(pdfPath, dpi=300):
        """
        将PDF文件的每一页转换为PIL Image对象列表。
        Args:
            pdfPath (str): PDF文件的路径。
            dpi (int): 渲染图像的分辨率。
        Returns:
            list: 包含每个PDF页面PIL Image对象的列表。
        """
        images = []
        try:
            document = fitz.open(pdfPath)
            for pageNumber in range(document.page_count):
                page = document.load_page(pageNumber)
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            document.close()
        except Exception as e:
            LogTool.error(f"Failed to convert PDF to image for {pdfPath}: {e}")
        return images

    @staticmethod
    def downloadFile(url, destinationPath):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destinationPath), exist_ok=True)

        if FileTool.exists(destinationPath):
            LogTool.info(f"文件已存在: {destinationPath}")
            return True

        LogTool.info(f"开始从 {url} 下载文件到 {destinationPath}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # total_size = int(response.headers.get('content-length', 0)) # 暂时不显示进度条
            # downloaded_size = 0

            with open(destinationPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # downloaded_size += len(chunk)
                        # if total_size:
                        #     progress = (downloaded_size / total_size) * 100
                        #     LogTool.info(f"下载进度: {progress:.2f}%")
            LogTool.info(f"文件下载成功: {destinationPath}")
            return True
        except requests.exceptions.RequestException as e:
            LogTool.error(f"下载文件失败: {e}", e)
            return False
        except Exception as e:
            LogTool.error(f"处理下载文件时发生未知错误: {e}", e)
            return False
