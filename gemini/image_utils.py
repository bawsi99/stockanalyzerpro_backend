import io
from PIL import Image
import matplotlib.pyplot as plt

class ImageUtils:
    @staticmethod
    def load_image_bytes(image_path: str) -> bytes:
        with open(image_path, 'rb') as f:
            return f.read()

    @staticmethod
    def bytes_to_image(image_data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_data))

    @staticmethod
    def figure_to_image(figure) -> Image.Image:
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(figure)
        return image 