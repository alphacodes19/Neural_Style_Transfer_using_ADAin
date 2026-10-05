import os
import uuid
import threading
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, SubmitField
from wtforms.validators import InputRequired, NumberRange

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# Application setup
# ============================================================

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "local-development-secret-change-before-deployment",
)

app.config["UPLOAD_FOLDER"] = str(
    PROJECT_ROOT / "static" / "uploads"
)

# Maximum total request size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg",
}

Bootstrap(app)

UPLOAD_DIR = Path(app.config["UPLOAD_FOLDER"])
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Forms
# ============================================================

class UploadForm(FlaskForm):

    content = FileField(
        "Content Image",
        validators=[InputRequired()],
    )

    style = FileField(
        "Style Image",
        validators=[InputRequired()],
    )

    alpha = FloatField(
        "Alpha",
        default=0.7,
        validators=[
            NumberRange(
                min=0.0,
                max=1.0,
                message="Style strength must be between 0 and 1.",
            )
        ],
    )

    submit = SubmitField("Transfer Style")


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Model paths
# ============================================================

VGG_PATH = Path(
    os.environ.get(
        "VGG_PATH",
        PROJECT_ROOT / "weights" / "vgg_normalised.pth",
    )
)

# Verified final model from the 160k training run.
DECODER_PATH = Path(
    os.environ.get(
        "DECODER_PATH",
        PROJECT_ROOT
        / "experiment"
        / "full_training_160k"
        / "best_decoder.pth",
    )
)


if not VGG_PATH.is_file():
    raise FileNotFoundError(
        f"VGG weights not found:\n{VGG_PATH}\n\n"
        "Set VGG_PATH or place vgg_normalised.pth in weights/."
    )


if not DECODER_PATH.is_file():
    raise FileNotFoundError(
        f"Decoder weights not found:\n{DECODER_PATH}\n\n"
        "Expected the verified full-training checkpoint "
        "at experiment/full_training_160k/best_decoder.pth."
    )


# ============================================================
# Load models once at startup
# ============================================================

print(f"Loading VGG weights: {VGG_PATH}")
print(f"Loading decoder weights: {DECODER_PATH}")

encoder = VGGEncoder(
    str(VGG_PATH),
    map_location=device,
).to(device)

decoder = Decoder().to(device)

decoder_state = torch.load(
    str(DECODER_PATH),
    map_location=device,
)

decoder.load_state_dict(decoder_state)

encoder.eval()
decoder.eval()


# All inference is protected by this lock.
# This is especially useful on a 6 GB GPU.
inference_lock = threading.Lock()

print("Models loaded successfully.")


# ============================================================
# Image preprocessing
# ============================================================

# This matches the preprocessing used during evaluation:
# Resize(512) -> CenterCrop(256) -> ToTensor()
inference_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


# ============================================================
# Helpers
# ============================================================

def allowed_file(filename: str) -> bool:
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def generate_filename(
    original_filename: str,
    prefix: str,
) -> str:
    """
    Generate a safe unique filename.

    Example:
        content_8f3c...abc.jpg
    """

    safe_name = secure_filename(original_filename)

    if "." not in safe_name:
        raise ValueError(
            "Uploaded file must have an extension."
        )

    extension = safe_name.rsplit(".", 1)[1].lower()

    if extension not in app.config["ALLOWED_EXTENSIONS"]:
        raise ValueError(
            "Unsupported image format."
        )

    unique_id = uuid.uuid4().hex

    return f"{prefix}_{unique_id}.{extension}"


def validate_image(file_storage) -> Image.Image:
    """
    Validate that the uploaded file is actually an image.

    The file extension alone is not trusted.
    """

    if not file_storage or not file_storage.filename:
        raise ValueError(
            "No image was uploaded."
        )

    if not allowed_file(file_storage.filename):
        raise ValueError(
            "Only PNG, JPG, and JPEG images are supported."
        )

    try:
        image = Image.open(file_storage.stream)

        # Force Pillow to read/decode the file now.
        image.verify()

        # Reset stream after verify().
        file_storage.stream.seek(0)

        image = Image.open(
            file_storage.stream
        ).convert("RGB")

        return image

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from exc


def style_transfer(
    content_image: Image.Image,
    style_image: Image.Image,
    alpha: float,
) -> torch.Tensor:

    content_tensor = inference_transform(
        content_image
    ).unsqueeze(0).to(device)

    style_tensor = inference_transform(
        style_image
    ).unsqueeze(0).to(device)

    with inference_lock:

        with torch.inference_mode():

            content_features = encoder(
                content_tensor,
                is_test=True,
            )

            style_features = encoder(
                style_tensor,
                is_test=True,
            )

            target_features = (
                adaptive_instance_normalization(
                    content_features,
                    style_features,
                )
            )

            blended_features = (
                alpha * target_features
                + (1.0 - alpha) * content_features
            )

            output = decoder(
                blended_features
            )

    return output


def tensor_to_pil(
    image_tensor: torch.Tensor,
) -> Image.Image:

    image_tensor = image_tensor.detach().cpu()

    image_tensor = image_tensor.squeeze(0)

    image_tensor = image_tensor.clamp(
        0.0,
        1.0,
    )

    return transforms.ToPILImage()(
        image_tensor
    )


def save_pil_image(
    image: Image.Image,
    filename: str,
) -> None:

    output_path = UPLOAD_DIR / filename

    image.save(
        output_path,
        format="PNG",
        optimize=True,
    )


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():

        try:

            # ------------------------------------------------
            # Validate and load uploaded images
            # ------------------------------------------------

            content_image = validate_image(
                form.content.data
            )

            style_image = validate_image(
                form.style.data
            )

            # ------------------------------------------------
            # Alpha validation
            # ------------------------------------------------

            alpha = float(form.alpha.data)

            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "Style strength must be between 0 and 1."
                )

            # ------------------------------------------------
            # Generate unique filenames
            # ------------------------------------------------

            content_filename = generate_filename(
                form.content.data.filename,
                "content",
            )

            style_filename = generate_filename(
                form.style.data.filename,
                "style",
            )

            result_filename = (
                f"stylized_{uuid.uuid4().hex}.png"
            )

            # ------------------------------------------------
            # Save uploaded images
            # ------------------------------------------------

            content_image.save(
                UPLOAD_DIR / content_filename
            )

            style_image.save(
                UPLOAD_DIR / style_filename
            )

            # ------------------------------------------------
            # Run AdaIN
            # ------------------------------------------------

            stylized_tensor = style_transfer(
                content_image,
                style_image,
                alpha,
            )

            stylized_image = tensor_to_pil(
                stylized_tensor
            )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            save_pil_image(
                stylized_image,
                result_filename,
            )

            result_image = result_filename

        except ValueError as exc:

            error = str(exc)

        except torch.cuda.OutOfMemoryError:

            if device.type == "cuda":
                torch.cuda.empty_cache()

            error = (
                "The GPU ran out of memory while processing "
                "this image. Please try smaller input images."
            )

        except Exception:

            app.logger.exception(
                "Style transfer failed"
            )

            error = (
                "Something went wrong while generating "
                "the stylized image. Please try again."
            )

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def send_image(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename,
    )


@app.route("/examples/<path:filename>")
def send_example(filename):

    examples_dir = (
        PROJECT_ROOT / "examples"
    )

    return send_from_directory(
        examples_dir,
        filename,
    )


@app.errorhandler(413)
def request_too_large(_error):

    return render_template(
        "index.html",
        form=UploadForm(),
        result_image=None,
        content_image=None,
        style_image=None,
        error=(
            "Upload is too large. Please keep the total "
            "request under 16 MB."
        ),
    ), 413


@app.route("/health")
def health():

    return {
        "status": "ok",
        "device": str(device),
        "decoder": DECODER_PATH.name,
    }


# ============================================================
# Local development entry point
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )