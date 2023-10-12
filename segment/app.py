import io
import os
import sys
from pathlib import Path
import datetime
import argparse
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file

app = Flask(__name__, template_folder='templates')

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

import oneflow as torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import (  # noqa :E402
    cv2,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device,smart_inference_mode  # noqa :E402
from models.common import DetectMultiBackend  # noqa :E402
from utils.plots import Annotator, colors # noqa :E402
from utils.segment.general import process_mask # noqa :E402
from utils.augmentations import letterbox


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-seg.of", help="model path(s)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--project", type=str, default="yolov5_app", help="save results to ./project")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


opt = parse_opt()

# Load model
@smart_inference_mode()
def _load(weights, device):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    return model,device

model,device = _load(opt.weights, opt.device)
names, of = model.names, model.of
project_path = Path(opt.project)

def draw_pred(pred, proto, img, im_gpu, save_path):
    for i, det in enumerate(pred):  # per image
        im0 = img.copy()
        im0 = np.array(img)

        annotator = Annotator(im0, line_width=2, example=str(names))
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im_gpu.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im_gpu.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Mask plotting
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=im_gpu.squeeze(dim=0)
            )

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                c = int(cls)  # integer class
                label = names[c]
                annotator.box_label(xyxy, label, color=colors(c, True))
                # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

        # Save results (image with detections)
        cv2.imwrite(save_path, im0)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        with torch.no_grad():
            img_bytes = file.read()
            img_pil = Image.open(io.BytesIO(img_bytes))
            img = np.array(img_pil)
            img, ratio, pad = letterbox(img, opt.imgsz)
            im = torch.from_numpy(img).to(device)
            im = im.permute(2,0,1)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred, proto = model(im)[:2]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=1000, nm=32)

        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        filename = now_time + ".png"
        file_path = project_path / filename
        draw_pred(pred, proto, img_pil, im, file_path)

        return redirect(url_for("show_image", filename=filename))

    return render_template("index.html")

@app.route("/image/<filename>")
def show_image(filename):
    file_path = project_path / filename
    return send_file(file_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
