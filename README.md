# Web-Based Malaria Screener

A web adaptation of the archived [Malaria Screener](https://github.com/LHNCBC/MalariaScreener) Android project.

This project keeps the web boundary small and reserves the detection implementation for the original models and image-processing pipeline. It does not train a replacement model or fabricate predictions.

## Run

Requirements: Python 3.10+

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Original assets

Copy the verified assets from the upstream repository into `models/`:

- `malaria_thinsmear_44_retrainSudan_20P_4000C_separate.pb`
- `ThickSmearModel.h5.pb`
- Optional historical `.pb`/`.tflite` files
- SVM resources if the SVM path is enabled later

The current scaffold reports a clear configuration error until the active models are present. No result is produced without a model.

## Current scope

- Thin and thick smear selection
- Image upload
- Backend analysis boundary
- Original frozen-graph loading
- 44x44 RGB batch inference with the upstream thin/thick class mappings
- Bounded OpenCV candidate generation
- Detection coordinates, counts, confidences, and annotated result image

The candidate-generation stage is a web port of the detector boundary, not a claim of byte-for-byte Android parity. Validate results against the Android application with representative smear images before using it for research conclusions.

The application is research software, not a clinical diagnostic device.
