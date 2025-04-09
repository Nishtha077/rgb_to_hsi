# 🌈 RGB to Hyperspectral Image Converter

Live demo: [rgbtohsi-mstpp.streamlit.app](https://rgbtohsi-mstpp.streamlit.app)

This Streamlit app converts an RGB image to a hyperspectral image (256×256×31) using a pretrained **MST++** model.

---

## 🧠 Model

- **MST++** (Multi-stage Spectral Translation++)
- Input: RGB image
- Output: Hyperspectral image (256×256×31)

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/rgb_to_hsi.git
cd rgb_to_hsi
pip install -r requirements.txt
streamlit run app.py
