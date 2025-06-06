# Dog Breed Classifier using Transfer Learning & Streamlit

A web app that identifies dog breeds from uploaded images using a ResNet18 model fine-tuned with transfer learning.

🚀 **Live Demo**
👉 [Try it here](https://dogbreed-classifier.streamlit.app/)

---

✨ **Features**

* 🐕 **Image Upload:** Upload a dog photo to get breed predictions.
* 🤖 **Transfer Learning:** Uses pretrained ResNet18 for accurate classification.
* 📈 **Confidence Scores:** Displays prediction confidence with results.
* ⚡ **GPU Accelerated:** Supports GPU for faster inference.
* 🖥️ **Streamlit UI:** Clean, responsive interface for easy use.

---

🛠️ **Technologies Used**

| Category      | Tools / Libraries            |
| ------------- | ---------------------------- |
| Language      | Python 3.12                  |
| Deep Learning | PyTorch, Torchvision         |
| Model         | ResNet18 (Transfer Learning) |
| UI            | Streamlit                    |
| Deployment    | Streamlit Cloud              |

---

🧠 **How It Works**

* **Image Input:** User uploads a dog image via the Streamlit app.
* **Preprocessing:** Image is resized and normalized for the model.
* **Inference:** ResNet18 predicts dog breed and confidence score.
* **Output:** Displays the predicted breed and confidence on the UI.
