import streamlit as st
import requests

API_URL_DEFAULT = "http://127.0.0.1:5001/predict"

st.set_page_config(page_title="Plant Disease Demo", layout="centered")
st.title("Plant Disease Classification Demo")

st.markdown("Upload a leaf image, choose backend, and get prediction from Flask API.")

api_url = st.text_input("Flask API URL", value=API_URL_DEFAULT)

backend = st.selectbox(
    "Backend",
    options=[
        "model1_model2",
        "global_cnn",
        "cnn_svm",
        "transfer_learning",
    ],
    index=0
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


def render_result(out: dict):
    st.subheader(" Analiz Sonuçları")

    # -------------------------------------------------
    # Case 1: model1_model2 style output
    # -------------------------------------------------
    if "plant" in out or "disease" in out or "ood" in out:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Bitki Türü", value=out.get('plant', 'Bilinmiyor'))
        with col2:
            st.metric(label="Hastalık Durumu", value=out.get('disease', 'Bilinmiyor'))
        
        if out.get('ood') == "True" or out.get('ood') is True:
            st.error(" Dikkat: Bu görsel eğitim veri setinin dışından (OOD) olabilir!")
        return

    # -------------------------------------------------
    # Case 2: transfer_learning / global / cnn_svm
    # -------------------------------------------------
    top1 = out.get("top1")
    preds = out.get("preds")

    # tahmin
    if isinstance(top1, dict):
        class_name = top1.get('class_name', 'Bilinmiyor')
        prob = top1.get('prob', 0.0)
    elif isinstance(preds, list) and len(preds) > 0:
        class_name = preds[0].get('class_name', 'Bilinmiyor')
        prob = preds[0].get('prob', 0.0)
    else:
        st.warning("Bilinmeyen çıktı formatı. Ham JSON verisini inceleyin.")
        return

    #  tahmin
    st.success(f"**En Yüksek Tahmin:** {class_name}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tahmin Edilen Sınıf", value=class_name.split("___")[-1].replace("_", " "))
    with col2:
        st.metric(label="Güven Oranı (Olasılık)", value=f"%{prob*100:.2f}")

    #  Grafik 
    if isinstance(preds, list) and len(preds) > 0:
        st.write("---")
        st.subheader(" Model Olasılık Dağılımı")
        
        # Grafik verileri
        chart_data = {
            "Sınıflar": [p.get("class_name").split("___")[-1].replace("_", " ") for p in preds],
            "Olasılık (%)": [p.get("prob", 0.0) * 100 for p in preds]
        }
        
        #  bar chart
        st.bar_chart(data=chart_data, x="Sınıflar", y="Olasılık (%)", color="#1f77b4")

if uploaded is not None:

    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):

        try:
            files = {
                "image": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type
                )
            }

            data = {"backend": backend}

            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"API error ({response.status_code}): {response.text}")
            else:
                out = response.json()
                render_result(out)

                st.subheader("Raw JSON")
                st.json(out)

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")