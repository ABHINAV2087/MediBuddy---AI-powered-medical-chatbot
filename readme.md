# 🩺 MediBuddy - AI powered Medical Chatbot 

**AI-powered conversational medical assistant** built with Flask, LangChain, HuggingFace, and Pinecone. Designed to answer health-related queries by retrieving information from verified medical PDFs.

---

### 🎯 **Core Capabilities**
- **📚 Medical Knowledge Base**: Trained on verified medical documents and literature
- **🧠 Intelligent Responses**: Powered by HuggingFace Transformers and LangChain
- **🔍 Semantic Search**: Advanced vector similarity search using Pinecone
- **💬 Conversational Memory**: Maintains context throughout interactions
- **⚡ Real-time Processing**: Fast query processing and response generation

### 🛡️ **Safety & Reliability**
- **✅ Verified Sources**: Only uses curated medical literature
- **🔒 Secure Architecture**: Built with security best practices
- **📋 Contextual Awareness**: Understands medical terminology and context
- **⚠️ Disclaimer Integration**: Clear medical advice limitations


---

## 🧠 Tech Stack

| Layer        | Technology                              |
| ------------ | --------------------------------------- |
| Backend API  | Flask, Flask-CORS                       |
| LLM Engine   | LangChain + HuggingFace Transformers    |
| Vector DB    | Pinecone with Sentence Transformers     |
| Parsing PDFs | PyPDF, DirectoryLoader                  |
| Deployment   | Ready for Render, Vercel, or Dockerized |

---



## ⚙️ Setup & Run

### 1. Clone the Repo

```bash
git clone https://github.com/ABHINAV2087/MediBuddy---AI-powered-medical-chatbot.git
cd medical-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_api_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=medical-chatbot-index
HUGGINGFACE_REPO_ID=microsoft/DialoGPT-medium
```

### 4. Ingest Medical PDFs (first-time setup)

Put your medical PDFs inside the `data/` folder and run:

```bash
SETUP_PIPELINE=true python app.py
```

This will:

* Initialize the Pinecone index
* Load medical PDFs
* Create embeddings and upload to Pinecone

### 5. Run the Server

```bash
python app.py
```

Access the health endpoint at:

```
GET http://localhost:5000/health
```

---

## 🔄 API Endpoints

| Method | Route     | Description                         |
| ------ | --------- | ----------------------------------- |
| GET    | `/health` | Check API status                    |
| GET    | `/status` | Returns chatbot initialization info |
| POST   | `/chat`   | Submit a query to the chatbot       |

Sample `POST /chat` body:

```json
{
  "query": "What are the symptoms of diabetes?"
}
```

---

## 🧪 Example Use Case

> 🗨️ User: *What should I do if I have a fever for more than 3 days?*

> 🤖 Chatbot:
>
> * A persistent fever can indicate an underlying infection.
> * Consider getting a complete blood count (CBC) done.
> * Consult a healthcare professional if it continues.

---

## 📌 Notes

* This chatbot **does not replace professional medical advice**.
* It is intended as a **knowledge aid** sourced from medical literature.
* Always consult a doctor for health-related decisions.

---


---

## 🧑‍💻 Contributors

* **Abhinav Tirole** 

---

