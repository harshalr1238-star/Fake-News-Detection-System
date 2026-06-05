import numpy as np
import pickle
import os
import re
import nltk
from django.shortcuts import render
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from .models import NewsHistory

# --- PATH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'detector', 'model.pkl')
vect_path = os.path.join(BASE_DIR, 'detector', 'tfidfvect.pkl')

# --- LOAD MODEL ---
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("ERROR:", e)
    model = None
    vectorizer = None

# --- NLP ---
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in stop_words]
    return ' '.join(text)

# --- MAIN VIEW ---
def predict_news(request):
    history = NewsHistory.objects.all().order_by('-created_at')[:6]

    if request.method == 'POST':
        input_data = request.POST.get('message', '').strip()

        if not input_data:
            return render(request, 'index.html', {'error': "Enter text", 'history': history})

        if model is None:
            return render(request, 'index.html', {'error': "Model not loaded", 'history': history})

        cleaned = clean_text(input_data)

        # 🔴 SHORT INPUT FIX
        if len(input_data.split()) < 10:
            return render(request, 'index.html', {
                'output': "Uncertain",
                'confidence': 50,
                'prob_fake': 50,
                'prob_real': 50,
                'reasoning': "Provide more detailed news content",
                'history': history
            })

        vect = vectorizer.transform([cleaned])

        val = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]

        prob_fake = round(proba[1] * 100, 2)
        prob_real = round(proba[0] * 100, 2)

        # 🔥 IMPROVED LOGIC
        THRESHOLD = 55

        if max(prob_fake, prob_real) < THRESHOLD:
            output = "Uncertain"
            confidence = max(prob_fake, prob_real)
        else:
            if val == 1:
                output = "Fake News"
                confidence = prob_fake
            else:
                output = "Real News"
                confidence = prob_real

        reasoning = f"Fake: {prob_fake}% | Real: {prob_real}%"

        NewsHistory.objects.create(
            news_text=input_data[:200],
            prediction=output,
            confidence=confidence,
            reasoning=reasoning
        )

        return render(request, 'index.html', {
            'output': output,
            'confidence': confidence,
            'prob_fake': prob_fake,
            'prob_real': prob_real,
            'reasoning': reasoning,
            'history': NewsHistory.objects.all().order_by('-created_at')[:6],
        })

    return render(request, 'index.html', {'history': history})