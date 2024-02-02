import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import fasttext
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


with open('veri_sirali.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# TF-IDF matrisini oluşturma
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text])

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=['Document'])

# Metni uygun şekilde işleyerek kelime sayısını bulma
words = text.split()
total_word_counts = pd.Series(words).value_counts()

# En sık geçen 20 kelimeyi bulma
top_20_words = total_word_counts.head(20)

# Sonuçları yazdırma
print("En Sık Geçen 20 Kelime ve Toplam Geçiş Sayıları:")
print(top_20_words)

# Görselleştirme
plt.figure(figsize=(10, 6))
top_20_words.plot(kind='bar', color='skyblue')
plt.title('En Sık Geçen 20 Kelimenin Toplam Geçiş Sayıları')
plt.xlabel('Kelime')
plt.ylabel('Toplam Geçiş Sayısı')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# TF-IDF matrisini oluşturma
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text])

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=['Document'])

# fastText modelini eğitme
with open('veri_sirali.txt', 'w', encoding='utf-8') as file:
    file.write(text)

model = fasttext.train_unsupervised('veri_sirali.txt', model='cbow')


# En sık geçen 20 kelimenin her biri için en benzer 5 kelimeyi bulma
similar_words_dict = {}
for word in top_20_words.index:
    similar_words = model.get_nearest_neighbors(word, k=5)
    similar_words = [word_tuple[1] for word_tuple in similar_words]  # Sadece kelime isimleri
    similar_words_dict[word] = similar_words

# Sonuçları yazdırma
print("En Sık Geçen 20 Kelimenin Her Birinin En Benzer 5 Kelimesi:")
for word, similar_words in similar_words_dict.items():
    print(f"{word}: {similar_words}")



# Dosyadan cümleleri okuma
with open("veri_sirali.txt", "r", encoding="utf-8") as file:
    all_sentences = [line.strip() for line in file]

# Rasgele 5 cümleyi seçilir
selected_indices = random.sample(range(len(all_sentences)), 5)
selected_sentences = [all_sentences[i] for i in selected_indices]

# Cümlelerin sayısını belirlenir
num_sentences = len(all_sentences)

# TF-IDF matrisini oluşturulur
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_sentences)

# Benzerlik matrisini hesaplanır
similarity_matrix = cosine_similarity(tfidf_matrix)

# Her bir seçili cümle için en benzer 3 cümle bulunur
num_similar_sentences = 3
similar_sentences_dict = {}
for i, selected_sentence in zip(selected_indices, selected_sentences):
    # Cümlenin kendisiyle olan benzerlik skoru sıfıra ayarlanır
    similarity_matrix[i, i] = 0
    # Cümlenin en benzer olduğu indeksleri bulunur
    similar_indices = np.argsort(similarity_matrix[i])[-num_similar_sentences:][::-1]
    # Her bir benzer cümleyi ve benzerlik skorlarını kaydedilir
    similar_sentences = [(similar_index, similarity_matrix[i, similar_index]) for similar_index in similar_indices]
    similar_sentences_dict[i] = (selected_sentence, similar_sentences)

# Sonuçlar yazdırılır
for i, (selected_sentence, similar_sentences) in similar_sentences_dict.items():
    print(f"Seçilen cümle ({i + 1}. cümle): {selected_sentence}")
    for j, (similar_index, similarity_score) in enumerate(similar_sentences, 1):
        print(f"  En benzer {j}. cümle: {all_sentences[similar_index]} (Benzerlik: {similarity_score:.4f}, Derlemden sıra numarası: {similar_index + 1})")
    print()




# NLTK'nin sentiment analyzer yükle
nltk.download('vader_lexicon')


def duygu_analizi_metni(text):
    sid = SentimentIntensityAnalyzer()
    lines = text.split('\n')

    duygu_gruplari = {'Pozitif': 0, 'Negatif': 0, 'Nötr': 0}

    duygu_gruplarinin_sozlugu = {'Pozitif': [], 'Negatif': [], 'Nötr': []}

    for i, line in enumerate(lines, start=1):
        sentiment_scores = sid.polarity_scores(line)
        duygu = analiz_sonucu(sentiment_scores)

        duygu_gruplari[duygu] += 1
        duygu_gruplarinin_sozlugu[duygu].append((i, line.strip()))

    return duygu_gruplari, duygu_gruplarinin_sozlugu


def analiz_sonucu(sentiment_scores):
    if sentiment_scores['compound'] >= 0.05:
        return "Pozitif"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negatif"
    else:
        return "Nötr"



dosya_adi = 'veri_sirali.txt'

try:
    with open(dosya_adi, 'r', encoding='utf-8') as dosya:
        metin = dosya.read()
        gruplar, gruplu_satirlar = duygu_analizi_metni(metin)

        # Satır bazında duygu grupları
        for duygu, satirlar in gruplu_satirlar.items():
            print(f"{duygu} satırlar:")
            for satir in satirlar:
                print(f"  Satır {satir[0]}: {satir[1]}")
            print('\n')

        # Grafik için veriler gruplanır
        duygular = list(gruplar.keys())
        satir_sayilari = list(gruplar.values())

        # Çubuk grafik oluşturulur
        plt.bar(duygular, satir_sayilari, color=['green', 'red', 'gray'])
        plt.xlabel('Duygu Durumu')
        plt.ylabel('Toplam Satır Sayısı')
        plt.title('Metin Duygu Analizi')


        plt.show()

except FileNotFoundError:
    print(f"{dosya_adi} adında bir dosya bulunamadı.")
except Exception as e:
    print(f"Bir hata oluştu: {str(e)}")





