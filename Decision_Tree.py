import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

veri = {
    "Yas": [25, 32, 47, 51, 23, 36, 52, 44, 33, 45], 
    "Gelir_Duzeyi": ["Yüksek", "Orta", "Orta", "Yüksek", "Düşük", "Orta", "Düşük", "Yüksek", "Düşük", "Orta"],
    "Cinsiyet": ["Erkek", "Kadın", "Kadın", "Erkek", "Erkek", "Kadın","Erkek", "Kadın", "Kadın", "Erkek"],
    "Websitesinde_Gecirilen_Sure": [5, 8, 10, 15, 3, 7, 2, 12, 4, 9],
    "Urun_Fiyati": [100, 80, 90, 200, 50, 85, 40, 180, 60, 120], 
    "Satin_Alindi": [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    
}

df = pd.DataFrame(veri)

df["Gelir_Duzeyi"] = df["Gelir_Duzeyi"].map({"Düşük": 0, "Orta": 1, "Yüksek": 2})
df["Cinsiyet"] = df["Cinsiyet"].map({"Erkek": 0, "Kadın": 1})

X = df[["Yas", "Gelir_Duzeyi", "Cinsiyet", "Websitesinde_Gecirilen_Sure", "Urun_Fiyati"]]
y = df["Satin_Alindi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

dogruluk = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {dogruluk}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, class_names=["Satın Alınmadı", "Satın Alındı"], filled = True)
plt.title("Karar Ağacı")
plt.show()
