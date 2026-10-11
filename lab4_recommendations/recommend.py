"""Лаба 4. Оценка пользователя товару.

X — UserId и ProductId, y — Score.
Модель 1: матричная факторизация (dot эмбеддингов), loss MSE.
Модель 2: нейросетевая факторизация (dropout, dense, relu), loss MAE.
Данные: data/Reviews.csv (Amazon Fine Food Reviews), урезанная выборка.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, dot, Dense, Dropout

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "Reviews.csv")

print(f"Всего записей: {len(df)}")
print(f"Количество уникальных пользователей: {df['UserId'].nunique()}")
print(f"Количество уникальных товаров: {df['ProductId'].nunique()}")
print("\nПримеры записей:")
print(df.head())
print("\nСтатистика по оценкам:")
print(df['Score'].value_counts().sort_index())

df=df.sample(frac=0.4, random_state=13)

user_counts = df['UserId'].value_counts()
item_counts = df['ProductId'].value_counts()

df = df[df['UserId'].isin(user_counts[user_counts >= 3].index)]
df = df[df['ProductId'].isin(item_counts[item_counts >= 3].index)]


df = df[['UserId', 'ProductId', 'Score']].dropna()

users = df.UserId.unique()
items = df.ProductId.unique()

user2idx = {u: i for i, u in enumerate(users)}
item2idx = {i: j for j, i in enumerate(items)}

df['UserId'] = df['UserId'].map(user2idx)
df['ProductId'] = df['ProductId'].map(item2idx)

train, test = train_test_split(df, test_size=0.2, random_state=42)

#----------------

n_users = len(user2idx)
n_items = len(item2idx)
n_latent_factors = 64

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(n_users, n_latent_factors)(user_input) #вектор вложений
item_embedding = Embedding(n_items, n_latent_factors)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)
sim = dot([user_vec, item_vec], axes=1)

model = Model([user_input, item_input], sim)
model.compile('adam', 'mse')

history = model.fit([train.UserId, train.ProductId], train.Score,
                    validation_data=([test.UserId, test.ProductId], test.Score),
                    epochs=4, batch_size=128)

#-------------------

user_vec = Dropout(0.3)(user_vec)
item_vec = Dropout(0.3)(item_vec)

merged = dot([user_vec, item_vec], axes=1)
dense = Dense(64, activation='relu')(merged)
drop = Dropout(0.4)(dense)
output = Dense(1, activation='relu')(drop)

nn_model = Model([user_input, item_input], output)

nn_model.compile('adam', 'mae')

history2 = nn_model.fit([train.UserId, train.ProductId], train.Score,
                        validation_data=([test.UserId, test.ProductId], test.Score),
                        epochs=5, batch_size=128)

# plt.figure(figsize=(8, 5))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model 1: Loss по эпохам')
# plt.xlabel('Эпоха')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid(True)
# plt.savefig('model1_loss_plot.png', dpi=300)
# plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Model 2: Loss по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.savefig(BASE / 'model2_loss_plot.png', dpi=300)
plt.show()