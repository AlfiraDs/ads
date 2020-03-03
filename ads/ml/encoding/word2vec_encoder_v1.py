# the model is build on top of keras

# m is the no of categories per feature
embedding_size = min(50, m + 1 / 2)

embedding_size = 3
model = models.Sequential()
model.add(Embedding(input_dim=12, output_dim=embedding_size, input_length=1, name="embedding"))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x=data_small_df['mnth'].as_matrix(), y=data_small_df['cnt_Scaled'].as_matrix(), epochs=50, batch_size=4)
