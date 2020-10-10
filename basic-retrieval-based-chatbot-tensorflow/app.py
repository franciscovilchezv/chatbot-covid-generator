from preprocessor import *
from sklearn.preprocessing import LabelEncoder


data = load_doc('dataset/intents.json')

# df1 = questions
df1 = frame_data('questions','labels',True, data)

# df2 = answers
df2 = frame_data('response','labels',False, data)

create_vocab(tokenizer, df1, 'questions')
# print(df1['questions'])
remove_stop_words(tokenizer,df1,'questions')
# print(df1['questions'])

vocab_size = len(vocab)

# print(df1)

test_list = list(df1.groupby(by='labels',as_index=False).first()['questions'])

# print(test_list)
# print(len(test_list))

# indexes of the test questions
test_index = []
for i,_ in enumerate(test_list):
    idx = df1[df1.questions == test_list[i]].index[0]
    test_index.append(idx)

# indexes of the train questions (all except the test ones)
train_index = [i for i in df1.index if i not in test_index]

# encoding questions based on vocabulary
X,vocab_size = encoder(df1,'questions')
df_encoded = pd.DataFrame(X)
df_encoded['labels'] = df1.labels

# adding the two confused questions
for i in range(0,2):
    dt = [0]*16
    dt.append('confused')
    dt = [dt]
    pd.DataFrame(dt).rename(columns = {16:'labels'})
    df_encoded = df_encoded.append(pd.DataFrame(dt).rename(columns = {16:'labels'}),ignore_index=True)

# print(df_encoded)

train_index.append(87)
test_index.append(88)

# encoding labels
lable_enc = LabelEncoder()
labl = lable_enc.fit_transform(df_encoded.labels)

# map with labels
mapper = {}
for index,key in enumerate(df_encoded.labels):
    if key not in mapper.keys():
        mapper[key] = labl[index]

# change labels in responses to numbers
df2.labels = df2.labels.map(mapper).astype({'labels': 'int32'})

# access to the rows by indexes
train = df_encoded.loc[train_index]
test = df_encoded.loc[test_index]

# print(test)

X_train = train.drop(columns=['labels'],axis=1)
y_train = train.labels
X_test = test.drop(columns=['labels'],axis=1)
y_test = test.labels

y_train =pd.get_dummies(y_train).values
y_test =pd.get_dummies(y_test).values


max_length = X_train.shape[1]

early_stopping = EarlyStopping(monitor='val_loss',patience=10)
checkpoint = ModelCheckpoint("model-v1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)
callbacks = [early_stopping,checkpoint,reduce_lr]

model = define_model(vocab_size, max_length)

history = model.fit(X_train, y_train, epochs=500, verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


from tensorflow.keras.models import load_model


model = load_model('model-v1.h5')
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')


while(True):
    text = input()
    df_input = get_text(text)

    #load artifacts 
    tokenizer_t = joblib.load('tokenizer_t.pkl')
    vocab = joblib.load('vocab.pkl')

    df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
    encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

    pred = get_pred(model,encoded_input)
    pred = bot_precausion(df_input,pred)

    response = get_response(df2,pred)
    bot_response(response)