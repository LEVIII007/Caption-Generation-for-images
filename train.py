{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"nvidiaTeslaT4","dataSources":[{"sourceId":1111676,"sourceType":"datasetVersion","datasetId":623289},{"sourceId":7486575,"sourceType":"datasetVersion","datasetId":4358540}],"dockerImageVersionId":30636,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":true}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:22:20.198303Z\",\"iopub.execute_input\":\"2024-03-07T17:22:20.198774Z\",\"iopub.status.idle\":\"2024-03-07T17:22:31.151192Z\",\"shell.execute_reply.started\":\"2024-03-07T17:22:20.198733Z\",\"shell.execute_reply\":\"2024-03-07T17:22:31.150291Z\"}}\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport os \nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nimport pickle\nimport numpy as np\nfrom tqdm.notebook import tqdm\n\nfrom tensorflow.keras.applications.vgg16 import VGG16, preprocess_input\nfrom tensorflow.keras.preprocessing.image import load_img, img_to_array\nfrom tensorflow.keras.preprocessing.text import Tokenizer\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.utils import to_categorical, plot_model\nfrom tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:22:31.153483Z\",\"iopub.execute_input\":\"2024-03-07T17:22:31.154418Z\",\"iopub.status.idle\":\"2024-03-07T17:22:31.159270Z\",\"shell.execute_reply.started\":\"2024-03-07T17:22:31.154376Z\",\"shell.execute_reply\":\"2024-03-07T17:22:31.158297Z\"}}\nBASE_DIR = '/kaggle/input/flickr8k'\nWORKING_DIR = '/kaggle/working'\n\n# %% [markdown]\n# ## Extracting Image Features\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:22:31.160941Z\",\"iopub.execute_input\":\"2024-03-07T17:22:31.161518Z\",\"iopub.status.idle\":\"2024-03-07T17:22:38.294864Z\",\"shell.execute_reply.started\":\"2024-03-07T17:22:31.161478Z\",\"shell.execute_reply\":\"2024-03-07T17:22:38.292633Z\"}}\nmodel = VGG16()\n# This sets the output layer for the new model. It's extracting the output from the second-to-last layer of the existing model.\n# i.e We are leaving the \"Prediction (Dense)\" layer of the VGG16 model\nmodel = Model(inputs=model.inputs, outputs=model.layers[-2].output)\nprint(model.summary())\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:22:38.295971Z\",\"iopub.execute_input\":\"2024-03-07T17:22:38.296230Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.416014Z\",\"shell.execute_reply.started\":\"2024-03-07T17:22:38.296207Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.415070Z\"}}\n# Define a dictionary to store image features\nfeatures = {}\n\n# Specify the directory containing the images\ndirectory = os.path.join(BASE_DIR, 'Images')\n\n# Iterate through each image in the directory\nfor img_name in tqdm(os.listdir(directory)):\n    # Load the image from file\n    img_path = directory + '/' + img_name\n    image = load_img(img_path, target_size=(224, 224))\n\n    # Convert image pixels to numpy array\n    image = img_to_array(image)\n\n    # Reshape data for the model\n    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))\n\n    # Preprocess image for VGG\n    image = preprocess_input(image)\n\n    # Extract features using the pre-trained model\n    feature = model.predict(image, verbose=0)\n\n    # Get image ID from the filename\n    image_id = img_name.split('.')[0]\n\n    # Store extracted features in the dictionary\n    features[image_id] = feature\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.418395Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.418682Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.648304Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.418655Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.647296Z\"}}\n# Store features in pickle\npickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))\n\n# Load features from pickle\n# with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:\n#     features = pickle.load(f)\n\n# %% [markdown]\n# ## Loading Captions Data\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.649500Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.649810Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.703725Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.649783Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.702750Z\"}}\nwith open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:\n    next(f)\n    captions_doc = f.read()\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.705030Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.705315Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.846275Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.705290Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.845290Z\"}}\n# Create an empty dictionary to store the mapping of image IDs to captions\nmapping = {}\n\n# Iterate through each line in the 'captions_doc'\nfor line in tqdm(captions_doc.split('\\n')):\n    # Split the line by comma(,)\n    tokens = line.split(',')\n    \n    # Check if the line has enough elements (at least 2)\n    if len(line) < 2:\n        continue\n    # Extract the image ID and caption from the tokens    \n    image_id, caption = tokens[0], tokens[1:]\n    \n    # Remove extension from image ID\n    image_id = image_id.split('.')[0]\n    \n    # Convert the caption list to a string by joining its elements\n    caption = \" \".join(caption)\n    \n    # Create a list if the image ID is not already in the mapping dictionary\n    if image_id not in mapping:\n        mapping[image_id] = []\n    \n    # Store the caption in the list associated with the image ID\n    mapping[image_id].append(caption)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.847441Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.847712Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.854375Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.847677Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.853508Z\"}}\nlen(mapping)\n\n# %% [markdown]\n# ## Text Preprocessing\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.855504Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.855929Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.867619Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.855897Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.866630Z\"}}\ndef clean(mapping):\n    \n    # Iterate through each key-value pair in the 'mapping' dictionary\n    for key, captions in mapping.items():\n        \n        # Iterate through each caption associated with the current key\n        for i in range(len(captions)):\n            # Take one caption at a time\n            caption = captions[i]\n        \n            # Preprocessing steps for the caption:\n\n            # Convert the caption to lowercase\n            caption = caption.lower()\n            \n            # delete digits, special chars, etc., \n            caption = caption.replace('[^A-Za-z]', '')\n            # delete additional spaces\n            caption = caption.replace('\\s+', ' ')\n            \n            # add start and end tags to the caption\n            caption = 'startseq ' + \" \".join([word for word in caption.split() if len(word)>1]) + ' endseq'\n            captions[i] = caption\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.868867Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.869187Z\",\"iopub.status.idle\":\"2024-03-07T17:33:48.879208Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.869162Z\",\"shell.execute_reply\":\"2024-03-07T17:33:48.878342Z\"}}\n# before preprocess of text\nmapping['1000268201_693b08cb0e']\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:48.880304Z\",\"iopub.execute_input\":\"2024-03-07T17:33:48.880574Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.047973Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:48.880550Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.047028Z\"}}\n# preprocess the text\nclean(mapping)\n\n# after preprocess of text\nmapping['1000268201_693b08cb0e']\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.049061Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.049351Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.067995Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.049326Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.067174Z\"}}\nall_captions = []\nfor key in mapping:\n    for caption in mapping[key]:\n        all_captions.append(caption)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.069341Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.069727Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.075441Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.069675Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.074495Z\"}}\nprint(len(all_captions))\nprint()\nprint(all_captions[:10])\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.082089Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.082375Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.861235Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.082350Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.860394Z\"}}\n# Create a tokenizer object to handle text tokenization\ntokenizer = Tokenizer()\n\n# Fit the tokenizer on the provided text data (all_captions)\n# This process:\n#   - Analyzes the text to identify unique words (vocabulary)\n#   - Assigns a unique integer ID to each word\ntokenizer.fit_on_texts(all_captions)\n\n# Calculate the vocabulary size (number of unique words) plus 1\n# The extra 1 accounts for the 0-th index, which is often reserved for padding\nvocab_size = len(tokenizer.word_index) + 1\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:41:17.429579Z\",\"iopub.execute_input\":\"2024-03-07T18:41:17.430313Z\",\"iopub.status.idle\":\"2024-03-07T18:41:17.442734Z\",\"shell.execute_reply.started\":\"2024-03-07T18:41:17.430277Z\",\"shell.execute_reply\":\"2024-03-07T18:41:17.441916Z\"}}\npickle.dump(tokenizer, open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb'))\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.862230Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.862509Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.868376Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.862484Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.867453Z\"}}\nvocab_size\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.869642Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.870014Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.918862Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.869980Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.917966Z\"}}\n# Get maximum length of the caption available\nmax_length = max(len(caption.split()) for caption in all_captions)\nmax_length\n\n# %% [markdown]\n# ## Train Test Split\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.919880Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.920167Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.925305Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.920131Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.924400Z\"}}\nimage_ids = list(mapping.keys())\nsplit = int(len(image_ids) * 0.90)\ntrain = image_ids[:split]\ntest = image_ids[split:]\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.926379Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.926650Z\",\"iopub.status.idle\":\"2024-03-07T17:33:49.939535Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.926626Z\",\"shell.execute_reply\":\"2024-03-07T17:33:49.938758Z\"}}\n# Create data generator to get data in batch (avoids session crash)\ndef data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):\n    \n    # Initialize empty lists to store data\n    X1, X2, y = list(), list(), list()\n    \n    n = 0 # Counter for tracking batch size\n    \n    while 1: # Infinite Loop to yield batches\n        \n        for key in data_keys: # Iterate through each image key\n            n += 1\n            \n            captions = mapping[key]  # Retrieve captions associated with the image\n            \n            # process each caption\n            for caption in captions:\n               \n            # Tokenize the caption into a sequence of integer IDs\n                seq = tokenizer.texts_to_sequences([caption])[0]\n                \n                # Split the sequence into multiple X, y pairs for training\n                for i in range(1, len(seq)):\n                    # split into input and output pairs\n                    in_seq, out_seq = seq[:i], seq[i]\n                    # pad input sequence\n                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]\n                    # One-Hot Encode output sequence\n                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]\n                    \n                    # store the sequences\n                    X1.append(features[key][0])\n                    X2.append(in_seq)\n                    y.append(out_seq)\n                    \n            # If a batch is complete, yield it and reset        \n            if n == batch_size:\n                X1, X2, y = np.array(X1), np.array(X2), np.array(y) # Convert to NumPy arrays\n                yield [X1, X2], y\n                X1, X2, y = list(), list(), list()  # Reset lists for the next batch\n                n = 0  # Reset counter\n\n# %% [markdown]\n# ## Modelling\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:49.940818Z\",\"iopub.execute_input\":\"2024-03-07T17:33:49.941190Z\",\"iopub.status.idle\":\"2024-03-07T17:33:51.643874Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:49.941158Z\",\"shell.execute_reply\":\"2024-03-07T17:33:51.642838Z\"}}\n# Encoder Layers:-\n\n# Image feature layers:\ninputs1 = Input(shape=(4096,)) # Input for image features (4096 dimensions)\nfe1 = Dropout(0.2)(inputs1) # Apply dropout for regularization\nfe2 = Dense(256, activation='elu')(fe1) # Dense layer with ReLU activation\n\n# Sequence feature layers:\ninputs2 = Input(shape=(max_length,)) # Input for text sequences\nse1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) # Word embedding layer\nse2 = Dropout(0.2)(se1)\nse3 = LSTM(256)(se2) # LSTM layer to model sequential information\n\n# Decoder layers:-\ndecoder1 = add([fe2, se3]) # Combine image and sequence features\ndecoder2 = Dense(256, activation='elu')(decoder1)\noutputs = Dense(vocab_size, activation='softmax')(decoder2) # Output layer with softmax for word probabilities\n\nmodel = Model(inputs=[inputs1, inputs2], outputs=outputs) # Define model with two inputs and one output\nmodel.compile(loss='categorical_crossentropy', optimizer='adam')\n\n# Plot the model\nplot_model(model, show_shapes=True)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T17:33:51.645281Z\",\"iopub.execute_input\":\"2024-03-07T17:33:51.645659Z\",\"iopub.status.idle\":\"2024-03-07T18:16:17.757645Z\",\"shell.execute_reply.started\":\"2024-03-07T17:33:51.645624Z\",\"shell.execute_reply\":\"2024-03-07T18:16:17.756844Z\"}}\n# train the model\nepochs = 50\nbatch_size = 32\nsteps = len(train) // batch_size\n\nfor i in range(epochs):\n    # create data generator\n    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)\n    # fit for one epoch\n    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:16:17.758925Z\",\"iopub.execute_input\":\"2024-03-07T18:16:17.759227Z\",\"iopub.status.idle\":\"2024-03-07T18:16:17.890126Z\",\"shell.execute_reply.started\":\"2024-03-07T18:16:17.759201Z\",\"shell.execute_reply\":\"2024-03-07T18:16:17.889039Z\"}}\n# save the model\nmodel.save(WORKING_DIR+'/img_caption_model.h5')\n\n# %% [markdown]\n# ## Generate Captions for Image\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:16:17.891279Z\",\"iopub.execute_input\":\"2024-03-07T18:16:17.891543Z\",\"iopub.status.idle\":\"2024-03-07T18:16:17.897125Z\",\"shell.execute_reply.started\":\"2024-03-07T18:16:17.891519Z\",\"shell.execute_reply\":\"2024-03-07T18:16:17.896277Z\"}}\ndef idx_to_word(integer, tokenizer):\n    \"\"\"\n    Converts a numerical token ID back to its corresponding word using a tokenizer.\n\n    Args:\n        integer: The integer ID representing the word.\n        tokenizer: The tokenizer object that was used to tokenize the text.\n\n    Returns:\n        The word corresponding to the integer ID, or None if the ID is not found.\n    \"\"\"\n\n    # Iterate through the tokenizer's vocabulary\n    for word, index in tokenizer.word_index.items():\n        # If the integer ID matches the index of a word, return the word\n        if index == integer:\n            return word\n\n    # If no matching word is found, return None\n    return None\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:43:24.495084Z\",\"iopub.execute_input\":\"2024-03-07T18:43:24.495446Z\",\"iopub.status.idle\":\"2024-03-07T18:43:24.503762Z\",\"shell.execute_reply.started\":\"2024-03-07T18:43:24.495419Z\",\"shell.execute_reply\":\"2024-03-07T18:43:24.502858Z\"}}\n# generate caption for an image\ndef predict_caption(model, image, tokenizer, max_length):\n    \n#         \"\"\"\n#     Generates a caption for an image using a trained image captioning model.\n\n#     Args:\n#         model: The trained image captioning model.\n#         image: The image to generate a caption for.\n#         tokenizer: The tokenizer used to convert text to numerical sequences.\n#         max_length: The maximum length of the generated caption.\n\n#     Returns:\n#         The generated caption as a string.\n#     \"\"\"\n        \n    # add start tag for generation process\n    in_text = 'startseq'\n    # iterate over the max length of sequence\n    for i in range(max_length):\n        \n       # Tokenize the current caption into a sequence of integers\n        sequence = tokenizer.texts_to_sequences([in_text])[0]\n        \n        # pad the sequence\n        sequence = pad_sequences([sequence], max_length)\n       \n        # predict next word\n        yhat = model.predict([image, sequence], verbose=0)\n       \n        # Get the index of the word with the highest probability\n        yhat = np.argmax(yhat)\n        \n        # convert index to word\n        word = idx_to_word(yhat, tokenizer)\n        \n        # stop if word not found\n        if word is None:\n            break\n        # append word as input for generating next word\n        in_text += \" \" + word\n        # stop if we reach end tag\n        if word == 'endseq':\n            break\n      \n    return in_text\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:43:26.366828Z\",\"iopub.execute_input\":\"2024-03-07T18:43:26.367496Z\",\"iopub.status.idle\":\"2024-03-07T18:44:10.080727Z\",\"shell.execute_reply.started\":\"2024-03-07T18:43:26.367466Z\",\"shell.execute_reply\":\"2024-03-07T18:44:10.079255Z\"}}\nfrom nltk.translate.bleu_score import corpus_bleu\n# validate with test data\nactual, predicted = list(), list()\n\nfor key in tqdm(test):\n    # get actual caption\n    captions = mapping[key]\n    # predict the caption for image\n    y_pred = predict_caption(model, features[key], tokenizer, max_length) \n    # split into words\n    actual_captions = [caption.split() for caption in captions]\n    y_pred = y_pred.split()\n    # append to the list\n    actual.append(actual_captions)\n    predicted.append(y_pred)\n    \n# calcuate BLEU score\nprint(\"BLEU-1: %f\" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))\nprint(\"BLEU-2: %f\" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))\n\n# %% [markdown]\n# ## Testing Model\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:14.401291Z\",\"iopub.execute_input\":\"2024-03-07T18:44:14.402241Z\",\"iopub.status.idle\":\"2024-03-07T18:44:14.409001Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:14.402197Z\",\"shell.execute_reply\":\"2024-03-07T18:44:14.407966Z\"}}\nfrom PIL import Image\nimport matplotlib.pyplot as plt\ndef generate_caption(image_name):\n    # load the image\n    # image_name = \"1001773457_577c3a7d70.jpg\"\n    image_id = image_name.split('.')[0]\n    img_path = os.path.join(BASE_DIR, \"Images\", image_name)\n    image = Image.open(img_path)\n    captions = mapping[image_id]\n    print('---------------------Actual---------------------')\n    for caption in captions:\n        print(caption)\n    # predict the caption\n    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)\n    print('--------------------Predicted--------------------')\n    print(y_pred)\n    plt.imshow(image)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:15.616451Z\",\"iopub.execute_input\":\"2024-03-07T18:44:15.617197Z\",\"iopub.status.idle\":\"2024-03-07T18:44:16.504922Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:15.617162Z\",\"shell.execute_reply\":\"2024-03-07T18:44:16.504014Z\"}}\ngenerate_caption(\"1096165011_cc5eb16aa6.jpg\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:25.285666Z\",\"iopub.execute_input\":\"2024-03-07T18:44:25.286041Z\",\"iopub.status.idle\":\"2024-03-07T18:44:26.232610Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:25.286013Z\",\"shell.execute_reply\":\"2024-03-07T18:44:26.231764Z\"}}\ngenerate_caption(\"112243673_fd68255217.jpg\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:28.480114Z\",\"iopub.execute_input\":\"2024-03-07T18:44:28.480500Z\",\"iopub.status.idle\":\"2024-03-07T18:44:29.338169Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:28.480466Z\",\"shell.execute_reply\":\"2024-03-07T18:44:29.337353Z\"}}\ngenerate_caption(\"1220401002_3f44b1f3f7.jpg\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:30.338507Z\",\"iopub.execute_input\":\"2024-03-07T18:44:30.339251Z\",\"iopub.status.idle\":\"2024-03-07T18:44:31.332129Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:30.339216Z\",\"shell.execute_reply\":\"2024-03-07T18:44:31.331134Z\"}}\ngenerate_caption(\"1118557877_736f339752.jpg\")\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:32.730383Z\",\"iopub.execute_input\":\"2024-03-07T18:44:32.730761Z\",\"iopub.status.idle\":\"2024-03-07T18:44:33.604208Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:32.730731Z\",\"shell.execute_reply\":\"2024-03-07T18:44:33.603207Z\"}}\ngenerate_caption(\"1055753357_4fa3d8d693.jpg\")\n\n# %% [markdown]\n# ## Testing with New Images\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:48.948687Z\",\"iopub.execute_input\":\"2024-03-07T18:44:48.949068Z\",\"iopub.status.idle\":\"2024-03-07T18:44:48.957613Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:48.949038Z\",\"shell.execute_reply\":\"2024-03-07T18:44:48.956734Z\"}}\nimport keras\nfrom keras.applications.vgg16 import VGG16, preprocess_input\nfrom keras.preprocessing.image import load_img, img_to_array\nimport matplotlib.pyplot as plt\n\ndef new_caption(image_path, model, tokenizer, max_length):\n    \"\"\"\n    Generates a caption for an image using a trained image captioning model.\n\n    Args:\n        image_path (str): Path to the image file.\n        model: The trained image captioning model.\n        tokenizer: The tokenizer used to encode text for the model.\n        max_length: The maximum length of the generated caption.\n\n    Returns:\n        str: The generated caption.\n    \"\"\"\n\n    # Load VGG16 model and restructure it to output features from the second-to-last layer\n    vgg_model = VGG16()\n    vgg_model = keras.models.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)\n\n    # Display the image\n    img = load_img(image_path)\n    plt.imshow(img)\n\n    # Load and preprocess the image\n    image = load_img(image_path, target_size=(224, 224))\n    image = img_to_array(image)\n    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))\n    image = preprocess_input(image)\n\n    # Extract features from the image using VGG16\n    feature = vgg_model.predict(image, verbose=0)\n\n    # Generate caption using the trained model\n    caption = predict_caption(model, feature, tokenizer, max_length)\n\n    return caption\n\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:44:50.457059Z\",\"iopub.execute_input\":\"2024-03-07T18:44:50.457874Z\",\"iopub.status.idle\":\"2024-03-07T18:44:54.060542Z\",\"shell.execute_reply.started\":\"2024-03-07T18:44:50.457839Z\",\"shell.execute_reply\":\"2024-03-07T18:44:54.059662Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/sitting_bench.jpg\",model,tokenizer,max_length)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:45:05.361429Z\",\"iopub.execute_input\":\"2024-03-07T18:45:05.361800Z\",\"iopub.status.idle\":\"2024-03-07T18:45:08.563983Z\",\"shell.execute_reply.started\":\"2024-03-07T18:45:05.361771Z\",\"shell.execute_reply\":\"2024-03-07T18:45:08.563050Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/children_playing.jpg\",model,tokenizer,max_length)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:46:15.109946Z\",\"iopub.execute_input\":\"2024-03-07T18:46:15.110580Z\",\"iopub.status.idle\":\"2024-03-07T18:46:18.627664Z\",\"shell.execute_reply.started\":\"2024-03-07T18:46:15.110545Z\",\"shell.execute_reply\":\"2024-03-07T18:46:18.626735Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/cafe.jpg\",model,tokenizer,max_length)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:46:18.629427Z\",\"iopub.execute_input\":\"2024-03-07T18:46:18.629734Z\",\"iopub.status.idle\":\"2024-03-07T18:46:22.067051Z\",\"shell.execute_reply.started\":\"2024-03-07T18:46:18.629687Z\",\"shell.execute_reply\":\"2024-03-07T18:46:22.066103Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/dog-playtime-927x388.jpg\",model,tokenizer,max_length)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:46:22.068264Z\",\"iopub.execute_input\":\"2024-03-07T18:46:22.068568Z\",\"iopub.status.idle\":\"2024-03-07T18:46:25.278544Z\",\"shell.execute_reply.started\":\"2024-03-07T18:46:22.068541Z\",\"shell.execute_reply\":\"2024-03-07T18:46:25.277669Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/dirt bike.jpg\",model,tokenizer,max_length)\n\n# %% [markdown]\n# ## The model is generating decent captions. Although it is not highly accurate, but still it is able to capture details of the image and is generating text related to it. If we feed our model with more data, our caption will be more accurate (You can use Flickr30k dataset,or try out with other pretrained models like => ResNet, Inception, Xception)\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2024-03-07T18:16:17.941742Z\",\"iopub.status.idle\":\"2024-03-07T18:16:17.942185Z\",\"shell.execute_reply.started\":\"2024-03-07T18:16:17.941955Z\",\"shell.execute_reply\":\"2024-03-07T18:16:17.941977Z\"}}\nnew_caption(\"/kaggle/input/image-caption-test-v2-0/kid_smiling.jpg\",model,tokenizer,max_length)","metadata":{"_uuid":"85e4236e-20da-41bd-9962-e107669c8c8d","_cell_guid":"528564fc-7cb8-485d-a656-ca14b7d4107e","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}