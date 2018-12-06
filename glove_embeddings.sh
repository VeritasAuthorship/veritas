if [ -e "./data/glove.6B.300d.txt" ]; then
    echo "glove embeddings already exist"
else
    echo "downloading glove embeddings..."
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.zip
    mv glove* data
    echo "finished downloading glove embeddings."
fi

if [ -e "./data/C50train" ]; then
    echo "Reuters dataset already exists"
else
    echo "downloading reuters dataset..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip
    unzip C50.zip
    mv C50* data

    echo "finished downloading reuters dataset..."
fi



