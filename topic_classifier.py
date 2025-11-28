import joblib
import numpy as np
import ollama  
class GuardRailClassic:
    def __init__(self, model="svm"):
        self.pca = None
        self.model = None
        self._load_model(model)

    def _load_model(self, model="logistic"):
        """
        Load the PCA transformation and the specified model.
        """
        try:
            self.pca = joblib.load('./classifiers/models/pca_model.pkl')
            
            if model == "logistic":
                self.model = joblib.load('./classifiers/models/logistic_model.pkl')
            elif model == "svm":
                self.model = joblib.load('./classifiers/models/svm_model.pkl')
            else:
                raise ValueError("Invalid model specified. Choose 'logistic' or 'svm'.")
        except FileNotFoundError as e:
            print(f"Error loading model or PCA: {e}")
            raise
    

    def _embed(self,sentence):
        response = ollama.embed("nomic-embed-text", "classification: " + sentence)
        return response['embeddings'][0]

    def classify(self, sentence):
        """
        Classify a new sentence by embedding it, applying PCA, and returning the logit probability of being on topic.
        """
        if self.pca is None or self.model is None:
            raise RuntimeError("Model or PCA not loaded. Call _load_model() first.")

        # Step 1: Embed the sentence
        embedding = np.array([self._embed(sentence)])

        # Step 2: Apply PCA transformation
        embedding_pca = self.pca.transform(embedding)

        # Step 3: Get the logit probability
        probability = self.model.predict_proba(embedding_pca)[:, 1]  # Probability of the positive class

        return probability.item()


    

if __name__ == '__main__':
    clas = GuardRailClassic()
    print("READY")
    while True:
        q = input()
        ret = clas.classify(q)
        print(f" classified as {ret}")