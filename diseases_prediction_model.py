import numpy as np
import pandas as pd

class DiseasesPredictionModel:
    def __init__(self, model, encoders,normalizer):
        self.model = model
        self.encoders = encoders
        self.normalizer = normalizer

    def validate_data(self, X):
        if not isinstance(X, dict):
            raise ValueError('Les données doivent être un dictionnaire')
        
        return True

    def preprocess_data(self, X):
        df = pd.DataFrame([X])
        #normalisation
        scaled_data = self.normalizer.transform(df.select_dtypes(include=['int', 'float']))
        df.loc[:, df.select_dtypes(include=['int', 'float']).columns] = scaled_data
        
        #encodage
        for col, encoder in self.encoders.items():
            if col != 'Pathologie':
                enc_data = encoder.transform(df.loc[:, col].astype(str))
                prev_mod = [f"{col}_{mod}" for mod in list(encoder.classes_)]
                if(len(prev_mod)==enc_data.shape[1]):
                    df[prev_mod] = enc_data
                    df.drop(col,axis=1,inplace=True)
                else:
                    df.drop(col,axis=1,inplace=True) 
                    df[col] = enc_data
                
        return df

    def predict_disease(self, X):
        try:
            if not self.validate_data(X):
                return None 
            preprocessed_data = self.preprocess_data(X)
            preprocessed_data = preprocessed_data.reindex(sorted(preprocessed_data.columns), axis=1)
            probas = self.model.predict_proba(preprocessed_data)
            label_ind, proba = probas.argmax(),probas.max()
            classe = self.encoders['Pathologie'].inverse_transform([label_ind])[-1]
            return classe,proba
        except Exception as e:
            raise Exception(f'Erreur lors de la prédiction : {str(e)}')

