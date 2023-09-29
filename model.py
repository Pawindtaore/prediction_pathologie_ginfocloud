import numpy as np
import pandas as pd


class Model :
    def __init__(self,model,encoders):
        self.model = model
        self.encoder_sexe = encoders[0]
        self.encoder_symptome_1 = encoders[1]
        self.encoder_symptome_2 = encoders[2]
        self.encoder_symptome_3 = encoders[3]
        self.encoder_symptome_4=encoders[4]
        self.encoder_symptome_5=encoders[5]
        self.encoder_pathologie = encoders[6]
    
    def __validate_date(self,X):
        return type(X) is dict
    
    def __preprocessing(self,X):
        df = pd.DataFrame([X])
        #encodage
        cols_to_encoder = df.select_dtypes(include=['object']).columns
        cols = [col!='Pathologie' for col in cols_to_encoder] #colonnes a encoder sauf la cible 
        cible = "Pathologie" #cible a encoder
        
        #Encodage des variables explicatives qualitative avec LabelBinarizer
        cat_col = list(_ for _ in cols)
        for col in cat_col:
            enc_data = None  # Initialisation de enc_data
            if col == "Symptome_5":
                enc_data = encoder_symptome_5.transform(df.loc[:, col].astype(str))
            elif col == "Symptome_4":
                enc_data = encoder_symptome_4.transform(df.loc[:, col].astype(str))
            elif col == "Symptome_3":
                enc_data = encoder_symptome_3.transform(df.loc[:, col].astype(str))
            elif col == "Symptome_2":
                enc_data = encoder_symptome_2.transform(df.loc[:, col].astype(str))
            elif col == "Symptome_1":
                enc_data = encoder_symptome_1.transform(df.loc[:, col].astype(str))
            elif col == "Sexe":
                enc_data = encoder_sexe.transform(df.loc[:, col].astype(str))

            if enc_data is not None:
                prev_mod = [f"{col}_{mod}" for mod in list(df.loc[:, col].unique())]
                if len(prev_mod) == enc_data.shape[1]:
                    df[prev_mod] = enc_data
                    df.drop(col, axis=1, inplace=True)
                else:
                    df.drop(col, axis=1, inplace=True)
                    df[col] = enc_data

                
        #Encodage de la variable cible avec LabelEncoder
        enc_cible = self.encoder_pathologie.transform(df.loc[:,cible])
        df.drop(cible,axis=1,inplace=True)
        df[cible] = enc_cible
        return df
    
    def predict(self,X):
        if not self.__validate_date(X):
            raise Exception('Données non valides')
        
        enc_data = self.__preprocessing(X)
        probas = self.model.predict_proba(enc_data)
        label_ind, proba = probas.argmax(), probas.max()
        classe_encodee = self.model.classes_[label_ind]
        classe_reelle = self.encoder_pathologie.inverse_transform([classe_encodee])[0]
        return classe_reelle,proba
    