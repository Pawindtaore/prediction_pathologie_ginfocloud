import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd


model_name = "Modele de prediction de pathologies au Burkina Faso"
version = "v1.0.0"
algorithmes = ["K-Nearest Neighbors (KNN)","Decision Tree","Random Forest","Bagging","Logistic Regression","Naive Bayes"]

# Input for data validation
class Input(BaseModel):
    Age: int = Field()
    Sexe: str  = Field()
    Poids: float = Field()
    Symptome_1: str = Field()
    Symptome_2: str = Field()
    Symptome_3:  str = Field()
    Symptome_4: str = Field()
    Symptome_5: str = Field()
    

    class Config:
        json_schema_extra  = {
            'Age': 0, 
            'Sexe': 'string', 
            'Poids': 0.0, 
            'Symptome_1': 'string', 
            'Symptome_2': 'string', 
            'Symptome_3': 'string', 
            'Symptome_4': 'string', 
            'Symptome_5': 'string', 
        }



# Ouput for data validation
class Output(BaseModel):
    Pathologie: str
    # Details: dict



app = FastAPI()
load_models={}

@app.on_event("startup")
def lmodels_loading():
    # Chargement des modeles
    for al in algorithmes:
        current_file = f'modele_{al}.joblib'
        chemin_complet = current_file
        modele_charge = joblib.load(chemin_complet)
        load_models.__setitem__(al,modele_charge)
        
@app.get('/')
def index():
    return {'message':'Salut tout le monde'}

@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }

@app.post('/predict',response_model= Output)
async def predict_disease(data : Input):
    """Retourne la pathologie predite et les details de chaque model"""
    patient = {
        "Age (en mois)" : int(data.Age),	
        "Sexe" : data.Sexe,
        "Poids (kg)" : float(data.Poids),	
        "Symptome_5" : data.Symptome_5,		
        "Symptome_4" : data.Symptome_4,	
        "Symptome_3" : data.Symptome_3,	
        "Symptome_2" : data.Symptome_2,	
        "Symptome_1" : data.Symptome_1,
    }
    details_prediction = predict_for_all(load_models,patient)
    classe_predite = get_best_prediction(details_prediction)
    return {
        'Pathologie':classe_predite,
        # 'Details':details_prediction.to_dict()
    }
 
def predict_for_all(models,data):
    classes_pred=[]
    probas_pred=[]

    for algo, model in models.items():
        classe,proba = model.predict_disease(data)
        if classe:
            classes_pred.append(classe)
            probas_pred.append(proba)
        else:
            return None
    predictions = pd.DataFrame({"Algorithme": algorithmes, "Pathologie prédite": classes_pred, "Probabilités" : probas_pred})
    return predictions


def get_best_prediction(results):
    
    # Comptage du nombre de modèles qui prédisent chaque classe
    predictions_count = {}
    seuil_decision = 3

    # Créer un dictionnaire pour stocker les probabilités de chaque classe
    classe_probabilities = {}

    for index, row in results.iterrows():
        classe_predite = row['Pathologie prédite']
        probabilite = row['Probabilités'] 
        if classe_predite not in predictions_count:
            predictions_count[classe_predite] = 0
            classe_probabilities[classe_predite] = []
        predictions_count[classe_predite] += 1
        classe_probabilities[classe_predite].append(probabilite)

    # Sélection de la prédiction finale
    predictions_finales = []
    for classe, count in predictions_count.items():
        if count >= seuil_decision:
            moyenne_probabilites = sum(classe_probabilities[classe]) / count
            if moyenne_probabilites >= 0.75:
                predictions_finales.append((classe, count))

    # Trie des prédictions finales par nombre de prédictions (du plus élevé au plus bas)
    predictions_finales.sort(key=lambda x: x[1], reverse=True)

    # La classe la plus fréquente parmi les prédictions finales est la prédiction finale
    if predictions_finales:
        classe_predite_finale = predictions_finales[0][0]
        return classe_predite_finale
    else:
        return None

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
    