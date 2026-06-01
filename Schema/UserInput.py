from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,computed_field
from typing import Literal,Annotated
import pandas as pd
import numpy as np


# here the names of the columns should be exact like names of columns in the ml model ,but for the computed fields name should be different their original names should be name of  computed_fields function
class UserInput(BaseModel):
    age: Annotated[int,Field(...,gt=0,lt=120,description='Age of the user')]
    bmi:  Annotated[float,Field(...,gt=0,lt=50,description='bmi of the user')]
    income_lvl: Annotated[str,Field(...,description='income level Low,Middle,Lower-Middle,Upper-Middle,High')]
    smoking_sts: Annotated[Literal['Never','Current','Former'],Field(...,description='Never,Current,Former')]
    ed_level:Annotated[Literal['Highschool', 'Graduate', 'Postgraduate', 'No formal'],Field(...,description='No formal ,Highschool,Graduate,Postgraduate')]
    empl_status: Annotated[Literal['Employed' ,'Retired' ,'Student' ,'Unemployed'],Field(...,description='Employed,Retired,Student,Unemployed')]
    cholesterol_total: Annotated[int,Field(...,description='Total cholestrol level')]
    ldl_cholesterol: Annotated[int,Field(...,description='ldl  choleterol level')]
    triglycerides:Annotated[int,Field(...,description='triglycerides level')]
    hdl_cholesterol:Annotated[int,Field(...,description='hdl choleterol level')]
    waist_to_hip_ratio:Annotated[float,Field(...,description='waist to hip ratio')]
    alcohol_consumption_per_week:Annotated[int,Field(...,description='alchol consumption per week')]
    physical_activity_minutes_per_week:Annotated[int,Field(...,description='Physical activity per week in minutes')]
    diet_score:Annotated[float,Field(...,description='diet score ')]
    sleep_hours_per_day: Annotated[float,Field(...,gt=0,lt=13,description='Hours of sleep per day')]
    screen_time_hours_per_day: Annotated[float,Field(...,gt=0,lt=18,description='Hours of Screen time per day ')]
    systolic_bp: Annotated[int,Field(...,description='Systolic Bp')]
    diastolic_bp:Annotated[int,Field(...,description='Diastolic Bp')]
    heart_rate:Annotated[int,Field(...,description='heart rate')]
    gender:Annotated[str,Field(...,description='Gender')]
    ethnicity:Annotated[Literal['Hispanic', 'White' ,'Asian', 'Black', 'Other'],Field(...,description='Ethnicity')]
    family_history_diabetes:Annotated[int,Field(...,description='History of diabetes in family (0-No,1-Yes)')]
    hypertension_history: Annotated[int,Field(...,description='History of hypertension in family (0-No,1-Yes)')]
    cardiovascular_history: Annotated[int,Field(...,description='ldl  choleterol level (0-No,1-Yes)')]

    @computed_field
    @property
    def education_level(self)->int:
        ed=self.ed_level.lower()
        if ed=='no formal':
            return 0
        elif ed == 'highschool':
            return 1
        elif ed == 'graduate':
            return 2
        else:
            return 3
        
    @computed_field
    @property
    def smoking_status(self)->int:
        sm = self.smoking_sts.lower()
        if sm=="never":
            return 0
        elif sm =="current":
            return 2
        else:
            return 1
        
    @computed_field
    @property
    def income_level(self)->int:
        inc = self.income_lvl
        if inc.lower() == "low":
            return 0
        elif inc.lower() =="middle":
            return 1
        elif inc.lower() =="lower-middle":
            return 2
        elif inc.lower() == "upper-middle":
            return 3
        else:
            return 4
        
    @computed_field
    @property
    def employment_status(self)->int:
        emp = self.empl_status
        if emp == "Employed":
            return 0
        elif emp == "Retired":
            return 1
        elif emp == "Student":
            return 2
        else:
            return 3
    
    @computed_field
    @property
    def MAP(self)->float:
        return self.diastolic_bp + 0.33*(self.systolic_bp-self.diastolic_bp)
    
    @computed_field
    @property
    def chol_hdl_ratio(self)->float:
        return self.cholesterol_total / self.hdl_cholesterol

    @computed_field
    @property
    def ldl_hdl_ratio(self)->float:
        return self.ldl_cholesterol / self.hdl_cholesterol

        
    @computed_field
    @property
    def tg_hdl_ratio(self)->float:
        return self.triglycerides/self.hdl_cholesterol

    @computed_field
    @property
    def age_bmi(self)->float:
        return self.age * self.bmi
    
    @computed_field
    @property
    def age_waist_hip(self)->float:
        return self.age * self.waist_to_hip_ratio
    

def preprocess_features(data:dict):
    process = data.copy()
    TO_LOG_COLS = [
            'alcohol_consumption_per_week',
            'ldl_hdl_ratio',
            'chol_hdl_ratio',
            'tg_hdl_ratio',
            'physical_activity_minutes_per_week'
            ]
    training_bounds = {
        'age': {'lower': 25.0, 'upper': 76.0}, 
        'alcohol_consumption_per_week': {'lower': 0.6931471805599453, 'upper': 1.791759469228055}, 
        'physical_activity_minutes_per_week': {'lower': 2.833213344056216, 'upper': 5.720311776607412}, 
        'diet_score': {'lower': 2.5, 'upper': 9.3}, 
        'sleep_hours_per_day': {'lower': 4.9, 'upper': 9.1}, 
        'screen_time_hours_per_day': {'lower': 1.5, 'upper': 10.9}, 
        'bmi': {'lower': 19.3, 'upper': 32.6}, 
        'waist_to_hip_ratio': {'lower': 0.77, 'upper': 0.95}, 
        'systolic_bp': {'lower': 93.0, 'upper': 141.0}, 
        'diastolic_bp': {'lower': 60.0, 'upper': 91.0}, 
        'heart_rate': {'lower': 54.0, 'upper': 86.0}, 
        'cholesterol_total': {'lower': 150.0, 'upper': 225.0}, 
        'hdl_cholesterol': {'lower': 35.0, 'upper': 73.0}, 
        'ldl_cholesterol': {'lower': 61.0, 'upper': 148.0}, 
        'triglycerides': {'lower': 67.0, 'upper': 187.0}, 
        'education_level': {'lower': 0.0, 'upper': 3.0}, 
        'income_level': {'lower': 0.0, 'upper': 4.0}, 
        'smoking_status': {'lower': 0.0, 'upper': 2.0}, 
        'employment_status': {'lower': 0.0, 'upper': 3.0}, 
        'family_history_diabetes': {'lower': 0.0, 'upper': 1.0}, 
        'hypertension_history': {'lower': 0.0, 'upper': 1.0}, 
        'cardiovascular_history': {'lower': 0.0, 'upper': 1.0}, 
        'MAP': {'lower': 74.986, 'upper': 103.647}, 
        'chol_hdl_ratio': {'lower': 1.2315307483895939, 'upper': 1.8821435306963241}, 
        'ldl_hdl_ratio': {'lower': 0.6717940560893763, 'upper': 1.5626457509273415}, 
        'tg_hdl_ratio': {'lower': 0.7498425242364906, 'upper': 1.6739764335716716}, 
        'age_bmi': {'lower': 600.6, 'upper': 2160.0}, 
        'age_waist_hip': {'lower': 21.060000000000002, 'upper': 66.75}
    }
    for col in TO_LOG_COLS:
        if col in process:
            process[col] = np.log1p(process[col])

    for col, bounds in training_bounds.items():
        if col in process:
            process[col] = np.clip(process[col],bounds['lower'],bounds['upper'])

    return process
