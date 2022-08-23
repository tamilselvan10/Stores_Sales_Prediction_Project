import os,sys
from sales.logger import logging
from sales.exception import SalesException
from sales.util import *
from collections import namedtuple
import importlib
from typing import List
from sklearn.metrics import r2_score,mean_squared_error


GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])
BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])          

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.5) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model
    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature
    return
    It retured a named tuple
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:
        
    
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Calculating r squared score on training and testing dataset
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            
            #Calculating mean squared error on training and testing dataset
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Calculating harmonic mean of train_accuracy and test_accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")


            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise SalesException(e, sys) from e

class ModelFactory:
    
    def __init__(self,model_config_file_path:str)->None:
        try:
            self.config:dict=read_yaml_file(file_path=model_config_file_path)

            self.grid_search_module:str=self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class:str=self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_params:dict=dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config:dict=dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list=None
            self.grid_searched_best_model_list=None

        except Exception as e:
            raise SalesException(e,sys) from e
    
    @staticmethod
    def update_property_of_class(instance_ref:object,property_data:dict):
        try:
            if not isinstance(property_data,dict):
                raise Exception('property_data parameter required to dictionary')

            for key,value in property_data.items():
                setattr(instance_ref,key,value)

            return instance_ref                    
        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        try:
            module=importlib.import_module(module_name)
            return getattr(module,class_name)
        except Exception as e:
            raise SalesException(e,sys) from e


    def execute_grid_search_operation(self,initialized_model:InitializedModelDetail,
                                           input_feature,
                                           output_feature)->GridSearchedBestModel:
        try:
            grid_search_cv_ref=ModelFactory.class_for_name(module_name=self.grid_search_module,
                                                           class_name=self.grid_search_class)
            grid_search_cv=grid_search_cv_ref(estimator=initialized_model.model,
                                              param_grid=initialized_model.param_grid_search)   

            grid_search_cv=ModelFactory.update_property_of_class(instance_ref=grid_search_cv,
                                                                 property_data=self.grid_search_params)

            logging.info(f'Training {type(initialized_model.model).__name__} started')   
            grid_search_cv.fit(input_feature,output_feature)
            logging.info(f'Training {type(initialized_model.model).__name__} completed')  

            grid_searched_best_model=GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number, 
                                  model=initialized_model.model, 
                                  best_model=grid_search_cv.best_estimator_, 
                                  best_parameters=grid_search_cv.best_params_, 
                                  best_score=grid_search_cv.best_score_)    
            return grid_searched_best_model                                                                                                                             
        except Exception as e:
            raise SalesException(e,sys) from e    

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise SalesException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:
        try:
            self.grid_searched_best_model_list=[]
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model=self.initiate_best_parameter_search_for_initialized_model(
                                          initialized_model=initialized_model_list,
                                          input_feature=input_feature,
                                          output_feature=output_feature)

                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list                              
        except Exception as e:
            raise SalesException(e,sys) from e           

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        try:
            initialized_model_list=[]
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config=self.models_initialization_config[model_serial_number]
                model_class=model_initialization_config[CLASS_KEY]
                model_module=model_initialization_config[MODULE_KEY]                
                model_param_grid_search=model_initialization_config[SEARCH_PARAM_GRID_KEY]

                model_obj_ref=ModelFactory.class_for_name(module_name=model_module,
                                                          class_name=model_class)

                model=model_obj_ref()

                if PARAM_KEY in model_initialization_config.keys():
                    model_obj_property_data=model_initialization_config[PARAM_KEY]
                    model=ModelFactory.update_property_of_class(instance_ref=model,
                                                                property_data=model_obj_property_data)
                model_name=f"{model_module}.{model_class}"

                model_initialization_config=InitializedModelDetail(model_serial_number=model_serial_number, 
                                       model=model, 
                                       param_grid_search=model_param_grid_search, 
                                       model_name=model_name)                                                

                initialized_model_list.append(model_initialization_config)
            self.initialized_model_list=initialized_model_list    
            return self.initialized_model_list    
        except Exception as e:
            raise SalesException(e,sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.5
                                                          ) -> BestModel:
        try:
            best_model = None
            print('base_accuracy',base_accuracy)

            for grid_searched_best_model in grid_searched_best_model_list:
                print('model',grid_searched_best_model.best_model)
                print('model_accuracy',grid_searched_best_model.best_score)
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise SalesException(e, sys) from e


    def get_best_model(self, X, y,base_accuracy=0.5) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=0.5)
        except Exception as e:
            raise SalesException(e, sys) from e

